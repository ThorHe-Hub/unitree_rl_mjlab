"""Execute run_play against a CPU fake environment; no mjlab, viewer or checkpoint load.

The physical trace deliberately ignores actions to isolate wiring and reset behavior.
These are integration checks, not robot dynamics or control-performance experiments.
For a staged checkout without src/, set UNITREE_TEST_PROJECT_ROOT to the source repo.
"""
import ast
import contextlib
import hashlib
from dataclasses import asdict, dataclass, field
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Literal
import unittest

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from experiment_record import build_run_metadata, episode_seed_metadata
from agg import DIMS, load_record
from heading_feedback import HeadingFeedback
from contact_settings import configure_contact_friction, contact_record
from pinned_velocity_command import make_pinned_velocity_cfg, UniformVelocityCommandCfg
from sensor_rig import SensorRig, SensorRigCfg


def load_filter(filename, class_name):
    source_root = Path(os.environ.get("UNITREE_TEST_PROJECT_ROOT", str(PROJECT)))
    path = source_root / "src/tasks/velocity/mdp" / filename
    spec = importlib.util.spec_from_file_location("play_test_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


KalmanFilter = load_filter("kalman_filter.py", "KalmanFilter")
AdaptiveKalmanfilter = load_filter("adaptive_kalman_filter.py", "AdaptiveKalmanfilter")


class FakeListener:
    def __init__(self, **kwargs):
        self.stopped = False

    def start(self):
        pass

    def stop(self):
        self.stopped = True


class Suppress:
    def __class_getitem__(cls, item):
        return item


@dataclass
class FakeAgentConfig:
    experiment_name: str = "test"
    clip_actions: float = 1.0


class FakeEnv:
    def __init__(self, cfg, device, render_mode, reset_every=None, timeout_only=False, stability_profile=False):
        self.cfg, self.device = cfg, device
        self.unwrapped = self
        self.step_dt = 0.02
        self.action_space = SimpleNamespace(shape=(1, 2))
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.reset_terminated = torch.zeros(1, dtype=torch.bool)
        self.reset_time_outs = torch.zeros(1, dtype=torch.bool)
        self.num_envs = 1
        self.scene = {"robot": SimpleNamespace(data=SimpleNamespace())}
        # Empty contact geometry represents the CPU wiring fixture, not a physical model.
        self.sim = SimpleNamespace(
            mj_model=SimpleNamespace(ngeom=0, npair=0, geom_friction=np.empty((0, 3))),
            model=SimpleNamespace(geom_friction=torch.empty((0, 3))))
        self.cmd = cfg.commands["twist"].build(self)
        self.command_manager = SimpleNamespace(get_term=lambda name: self.cmd,
                                               get_command=lambda name: self.cmd.command)
        self.seeds, self.source_observations, self.commands = [], [], []
        self.closed = False
        self.reset_every = reset_every
        self.timeout_only = timeout_only
        self.stability_profile = stability_profile
        self.termination_manager = SimpleNamespace(active_terms=["fell_over", "time_out"],
            get_term=lambda name: self.reset_terminated if name == "fell_over" else self.reset_time_outs)
        self.segment = 0

    def seed(self, value):
        self.seeds.append(value)

    def observation(self):
        k = int(self.episode_length_buf[0])
        rate = 0.12 + 0.025 * math.sin(k * 0.3)
        # Distinct origins after automatic resets expose stale reference state.
        yaw = 2.9 - self.segment * 1.2 + 0.0024 * k
        data = self.scene["robot"].data
        data.root_link_quat_w = torch.tensor([[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]], dtype=torch.float64)
        data.root_link_pos_w = torch.tensor([[k * 0.01, k * 0.0001, 0.7]], dtype=torch.float64)
        if self.stability_profile:
            pitch = (0, math.pi / 6, math.pi / 3)[k % 3]
            cy, sy, cp, sp = math.cos(yaw/2), math.sin(yaw/2), math.cos(pitch/2), math.sin(pitch/2)
            data.root_link_quat_w = torch.tensor([[cy*cp, -sy*sp, cy*sp, sy*cp]], dtype=torch.float64)
            data.root_link_pos_w[0, 2] = 0.7 + 0.01 * k
        data.root_link_lin_vel_b = torch.tensor([[0.5, 0, 0]], dtype=torch.float64)
        data.root_link_ang_vel_b = torch.tensor([[0.011 * k, -0.007 * k, rate]], dtype=torch.float64)
        actor = torch.zeros((1, 12), dtype=torch.float64)
        actor[0, 2] = rate
        actor[0, 6:9] = self.cmd.command[0]
        phase = (self.episode_length_buf * self.step_dt) % .6 / .6
        actor[0, 9], actor[0, 10] = torch.sin(phase[0]*torch.pi*2), torch.cos(phase[0]*torch.pi*2)
        if torch.linalg.norm(self.cmd.command[0]) < .1:
            actor[0, 9:11] = 0
        self.source_observations.append(actor)
        return {"actor": actor, "critic": torch.ones((1, 3), dtype=torch.float64)}

    def reset(self):
        self.segment = 0
        self.episode_length_buf.zero_()
        self.reset_terminated.zero_()
        self.reset_time_outs.zero_()
        self.cmd.reset(torch.tensor([0]))
        return self.observation(), {}

    def step(self, actions):
        self.commands.append(self.cmd.vel_command_b.clone().numpy())
        self.episode_length_buf += 1
        done = self.reset_every is not None and int(self.episode_length_buf[0]) == self.reset_every
        self.reset_terminated[0] = done and self.segment % 2 == 0 and not self.timeout_only
        self.reset_time_outs[0] = done and (self.segment % 2 == 1 or self.timeout_only)
        if done:
            self.segment += 1
            self.episode_length_buf.zero_()
            self.cmd.reset(torch.tensor([0]))
        self.cmd.compute(self.step_dt)
        return self.observation(), torch.zeros(1), torch.tensor([done]), {}

    def close(self):
        self.closed = True


def play_namespace(environment_factory, captured, seed_offset=0):
    """Compile actual definitions while excluding simulator/UI import side effects."""
    path = PROJECT / "scripts/play.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = {"TRACE_COLUMNS", "_prbs", "_first_attr", "KeyboardController", "PlayConfig", "run_play"}
    nodes = [node for node in tree.body if getattr(node, "name", None) in names or
             isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)]

    class Runner:
        def __init__(self, env, cfg, device):
            self.env = env

        def load(self, path, **kwargs):
            captured["checkpoint"] = path

        def get_inference_policy(self, device):
            def policy(obs):
                captured["actors"].append(obs["actor"].clone().numpy())
                captured["critics"].append(obs["critic"].clone().numpy())
                return torch.zeros((1, 2))
            return policy

    def metadata(project_dir, checkpoint, config, source_paths):
        captured["config"] = config
        return {"run_id": "fake-run", "checkpoint": str(checkpoint),
                "config_json": json.dumps(config, default=str),
                **{k: v for k, v in config.items() if isinstance(v, (str, bool, int, float))}}

    def env_config(*args, **kwargs):
        return SimpleNamespace(scene=SimpleNamespace(terrain=SimpleNamespace()),
                               sim=SimpleNamespace(nconmax=500), commands={"twist": UniformVelocityCommandCfg(
                                   entity_name="robot", resampling_time_range=(.04, .06),
                                   heading_command=True, rel_standing_envs=1.0,
                                   ranges=UniformVelocityCommandCfg.Ranges((-1,1),(-1,1),(-1,1),(-3.14,3.14)))},
                               episode_length_s=1e9,
                               terminations={"fell_over": SimpleNamespace(time_out=False, params={"limit_angle": math.radians(70)}),
                                             "time_out": SimpleNamespace(time_out=True, params={})},
                               events={key: object() for key in ("foot_friction", "encoder_bias", "base_com")})

    namespace = dict(globals())
    namespace.update(__file__=str(path),
        tyro=SimpleNamespace(conf=SimpleNamespace(Suppress=Suppress)),
        keyboard=SimpleNamespace(Listener=FakeListener),
        configure_torch_backends=lambda: None,
        load_env_cfg=env_config, load_rl_cfg=lambda task: FakeAgentConfig(),
        ManagerBasedRlEnv=environment_factory,
        RslRlVecEnvWrapper=lambda env, **kwargs: env,
        load_runner_cls=lambda task: Runner, MjlabOnPolicyRunner=Runner,
        build_run_metadata=metadata,
        episode_seed_metadata=lambda seed, index: episode_seed_metadata(seed, index + seed_offset))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class PlayFeedbackIntegrationTests(unittest.TestCase):
    def run_case(self, filt="fusion", mode="integrated", repeats=1, steps=12,
                 reset_every=None, seed_offset=0, real_metadata=False,
                 stop_on_done=False, timeout_only=False, stability_profile=False, **cfg_extra):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        directory = Path(temp.name)
        checkpoint = directory / "model_12000.pt"
        checkpoint.write_bytes(b"fake checkpoint; runner loading is stubbed")
        captured = {"actors": [], "critics": []}

        def factory(**kwargs):
            captured["env"] = FakeEnv(**kwargs, reset_every=reset_every, timeout_only=timeout_only,
                                      stability_profile=stability_profile)
            return captured["env"]

        ns = play_namespace(factory, captured, seed_offset)
        if real_metadata:
            ns["build_run_metadata"] = build_run_metadata
        cfg = ns["PlayConfig"](checkpoint_file=str(checkpoint), device="cpu",
            batch_steps=steps, batch_stop_on_done=stop_on_done, repeats=repeats, pin_cmd=True, fixed_body=True,
            friction=0.8, seed=73, kp=0.91, ki=0.02, kd=0.01,
            rig=cfg_extra.pop("rig", None) or SensorRigCfg(filt=filt, noise_std=0.03, gyro_bias=0.02,
                             abs_sigma=0.02, abs_hz=25),
            feedback_mode=mode, output_dir=directory / "traces", **cfg_extra)
        with contextlib.redirect_stdout(io.StringIO()):
            ns["run_play"]("Fake-Velocity-Task", cfg)
        traces = []
        for path in sorted(cfg.output_dir.glob("*.npz")):
            with np.load(path, allow_pickle=False) as saved:
                traces.append({key: saved[key].copy() for key in saved.files})
        captured.update(cfg=cfg, traces=traces, columns=ns["TRACE_COLUMNS"],
                        trace_paths=sorted(cfg.output_dir.glob("*.npz")))
        return captured

    def test_stop_on_fall_keeps_failure_and_next_repetition_separate(self):
        run = self.run_case(repeats=2, steps=12, reset_every=3, stop_on_done=True)
        self.assertEqual(len(run["traces"]), 2)
        for trace in run["traces"]:
            self.assertEqual(trace["data"].shape, (3, 49))
            self.assertEqual(trace["completed_duration_s"].item(), 0.06)
            self.assertEqual(trace["stop_reason"].item(), "terminated")
            self.assertEqual(trace["segments"].item(), 1)
            self.assertEqual(trace["requested_steps"].item(), 12)
            event = json.loads(trace["termination_events_json"].item())[0]
            self.assertEqual(event, {"step": 3, "terminated": True, "truncated": False, "terms": ["fell_over"]})
            np.testing.assert_array_equal(trace["data"][:, run["columns"].index("terminated_after_step")], [0, 0, 1])
            np.testing.assert_allclose(trace["data"][:, run["columns"].index("base_height_w")], 0.7)

    def test_timeout_is_distinguished_from_fall_and_budget_completion(self):
        run = self.run_case(steps=12, reset_every=3, stop_on_done=True, timeout_only=True)
        trace = run["traces"][0]
        self.assertEqual(trace["stop_reason"].item(), "time_out")
        self.assertFalse(json.loads(trace["termination_events_json"].item())[0]["terminated"])
        complete = self.run_case(steps=12, stop_on_done=True)["traces"][0]
        self.assertEqual(complete["stop_reason"].item(), "budget_reached")
        self.assertEqual(complete["completed_duration_s"].item(), 0.24)
        self.assertEqual(complete["termination_events_json"].item(), "[]")

    def test_stability_is_logged_before_the_step_and_does_not_replace_done_flags(self):
        run = self.run_case(steps=6, stability_profile=True, stop_on_done=True)
        data = run["traces"][0]["data"]
        col = lambda name: data[:, run["columns"].index(name)]
        np.testing.assert_allclose(col("base_tilt"), np.tile([0, math.pi/6, math.pi/3], 2), atol=1e-14)
        np.testing.assert_allclose(col("base_height_w"), .7 + .01*np.arange(6), atol=1e-14)
        np.testing.assert_array_equal(col("terminated_after_step"), np.zeros(6))
        np.testing.assert_array_equal(col("truncated_after_step"), np.zeros(6))
        self.assertEqual(run["traces"][0]["stability_timing"].item(), "pre_physics_step_same_as_gt_yaw")
        # schema v5：同一四元数的 ZYX 横滚/俯仰，与机体 x/y 角速度同步记录
        np.testing.assert_allclose(col("gt_pitch"), np.tile([0, math.pi/6, math.pi/3], 2), atol=1e-12)
        np.testing.assert_allclose(col("gt_roll"), np.zeros(6), atol=1e-12)
        k = np.arange(6)
        np.testing.assert_allclose(col("gt_wx"), 0.011 * k, atol=1e-15)
        np.testing.assert_allclose(col("gt_wy"), -0.007 * k, atol=1e-15)
        # 俯仰不改变 ZYX 航向：gt_yaw 仍是生成用的 yaw（缠绕到 ±π）
        yaw = np.angle(np.exp(1j * (2.9 + 0.0024 * k)))
        np.testing.assert_allclose(col("gt_yaw"), yaw, atol=1e-12)

    def test_interval_gyro_feeds_estimator_but_not_policy(self):
        point = self.run_case(mode="fusion_heading")
        inter = self.run_case(mode="fusion_heading",
            rig=SensorRigCfg(filt="fusion", noise_std=0.03, gyro_bias=0.02, abs_sigma=0.02,
                             abs_hz=25, gyro_signal="interval"))
        col = lambda run, name: run["traces"][0]["data"][:, run["columns"].index(name)]
        # 假环境的 yaw 每步增加 0.0024 rad，区间平均 z 角速度恒为 0.12 rad/s；段首退回点采样
        np.testing.assert_allclose(col(inter, "raw_yaw_interval")[1:], 0.12, atol=1e-9)
        self.assertEqual(col(inter, "raw_yaw_interval")[0], col(inter, "raw_yaw")[0])
        # 两种模式点采样相同；interval 模式下，估计器侧与策略侧只差真值部分
        np.testing.assert_array_equal(col(point, "raw_yaw"), col(inter, "raw_yaw"))
        np.testing.assert_allclose(col(inter, "used_yaw_feedback") - col(inter, "used_yaw"),
                                   col(inter, "raw_yaw_interval") - col(inter, "raw_yaw"), atol=1e-12)
        np.testing.assert_allclose(col(inter, "integrated_yaw"), np.cumsum(col(inter, "used_yaw_feedback")) * 0.02)
        self.assertGreater(np.abs(col(inter, "fusion_heading") - col(point, "fusion_heading")).max(), 1e-6)
        # 策略观测第 2 通道始终是点采样扣零偏
        np.testing.assert_array_equal(np.concatenate(inter["actors"])[:, 2], col(inter, "used_yaw"))

    def test_vx_step_and_vyaw_pulse_reach_the_environment(self):
        run = self.run_case(steps=12, pin_vx_step_s=0.1, pin_vx_after=0.8,
                            vyaw_pulse=0.3, vyaw_pulse_at_s=0.06, vyaw_pulse_s=0.08)
        data = run["traces"][0]["data"]
        col = lambda name: data[:, run["columns"].index(name)]
        t = np.arange(12) * 0.02
        np.testing.assert_allclose(col("environment_cmd_vx"), np.where(t >= 0.1 - 1e-9, 0.8, 0.5), atol=1e-7)
        pulse = ((t >= 0.06 - 1e-9) & (t < 0.14 - 1e-9)) * 0.3
        np.testing.assert_allclose(col("cmd_vyaw") - col("pid_correction"), pulse, atol=1e-12)
        np.testing.assert_allclose(col("environment_cmd_vyaw"), col("cmd_vyaw"), atol=1e-7)
        with self.assertRaises(ValueError):
            self.run_case(vyaw_pulse=0.3, vyaw_pulse_s=-1.0)

    def test_real_metadata_round_trip_into_aggregation(self):
        run = self.run_case(mode="fusion_heading", real_metadata=True)
        record = load_record(run["trace_paths"][0])
        meta = record["metadata"]
        self.assertEqual(record["cohort"], "recorded")
        self.assertEqual(record["missing_metadata"], [])
        self.assertEqual(record["assumed_defaults"], {})
        self.assertEqual(meta["checkpoint_sha256"], hashlib.sha256(
            Path(run["cfg"].checkpoint_file).read_bytes()).hexdigest())
        self.assertEqual(meta["dt"], 0.02)
        self.assertEqual(meta["noise_std"], 0.03)
        self.assertEqual(meta["kd"], 0.01)
        self.assertEqual(meta["reference_delay_steps"], 1)
        self.assertEqual(record["key"][DIMS.index("反馈方式")], "fusion_heading")
        self.assertEqual(record["key"][DIMS.index("源代码SHA256")], meta["source_tree_sha256"])
        self.assertTrue(np.isfinite(record["metrics"]).all())
        self.assertEqual(json.loads(meta["config_json"])["rig"]["abs_hz"], 25)

    def test_three_modes_save_schema_and_actual_pre_update_feedback(self):
        for filt, mode in (("dirty", "integrated"), ("fusion", "integrated"), ("fusion", "fusion_heading")):
            with self.subTest(filt=filt, mode=mode):
                run = self.run_case(filt, mode)
                trace = run["traces"][0]
                data = trace["data"]
                col = lambda name: data[:, run["columns"].index(name)]
                self.assertEqual(data.shape, (12, 49))
                self.assertEqual(trace["columns"].tolist(), run["columns"])
                self.assertEqual(trace["trace_schema_version"].item(), 6)
                self.assertEqual(trace["completed_steps"].item(), 12)
                self.assertEqual(trace["feedback_mode"].item(), mode)
                self.assertEqual(trace["collection_mode"].item(), "batch")
                self.assertTrue(trace["fixed_body"].item())
                self.assertEqual(Path(trace["checkpoint"].item()).name, "model_12000.pt")
                self.assertEqual(run["config"]["feedback_timing"], "pre_sensor_update")
                self.assertEqual(run["config"]["pid_feedback_delay_steps"], 1)
                self.assertEqual(col("pid_feedback_yaw")[0], 0.0)
                np.testing.assert_array_equal(col("pid_feedback_yaw")[1:], col("feedback_heading")[:-1])
                source = "integrated_yaw" if mode == "integrated" else "fusion_heading"
                np.testing.assert_array_equal(col("feedback_heading"), col(source))
                np.testing.assert_allclose(col("integrated_yaw"), np.cumsum(col("used_yaw_feedback")) * 0.02)
                np.testing.assert_array_equal(col("used_yaw_feedback"), col("used_yaw"))   # point 模式
                if filt == "dirty":
                    np.testing.assert_array_equal(col("used_yaw"), col("dirty_yaw"))
                    self.assertTrue(np.isnan(col("fusion_heading")).all())
                else:
                    np.testing.assert_allclose(col("used_yaw"), col("dirty_yaw") - col("fusion_bias"))
                error = col("pid_target_yaw") - col("pid_feedback_yaw")
                integral = np.clip(np.cumsum(error) * 0.02, -0.15, 0.15)
                expected = np.clip(0.91 * error + 0.02 * integral + 0.01 * np.diff(np.r_[0, error]) / 0.02, -1.6, 1.6)
                np.testing.assert_allclose(col("pid_correction"), expected, atol=1e-12)
                np.testing.assert_array_equal(col("cmd_vyaw"), col("pid_correction"))
                np.testing.assert_allclose(np.concatenate(run["env"].commands)[:, 2], col("cmd_vyaw"), atol=1e-7)
                self.assertTrue(run["env"].closed)

    def test_fusion_modes_keep_actor_sensor_channels_identical(self):
        integral = self.run_case(mode="integrated")
        direct = self.run_case(mode="fusion_heading")
        actors_i, actors_d = np.concatenate(integral["actors"]), np.concatenate(direct["actors"])
        # Column 8 is the PID command and is expected to change with feedback.
        keep = [index for index in range(actors_i.shape[1]) if index != 8]
        np.testing.assert_array_equal(actors_i[:, keep], actors_d[:, keep])
        self.assertGreater(np.max(np.abs(actors_i[:, 8] - actors_d[:, 8])), 1e-5)
        for run in (integral, direct):
            data = run["traces"][0]["data"]
            np.testing.assert_array_equal(np.concatenate(run["actors"])[:, 2], data[:, 18])
            original_rates = [obs[0, 2].item() for obs in run["env"].source_observations[:-1]]
            np.testing.assert_array_equal(original_rates, data[:, 0])
            np.testing.assert_array_equal(np.concatenate(run["critics"]), np.ones((12, 3)))

    def test_every_repetition_matches_fresh_run_with_its_own_seed(self):
        for mode in ("integrated", "fusion_heading"):
            with self.subTest(mode=mode):
                repeated = self.run_case(mode=mode, repeats=2)
                fresh = self.run_case(mode=mode, seed_offset=1)
                np.testing.assert_array_equal(repeated["traces"][1]["data"], fresh["traces"][0]["data"])
                self.assertEqual(repeated["env"].seeds,
                                 [episode_seed_metadata(73, index)["seed_env"] for index in (0, 1)])
                for index, trace in enumerate(repeated["traces"]):
                    self.assertEqual(trace["episode_index"].item(), index)
                    self.assertEqual(trace["segments"].item(), 1)
                    self.assertEqual(trace["data"][0, 21], 0.0)
                    self.assertEqual(trace["data"][0, 23], 0.0)
                    self.assertEqual(trace["data"][0, 25], 0.0)

    def test_automatic_resets_split_segments_and_clear_control_and_filter_state(self):
        run = self.run_case(mode="fusion_heading", steps=11, reset_every=4)
        trace, columns = run["traces"][0], run["columns"]
        data = trace["data"]
        col = lambda name: data[:, columns.index(name)]
        np.testing.assert_array_equal(col("segment_id"), [0] * 4 + [1] * 4 + [2] * 3)
        self.assertEqual(trace["segments"].item(), 3)
        np.testing.assert_array_equal(col("terminated_after_step"), [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(col("truncated_after_step"), [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        for start in (0, 4, 8):
            for name in ("pid_feedback_yaw", "pid_target_yaw", "pid_correction", "fusion_bias", "reference_updated", "reference_rejected"):
                self.assertEqual(col(name)[start], 0.0, (start, name))
            self.assertAlmostEqual(col("integrated_yaw")[start], col("used_yaw")[start] * 0.02)
            self.assertAlmostEqual(col("fusion_heading")[start], col("dirty_yaw")[start] * 0.02)
            # Genuine filters must restart too, with no previous-segment state.
            kf, akf = KalmanFilter(dt=0.02), AdaptiveKalmanfilter(dt=0.02)
            kf.predict()
            akf.predict()
            self.assertAlmostEqual(col("filtered_vx_kf")[start], kf.update(col("dirty_vx")[start]))
            self.assertAlmostEqual(col("filtered_vx_akf")[start], akf.update(col("dirty_vx")[start]))
            kf, akf = KalmanFilter(dt=0.02), AdaptiveKalmanfilter(dt=0.02)
            kf.predict()
            akf.predict()
            self.assertAlmostEqual(col("filtered_yaw_kf")[start], kf.update(col("dirty_yaw")[start]))
            self.assertAlmostEqual(col("filtered_yaw_akf")[start], akf.update_for_yaw(col("dirty_yaw")[start]))
        for index in (1, 2, 3, 5, 6, 7, 9, 10):
            self.assertEqual(col("pid_feedback_yaw")[index], col("feedback_heading")[index - 1])


if __name__ == "__main__":
    unittest.main()
