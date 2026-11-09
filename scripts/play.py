import copy
import json
import os
import sys
import time
from dataclasses import asdict, dataclass,field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from pynput import keyboard

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from src.tasks.velocity.mdp.adaptive_kalman_filter import AdaptiveKalmanfilter
from src.tasks.velocity.mdp.kalman_filter import KalmanFilter
from sensor_rig import SensorRig,SensorRigCfg
from heading_feedback import HeadingFeedback
from contact_settings import configure_contact_friction, contact_record
from pinned_velocity_command import make_pinned_velocity_cfg
from experiment_record import build_run_metadata, episode_seed_metadata
# trace .npz 的列名，顺序必须和 global_velocity_log.append 一致
TRACE_COLUMNS = [
    "raw_yaw", "dirty_yaw", "filtered_yaw_kf", "filtered_yaw_akf",
    "raw_vx", "dirty_vx", "filtered_vx_kf", "filtered_vx_akf",
    "pid_correction", "akf_R",
    "gt_yaw", "gt_x", "gt_y", "gt_wz",
    "phase_sin", "phase_cos",
    "integrated_yaw", "cmd_vyaw",  # historical first 18 columns retained
    "used_yaw", "fusion_heading", "fusion_bias", "pid_feedback_yaw",
    "feedback_heading", "pid_target_yaw", "pid_active", "reference_updated",
    "reference_rejected", "segment_id", "terminated_after_step", "truncated_after_step",
    "base_tilt", "base_height_w",  # pre-step root-link tilt [rad] and world-z [m]
    "observation_cmd_vx", "observation_cmd_vy", "observation_cmd_vyaw",
    "actor_cmd_vx", "actor_cmd_vy", "actor_cmd_vyaw",
    "environment_cmd_vx", "environment_cmd_vy", "environment_cmd_vyaw",
    "observation_standing", "environment_standing",
    # schema v5: root-link roll/pitch (ZYX, rad) and body-frame x/y angular rate (rad/s),
    # all read at the same instant as gt_yaw / gt_wz. Needed to test the gait kinematic term.
    "gt_roll", "gt_pitch", "gt_wx", "gt_wy",
    # schema v6: interval-mean body z rate from consecutive root-link quaternions (equals raw_yaw
    # on the first step of a segment), and the bias-corrected rate that drives the PID integrator.
    "raw_yaw_interval", "used_yaw_feedback",
]


def _prbs(n, amp, seed, lo=5, hi=40):
    """伪随机二值激励。

    二值给延迟辨识提供陡沿；保持步数在 [lo, hi] 内随机，
    使低频（定直流增益）与高频（定时间常数）都有能量。
    """
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        out += [amp if rng.random() < 0.5 else -amp] * int(rng.integers(lo, hi + 1))
    return np.array(out[:n], dtype=float)


def _first_attr(obj, *names):
    """按顺序返回第一个存在的属性，都没有则 None（各版本 mjlab 命名不一致）。"""
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


class KeyboardController:
    def __init__(self):
        self.vx, self.vy, self.vyaw = 0.0, 0.0, 0.0
        self.speed = 1.0  
        self.yaw_trend = 0.0

        # PID 参数定型组合
        self.Kp = 0.91   
        self.Ki = 0.000   
        self.Kd = 0.01   
                      
        self.error_integral = 0.0  
        self.last_error = 0.0      
        
        self.yaw_target = 0.0
        self.integrated_yaw = 0.0
        self.last_printed_v = [0.0, 0.0, 0.0] 
        
        self.active_keys = set()
        self.is_kb_active = False 
        self.is_turning = True 
        
        # 键盘控制使能锁，默认关闭
        self.is_enabled = False 
        
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()

    def _update_velocities(self):
        self.is_kb_active = len(self.active_keys) > 0 
        self.vx, self.vy, self.vyaw = 0.0, 0.0, 0.0
        
        if 'i' in self.active_keys: self.vx += 1.0
        if 'k' in self.active_keys: self.vx -= 1.0
        if 'j' in self.active_keys: self.vy += 0.8
        if 'l' in self.active_keys: self.vy -= 0.8
        if 'u' in self.active_keys: self.vyaw += 1.0
        if 'o' in self.active_keys: self.vyaw -= 1.0

    def on_press(self, key):
        # 使用空格键作为全局控制权切换开关，不干扰仿真器视角控制
        if key == keyboard.Key.space:
            self.is_enabled = not self.is_enabled
            state = "开启 (Keyboard)" if self.is_enabled else "关闭 (UI Panel)"
            print(f"\n[INFO]: 控制权限已切换为 -> {state}")
            return 

        try:
            if hasattr(key, 'char') and key.char is not None:
                char = key.char.lower() 
                self.active_keys.add(char)
                self._update_velocities()
        except AttributeError:
            pass

    def on_release(self, key):
        if key == keyboard.Key.space:
            return

        try:
            if hasattr(key, 'char') and key.char is not None:
                char = key.char.lower()
                if char in self.active_keys:
                    self.active_keys.remove(char)
                self._update_velocities()
        except AttributeError:
            pass


@dataclass(frozen=True)
class PlayConfig:
    agent: Literal["zero", "random", "trained"] = "trained"
    checkpoint_file: str | None = None
    registry_name: str | None = None
    wandb_run_path: str | None = None
    motion_file: str | None = None
    num_envs: int | None = None
    device: str | None = None
    video: bool = False
    video_length: int = 200
    video_height: int | None = None
    video_width: int | None = None
    camera: int | str | None = None
    viewer: Literal["auto", "native", "viser"] = "viser"
    no_terminations: bool = False
    pin_cmd: bool = False    # 钉死指令 vx=pin_vx, vy=0, vyaw=0：测航向锁定必开，否则环境会随机发转向指令
    pin_vx: float = 0.5      # 钉死时的前进速度
    friction: float | None = None  # 覆盖地面摩擦；play 默认被设成 0.05（≈冰面），要测控制得调回 0.6~0.8
    batch_steps: int | None = None  # 无头批量：每轮跑 N 步，不开 viewer
    batch_stop_on_done: bool = False
    """批量评估遇到首次终止或超时即保存并结束该轮，避免把重生轨迹拼入同次实验。"""
    repeats: int = 1                # 重复几轮，每轮 env.reset() 重抽地形和初始朝向
    fixed_body: bool = False        # 关掉 startup 随机化，让所有实验条件共用同一个机体
    sweep_vyaw: float = 0.0  # >0 开启**开环辨识**：PID 全程关闭，vyaw 注入 ±该幅值的伪随机
                             # 二值激励。闭环轨迹里 cmd_vyaw 就是 PID 的输出（实测相关系数 1.0），
                             # 输入完全由输出决定，被控对象在原理上不可辨识——必须有独立于反馈的激励。
    sweep_seed: int = 0      # 激励序列的随机种子，换种子即换一条独立实现
    kp: float | None = None         # 覆盖航向 PID 的比例增益（默认 0.91）
    ki: float | None = None         # 覆盖积分增益（默认 0）；error_integral 限幅 ±0.15。
    contact_friction: float | None = None
    """Actual G1 foot-ground sliding coefficient. None preserves the compiled asset defaults.
    Requires --fixed-body so startup foot-friction randomization cannot overwrite it."""
    rig: SensorRigCfg = field(default_factory=SensorRigCfg)
    """传感器加噪及滤波参数，命令行使用 --rig.*。"""
    feedback_mode: Literal["integrated", "fusion_heading", "fusion_offset_smoothed"] = "integrated"
    """PID 航向反馈：角速度独立积分，或 fusion 的航向后验估计。两者均在下一采样步使用。"""
    feedback_offset_tau: float = 0.1
    """fusion_offset_smoothed 中 f−i 的低通时间常数 [s]；其他模式不使用。默认值与原平滑批次一致。"""
    pin_vx_step_s: float = 0.0
    """>0 时，在回合内该时刻 [s] 把钉死的 vx 从 pin_vx 切到 pin_vx_after（需 --pin-cmd）。0 = 不切换。"""
    pin_vx_after: float = 0.5
    vyaw_pulse: float = 0.0
    """在 PID 输出上叠加的 vyaw 脉冲幅值 [rad/s]；0 = 关闭。用于测固定策略对指令阶跃的承受度。"""
    vyaw_pulse_at_s: float = 0.0
    vyaw_pulse_s: float = 0.0
    """脉冲起点与持续时间 [s]；脉冲作用于 [at, at+duration)。"""
    seed: int = 0  # Independent environment/gyro/reference/dropout/vx streams per repetition.
    kd: float | None = None
    output_dir: Path = Path("traces")
    _demo_mode: tyro.conf.Suppress[bool] = False


def run_play(task_id: str, cfg: PlayConfig):
    feedback = HeadingFeedback(cfg.feedback_mode, cfg.rig.filt, cfg.feedback_offset_tau)
    if cfg.seed < 0 or cfg.repeats < 1 or (cfg.batch_steps is not None and cfg.batch_steps < 1):
        raise ValueError("seed must be nonnegative; repeats and batch_steps must be positive")
    if not all(np.isfinite(v) for v in (cfg.pin_vx_step_s, cfg.pin_vx_after, cfg.vyaw_pulse,
                                        cfg.vyaw_pulse_at_s, cfg.vyaw_pulse_s)) \
            or min(cfg.pin_vx_step_s, cfg.vyaw_pulse_at_s, cfg.vyaw_pulse_s) < 0:
        raise ValueError("vx step / vyaw pulse times must be finite and nonnegative")
    if (cfg.pin_vx_step_s > 0 or cfg.vyaw_pulse != 0) and not cfg.pin_cmd:
        raise ValueError("vx step and vyaw pulse require --pin-cmd")
    # Validate sensor rates before loading the simulator/checkpoint.
    SensorRig(cfg.rig)
    episode_info = episode_seed_metadata(cfg.seed, 0)
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.seed = episode_info["seed_env"]
    if cfg.friction is not None:
        env_cfg.scene.terrain.static_friction = cfg.friction
        env_cfg.scene.terrain.dynamic_friction = cfg.friction
        print(f"[INFO]: legacy friction request={cfg.friction}; use --contact-friction for actual contact coefficients")
    if getattr(env_cfg.sim, "nconmax", None) is not None and env_cfg.sim.nconmax < 500:
        print(f"[INFO]: nconmax {env_cfg.sim.nconmax} -> 500（Rough 地形随机块接触点会超）")
        env_cfg.sim.nconmax = 500
    if cfg.fixed_body:
        # foot_friction(0.27~1.5) / encoder_bias / base_com 都是 mode="startup"，
        # 一个进程只抽一次。不同实验条件跑在不同进程里 = 每个条件配一个不同的机体，
        # 条件间比较会被机体差异污染（脚底摩擦 5.5 倍跨度足以盖过滤波器效应）。
        # 比滤波器就得钉死机体；代价是结论只对这一台标称机器成立。
        for _e in ("foot_friction", "encoder_bias", "base_com"):
            if env_cfg.events.pop(_e, None) is not None:
                print(f"[INFO]: 关掉 startup 随机化 {_e}")
    if cfg.contact_friction is not None and not cfg.fixed_body:
        raise ValueError("contact_friction requires fixed_body to prevent startup randomization")
    configure_contact_friction(env_cfg.scene, cfg.contact_friction)
    if cfg.pin_cmd:
        env_cfg.commands["twist"] = make_pinned_velocity_cfg(
            env_cfg.commands.get("twist"), cfg.pin_vx)
    agent_cfg = load_rl_cfg(task_id)

    DUMMY_MODE = cfg.agent in {"zero", "random"}
    TRAINED_MODE = not DUMMY_MODE

    if cfg.no_terminations:
        env_cfg.terminations = {}
        print("[INFO]: Terminations disabled")

    is_tracking_task = "motion" in env_cfg.commands and isinstance(
        env_cfg.commands["motion"], MotionCommandCfg
    )

    if is_tracking_task and cfg._demo_mode:
        motion_cmd = env_cfg.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)
        motion_cmd.sampling_mode = "uniform"

    if is_tracking_task:
        motion_cmd = env_cfg.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)

        if cfg.motion_file is not None and Path(cfg.motion_file).exists():
            print(f"[INFO]: Using local motion file: {cfg.motion_file}")
            motion_cmd.motion_file = cfg.motion_file
        elif DUMMY_MODE:
            if not cfg.registry_name:
                raise ValueError("Tracking tasks require either --motion-file or --registry-name")

    log_dir: Path | None = None
    resume_path: Path | None = None
    if TRAINED_MODE:
        log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
        if cfg.checkpoint_file is not None:
            resume_path = Path(cfg.checkpoint_file)
            if not resume_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
            print(f"[INFO]: Loading checkpoint: {resume_path.name}")
        else:
            if cfg.wandb_run_path is None:
                raise ValueError("`wandb_run_path` is required when `checkpoint_file` is not provided.")
            resume_path, was_cached = get_wandb_checkpoint_path(log_root_path, Path(cfg.wandb_run_path))
            run_id = resume_path.parent.name
            checkpoint_name = resume_path.name
            cached_str = "cached" if was_cached else "downloaded"
            print(f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})")
        log_dir = resume_path.parent

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs
        
    env_cfg.scene.num_envs = 1

    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if TRAINED_MODE and cfg.video:
        print("[INFO] Recording videos during play")
        assert log_dir is not None
        env = VideoRecorder(
            env,
            video_folder=log_dir / "videos" / "play",
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    control_dt = float(env.unwrapped.step_dt)
    if not np.isclose(control_dt, 0.02, rtol=0, atol=1e-9):
        env.close()
        raise ValueError(f"Current filter/trace convention requires dt=0.02, got {control_dt}")

    if DUMMY_MODE:
        action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
        if cfg.agent == "zero":
            class PolicyZero:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return torch.zeros(action_shape, device=env.unwrapped.device)
            policy = PolicyZero()
        else:
            class PolicyRandom:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1
            policy = PolicyRandom()
    else:
        runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)

    controller = KeyboardController()
    if cfg.kp is not None:
        controller.Kp = cfg.kp
    if cfg.ki is not None:
        controller.Ki = cfg.ki
    if cfg.kd is not None:
        controller.Kd = cfg.kd
    global_velocity_log = []
    base_policy = policy
    kf_yaw_rate = kf_vx_rate = ad_kf_vx_lite = ad_kf_yaw_lite = None
    episode_rngs = {}
    project_dir = Path(__file__).resolve().parents[1]
    config_record = {**asdict(cfg), **asdict(cfg.rig), "task_id": task_id,
                     "contact_audit_json": json.dumps(contact_record(env.unwrapped.sim, cfg.contact_friction), sort_keys=True),
                     "kp": controller.Kp, "ki": controller.Ki, "kd": controller.Kd,
                     "dt": control_dt, "trace_schema_version": 6,
                     "command_input_mode": "source_held_before_observation" if cfg.pin_cmd else "interactive",
                     "command_input_timing": "observation_previous_held_yaw_then_actor_and_environment_current_pid",
                     "standing_threshold": 0.1,
                     "resolved_twist_config": asdict(env_cfg.commands["twist"]) if cfg.pin_cmd else None,
                     "stability_timing": "pre_physics_step_same_as_gt_yaw",
                     "base_tilt_definition": "angle_between_root_link_z_and_world_vertical_rad",
                     "base_height_definition": "root_link_world_z_m_not_total_robot_height",
                     "pid_feedback_delay_steps": 1, "reference_delay_steps": 1,
                     "reference_timing": "previous_physics_sample",
                     "reference_kind": "relative_truth_plus_independent_gaussian_noise",
                     "feedback_timing": "pre_sensor_update",
                     "episode_length_s": env_cfg.episode_length_s,
                     "termination_terms": {name: {"time_out": term.time_out,
                         "params_repr": repr(term.params)}
                         for name, term in env_cfg.terminations.items() if term is not None},
                     "effective_abs_hz": SensorRig(cfg.rig).effective_reference_hz}
    run_metadata = build_run_metadata(
        project_dir, resume_path, config_record,
        source_paths=["scripts/play.py", "sensor_rig.py", "heading_fusion.py",
                      "heading_feedback.py", "experiment_record.py", "pinned_velocity_command.py", "contact_settings.py", "src"])
    # Keep every analysis parameter as a scalar as well as the full config JSON.
    run_metadata.update({k: v for k, v in config_record.items()
                         if isinstance(v, (str, bool, int, float))})
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    def teleop_policy(obs):
        # A simulator reset invalidates the heading origin, filter states and PID.
        episode_step = int(env.unwrapped.episode_length_buf[0].item())
        previous_step = getattr(teleop_policy, "last_episode_step", None)
        if previous_step is not None and episode_step <= previous_step:
            reset_policy_state(teleop_policy.segment_id + 1)
        teleop_policy.last_episode_step = episode_step
        if not hasattr(teleop_policy, 'step_counter'):
            teleop_policy.step_counter = 0
        teleop_policy.step_counter += 1

        dt = 0.02

        # ========================================================
        #  1：获取原始输入张量
        # ========================================================
        orig_tensor = None
        if isinstance(obs, torch.Tensor):
            orig_tensor = obs
        elif hasattr(obs, "keys"):
            for k in ["actor", "policy", "obs"]: 
                if k in obs.keys():
                    orig_tensor = obs[k]
                    break
            if orig_tensor is None: 
                for k in obs.keys():
                    if k != "critic" and isinstance(obs[k], torch.Tensor):
                        orig_tensor = obs[k]
                        break

        # ========================================================
        # 2：控制权限划分与隔离墙基础赋值
        # ========================================================
        observation_cmd = orig_tensor[0, 6:9].detach().cpu().tolist() if orig_tensor is not None else [np.nan]*3
        cmd_term = None
        observation_standing = np.nan
        try:
            cmd_term = env.unwrapped.command_manager.get_term("twist")
            observation_standing = float(cmd_term.is_standing_env[0].item())
        except (KeyError, AttributeError):
            if cfg.pin_cmd:
                raise RuntimeError("Pinned command diagnostics unavailable")
        is_keyboard_active = controller.is_enabled and (
            controller.is_kb_active or (not controller.is_turning and abs(controller.last_error) > 0.01)
        )

        if is_keyboard_active:
            # 分支 A：键盘控制接管
            base_vx = controller.vx * controller.speed
            base_vy = controller.vy * controller.speed
            base_vyaw = controller.vyaw * controller.speed
            is_user_turning = abs(base_vyaw) > 0.1 
        else:
            # 分支 B：网页面板完全接管 (支持滑动条指令下发)
            if orig_tensor is not None:
                base_vx = orig_tensor[0, 6].item()
                base_vy = orig_tensor[0, 7].item()
                panel_vyaw = orig_tensor[0, 8].item()
            else:
                base_vx, base_vy, panel_vyaw = 0.0, 0.0, 0.0

            base_vyaw = panel_vyaw
            is_user_turning = abs(panel_vyaw) > 0.1
        # 钉死指令：环境的 twist 会随机重采转向，那样 PID 有近 40% 时间是关的，
        # 航向锁定根本测不出来。开 --pin-cmd 后走直线，PID 全程在线。
        if cfg.pin_cmd:
            # 本段内第 k 个控制步的时刻（首步为 0），与 SensorRig 的采样计数一致。
            t_seg = (teleop_policy.step_counter - 1) * control_dt
            pin_vx_now = cfg.pin_vx_after if (cfg.pin_vx_step_s > 0 and t_seg >= cfg.pin_vx_step_s - 1e-9) else cfg.pin_vx
            base_vx, base_vy, base_vyaw = pin_vx_now, 0.0, 0.0
            is_user_turning = False
            if teleop_policy.step_counter == 1:
                print(f"[PIN] 指令钉死: vx={cfg.pin_vx} vy=0 vyaw=0，PID 全程激活")
        # 开环辨识模式：PID 关闭，vyaw 走已知激励。必须放在 pin_cmd 之后覆盖它。
        if cfg.sweep_vyaw > 0:
            if not hasattr(teleop_policy, "sweep"):
                teleop_policy.sweep = _prbs(max(cfg.batch_steps or 3000, 3000),
                                            cfg.sweep_vyaw, cfg.sweep_seed)
                print(f"[SWEEP] 开环辨识：vyaw 注入 ±{cfg.sweep_vyaw} rad/s 伪随机二值激励"
                      f"（seed={cfg.sweep_seed}），PID 全程关闭")
            base_vx, base_vy = cfg.pin_vx, 0.0
            base_vyaw = float(teleop_policy.sweep[teleop_policy.step_counter
                                                  % len(teleop_policy.sweep)])
            is_user_turning = True      # 走 else 分支：PID 不介入，指令原样下发
        # ========================================================
        #  3：统一 PID 航向锁定状态机
        # ========================================================
        pid_feedback_yaw = feedback.value
        correction_cmd = feedback.command(controller, is_user_turning, dt)
        if not is_user_turning:
            base_vyaw = correction_cmd
        # 指令阶跃承受度测试：在 PID 输出上叠加一段已知幅值的 vyaw 脉冲。
        # PID 看得到脉冲造成的航向偏差并会反向纠正，这正是恢复跳变的真实情形。
        if cfg.vyaw_pulse != 0 and cfg.pin_cmd:
            t_seg = (teleop_policy.step_counter - 1) * control_dt
            if cfg.vyaw_pulse_at_s - 1e-9 <= t_seg < cfg.vyaw_pulse_at_s + cfg.vyaw_pulse_s - 1e-9:
                base_vyaw = base_vyaw + cfg.vyaw_pulse

        # ========================================================
        #  4：执行器底层同步修改写入
        # ========================================================
        if orig_tensor is not None:
            orig_tensor[0, 6] = base_vx
            orig_tensor[0, 7] = base_vy
            orig_tensor[0, 8] = base_vyaw
            
        if cfg.pin_cmd:
            cmd_term.set_velocity(base_vx, base_vy, base_vyaw)
        elif cmd_term is not None and hasattr(cmd_term, "vel_command_b"):
            cmd_term.vel_command_b[:, 0] = base_vx
            cmd_term.vel_command_b[:, 1] = base_vy
            cmd_term.vel_command_b[:, 2] = base_vyaw
        environment_cmd = cmd_term.vel_command_b[0].detach().cpu().tolist() if cmd_term is not None else [np.nan]*3
        environment_standing = float(cmd_term.is_standing_env[0].item()) if cmd_term is not None else np.nan

        # ========================================================
        # 阶梯 5：建立观测张量深拷贝副本，进行传感器多路加噪与滤波对比
        # ========================================================
        if hasattr(obs, "keys"):
            fake_obs = obs.copy()  
            for k in list(fake_obs.keys()):
                if isinstance(fake_obs[k], torch.Tensor):
                    fake_obs[k] = fake_obs[k].clone() 
        else:
            fake_obs = obs.clone() if hasattr(obs, "clone") else obs

        target_tensor = None
        if isinstance(fake_obs, torch.Tensor):
            target_tensor = fake_obs
        elif hasattr(fake_obs, "keys"):
            for k in ["actor", "policy", "obs"]: 
                if k in fake_obs.keys():
                    target_tensor = fake_obs[k]
                    break

        raw_yaw, dirty_yaw, filtered_yaw, filtered_yaw_ad, used_yaw = 0.0, 0.0, 0.0, 0.0, 0.0
        raw_vx, dirty_vx, filtered_vx, filtered_vx_ad = 0.0, 0.0, 0.0, 0.0

        if target_tensor is None:
            raise RuntimeError("Actor observation tensor unavailable; refusing to record zero input")
        raw_yaw = target_tensor[0, 2].item()
        # 区间平均角速度：上一步与本步根链接姿态的相对旋转 R_prevᵀR_now 的旋转向量 / dt，取机体 z 分量。
        # 与 raw_yaw（本步点采样）描述同一个区间，差别只在点采样与区间平均。段首没有上一姿态，用 raw_yaw。
        raw_yaw_interval = raw_yaw
        quat_now = _first_attr(env.unwrapped.scene["robot"].data, "root_link_quat_w", "root_quat_w")
        if quat_now is not None:
            q_now = [quat_now[0, i].item() for i in range(4)]
            if teleop_policy.last_quat is not None:
                w0, x0, y0, z0 = teleop_policy.last_quat
                w1, x1, y1, z1 = q_now
                # q_rel = conj(q0) ⊗ q1，(w, x, y, z)
                rw = w0*w1 + x0*x1 + y0*y1 + z0*z1
                rz = w0*z1 - x0*y1 + y0*x1 - z0*w1
                rxy = np.hypot(w0*x1 - x0*w1 - y0*z1 + z0*y1, w0*y1 + x0*z1 - y0*w1 - z0*x1)
                if rw < 0:
                    rw, rz = -rw, -rz
                vec = np.hypot(rxy, rz)
                angle = 2.0 * np.arctan2(vec, rw)
                raw_yaw_interval = float((angle / vec if vec > 1e-12 else 2.0) * rz / dt)
            teleop_policy.last_quat = q_now
        # Keep the historical one-sample reference delay in both feedback modes.
        dirty_yaw, filtered_yaw, filtered_yaw_ad, used_yaw = teleop_policy.rig.step(
            raw_yaw, ref_psi=teleop_policy.last_gt_yaw, raw_interval=raw_yaw_interval)
        used_yaw_feedback = teleop_policy.rig.used_feedback
        target_tensor[0, 2] = used_yaw
        feedback.advance(used_yaw_feedback, dt, teleop_policy.rig.fusion_heading)
        controller.integrated_yaw = feedback.integrated_yaw

        # 3. 线速度真值提取
        gt_yaw, gt_x, gt_y, gt_wz = 0.0, 0.0, 0.0, 0.0
        gt_roll, gt_pitch, gt_wx, gt_wy = 0.0, 0.0, 0.0, 0.0
        try:
            robot = env.unwrapped.scene["robot"]
            vel_tensor = None
            if hasattr(robot.data, "root_link_lin_vel_b"):
                vel_tensor = robot.data.root_link_lin_vel_b
            elif hasattr(robot.data, "root_com_lin_vel_b"):
                vel_tensor = robot.data.root_com_lin_vel_b

            if vel_tensor is not None:
                if len(vel_tensor.shape) > 1:
                    raw_vx = vel_tensor[0, 0].item()
                else:
                    raw_vx = vel_tensor[0].item()

            # 物理引擎真值：绝对航向 / 世界位置 / 真实角速度。
            # Ground truth evaluates motion. In fusion mode, heading also feeds
            # the synthetic noisy reference through SensorRig (never directly).
            quat = _first_attr(robot.data, "root_link_quat_w", "root_quat_w")
            if quat is None:
                raise RuntimeError("Ground-truth quaternion unavailable")
            if quat is not None:
                w, x, y, z = (quat[0, i].item() for i in range(4))  # mjlab 为 (w,x,y,z)
                quat_norm_sq = w*w + x*x + y*y + z*z
                if not np.isfinite(quat_norm_sq) or quat_norm_sq <= 1e-12:
                    raise RuntimeError("Invalid quaternion for stability measurement")
                base_tilt = float(np.arccos(np.clip(1.0 - 2.0*(x*x + y*y)/quat_norm_sq, -1.0, 1.0)))
                gt_yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
                # 同一四元数的 ZYX 横滚/俯仰（与 gt_yaw 构成一组欧拉角）。
                gt_roll = float(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
                gt_pitch = float(np.arcsin(np.clip(2.0 * (w * y - z * x) / quat_norm_sq, -1.0, 1.0)))
                teleop_policy.last_gt_yaw = gt_yaw      # 供下一步的 fusion 臂当绝对参考

            pos = _first_attr(robot.data, "root_link_pos_w", "root_pos_w")
            if pos is None:
                raise RuntimeError("Ground-truth position unavailable")
            if pos is not None:
                gt_x, gt_y = pos[0, 0].item(), pos[0, 1].item()
                base_height_w = pos[0, 2].item()
                if not np.isfinite(base_height_w):
                    raise RuntimeError("Non-finite base height")

            ang = _first_attr(robot.data, "root_link_ang_vel_b", "root_com_ang_vel_b")
            if ang is not None:
                gt_wz = ang[0, 2].item()
                gt_wx, gt_wy = ang[0, 0].item(), ang[0, 1].item()
        except Exception as exc:
            raise RuntimeError("Cannot read physical state for reference/evaluation") from exc

        dirty_vx = raw_vx + episode_rngs["vx"].normal(0, 0.25)
        
        kf_vx_rate.predict()
        filtered_vx = kf_vx_rate.update(dirty_vx)
        
        ad_kf_vx_lite.predict()
        filtered_vx_ad = ad_kf_vx_lite.update(dirty_vx)

        # 步态相位（obs 通道 9/10 = sin/cos）。谐波滤波器要靠它锁频。
        phase_sin, phase_cos = 0.0, 0.0
        try:
            if orig_tensor is not None and orig_tensor.shape[1] > 10:
                phase_sin = orig_tensor[0, 9].item()
                phase_cos = orig_tensor[0, 10].item()
        except Exception:
            pass

        # 记录全维度日志
        global_velocity_log.append([
            raw_yaw, dirty_yaw, filtered_yaw, filtered_yaw_ad, # [0-3] 角速度组
            raw_vx,  dirty_vx,  filtered_vx,  filtered_vx_ad,  # [4-7] 线速度组
            correction_cmd,                                    # [8] PID纠偏量
            ad_kf_vx_lite.R[0, 0],                             # [9] 测量噪声协方差 R
            gt_yaw, gt_x, gt_y, gt_wz,                         # [10-13] 物理真值：航向/位置/角速度
            phase_sin, phase_cos,                              # [14-15] 步态相位
            float(controller.integrated_yaw),                  # [16] 独立积分航向（两种反馈模式均保留）
            float(base_vyaw),                                  # [17] 最终下发的 yaw 指令
            used_yaw,
            teleop_policy.rig.fusion_heading if teleop_policy.rig.fusion_heading is not None else np.nan,
            teleop_policy.rig.fusion_bias if teleop_policy.rig.fusion_bias is not None else np.nan,
            pid_feedback_yaw, feedback.value, controller.yaw_target,
            float(not is_user_turning), float(teleop_policy.rig.reference_updated),
            float(teleop_policy.rig.reference_rejected), float(teleop_policy.segment_id),
            np.nan, np.nan,  # populated after env.step in batch mode; viewer has no step hook
            base_tilt, base_height_w,
            *observation_cmd, *target_tensor[0, 6:9].detach().cpu().tolist(),
            *environment_cmd, observation_standing, environment_standing,
            gt_roll, gt_pitch, gt_wx, gt_wy,                   # schema v5
            raw_yaw_interval, used_yaw_feedback,               # schema v6
        ])
        return base_policy(fake_obs)

    def reset_policy_state(segment_id=0):
        nonlocal kf_yaw_rate, kf_vx_rate, ad_kf_vx_lite, ad_kf_yaw_lite
        kf_yaw_rate, kf_vx_rate = KalmanFilter(dt=control_dt), KalmanFilter(dt=control_dt)
        ad_kf_vx_lite = AdaptiveKalmanfilter(dt=control_dt)
        ad_kf_yaw_lite = AdaptiveKalmanfilter(dt=control_dt)
        def kf_update(z):
            kf_yaw_rate.predict()
            return kf_yaw_rate.update(z)
        def akf_update(z):
            ad_kf_yaw_lite.predict()
            return ad_kf_yaw_lite.update_for_yaw(z)
        teleop_policy.rig = SensorRig(cfg.rig, kf_update, akf_update, dt=control_dt,
            rng=episode_rngs["gyro"], reference_rng=episode_rngs["reference"],
            dropout_rng=episode_rngs["dropout"], bias_rng=episode_rngs["bias"])
        teleop_policy.last_gt_yaw = None
        teleop_policy.last_quat = None
        teleop_policy.last_episode_step = None
        teleop_policy.step_counter = 0
        teleop_policy.segment_id = segment_id
        feedback.reset()
        controller.integrated_yaw = 0.0
        controller.yaw_target = controller.error_integral = controller.last_error = 0.0
        controller.is_turning = False

    def start_repetition(index):
        nonlocal episode_info, episode_rngs
        episode_info = episode_seed_metadata(cfg.seed, index)
        episode_rngs = {name: np.random.default_rng(episode_info["seed_" + name])
                       for name in ("gyro", "reference", "dropout", "vx", "bias")}
        reset_policy_state()

    def save_trace(index, batch, stop_reason="viewer_exit", termination_events=None):
        if not global_velocity_log:
            return
        fr = cfg.friction if cfg.friction is not None else 0.05
        tag = f"f{fr}_s{cfg.rig.noise_std}_b{cfg.rig.gyro_bias}_{cfg.rig.filt}"
        if cfg.sweep_vyaw > 0:
            tag += f"_sweep{cfg.sweep_vyaw}seed{cfg.sweep_seed}"
        tag += f"_fb{cfg.feedback_mode}_seed{cfg.seed}_r{index}_{run_metadata['run_id']}"
        path = cfg.output_dir / f"trace_{tag}.npz"
        metadata = {**run_metadata, **episode_info, "friction": fr,
                    "collection_mode": "batch" if batch else "viewer",
                    "segments": teleop_policy.segment_id + 1,
                    "completed_steps": len(global_velocity_log),
                    "requested_steps": cfg.batch_steps or -1,
                    "completed_duration_s": len(global_velocity_log) * control_dt,
                    "stop_reason": stop_reason,
                    "termination_events_json": json.dumps(termination_events or [])}
        # Exclusive creation prevents accidental overwrites, including repeated timestamps.
        with path.open("xb") as stream:
            np.savez_compressed(stream, data=np.array(global_velocity_log),
                                columns=np.array(TRACE_COLUMNS), **metadata)
        print(f"[TRACE] {path.resolve()} ({len(global_velocity_log)} steps)")

    policy = teleop_policy
    start_repetition(0)
    if cfg.batch_steps:
        try:
            for rep in range(cfg.repeats):
                global_velocity_log.clear()
                start_repetition(rep)
                env.seed(episode_info["seed_env"])
                obs, _ = env.reset()
                stop_reason, termination_events = "budget_reached", []
                try:
                    with torch.inference_mode():
                        for step in range(cfg.batch_steps):
                            obs, _, dones, extras = env.step(policy(obs))
                            terminated = bool(env.unwrapped.reset_terminated[0].item())
                            truncated = bool(env.unwrapped.reset_time_outs[0].item())
                            global_velocity_log[-1][TRACE_COLUMNS.index("terminated_after_step")] = float(terminated)
                            global_velocity_log[-1][TRACE_COLUMNS.index("truncated_after_step")] = float(truncated)
                            if bool(dones[0].item()):
                                manager = env.unwrapped.termination_manager
                                terms = [name for name in manager.active_terms
                                         if bool(manager.get_term(name)[0].item())]
                                termination_events.append({"step": step + 1,
                                    "terminated": terminated, "truncated": truncated, "terms": terms})
                                if cfg.batch_stop_on_done:
                                    stop_reason = "terminated" if terminated else "time_out" if truncated else "done"
                                    break
                            if (step + 1) % 500 == 0:
                                print(f"[PROGRESS] rep={rep} steps={step + 1}/{cfg.batch_steps}", flush=True)
                except BaseException:
                    save_trace(rep, batch=True, stop_reason="interrupted_or_error",
                               termination_events=termination_events)
                    raise
                save_trace(rep, batch=True, stop_reason=stop_reason, termination_events=termination_events)
        finally:
            env.close()
            controller.listener.stop()
        return
    try:
        if cfg.viewer == "auto":
            has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
            resolved_viewer = "native" if has_display else "viser"
        else:
            resolved_viewer = cfg.viewer

        if resolved_viewer == "native":
            NativeMujocoViewer(env, policy).run()
        elif resolved_viewer == "viser":
            ViserPlayViewer(env, policy).run()
    except KeyboardInterrupt:
        print("\n[INFO] 接收到退出指令...")
    finally:
        env.close()
        controller.listener.stop()

        # ========================================================
        # 轨迹落盘：先存全量原始数据，再画图。
        # 画图会截断到 350 步，落盘必须在截断之前，且不能被画图异常带走。
        # ========================================================
        try:
            if global_velocity_log:
                save_trace(0, batch=False)
        except Exception as e:
            print(f"[ERROR] 轨迹落盘失败: {e}")

        # ========================================================
        # 绘图模块
        # ========================================================
        print("\n[INFO] 正在生成PID和滤波图")
        try:
            data_full = np.array(global_velocity_log)
            if len(data_full):
                segment_col = TRACE_COLUMNS.index("segment_id")
                final_segment = data_full[-1, segment_col]
                data_full = data_full[data_full[:, segment_col] == final_segment]
                print(f"[PLOT] 仅绘制最后一个连续区段 {int(final_segment)}，完整记录已保存")
            data = data_full

            if len(data) > 0:
                if len(data) > 350:
                    data = data[:350, :]   # 滤波对比图看细节，只取前 350 步
                
                fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
                
                ax1.plot(data[:, 0], label="Raw Yaw Rate (Ground Truth)", color='#2ca02c', alpha=0.8, linewidth=1.8)
                ax1.plot(data[:, 1], label="Noisy Yaw Rate (Sensor)", color='#d62728', alpha=0.15, linewidth=0.8) 
                ax1.plot(data[:, 2], label="Filtered Yaw Rate (Kalman)", color='#1f77b4', linewidth=2.8) 
                ax1.plot(data[:, 3], label="Filtered Yaw Rate (Adaptive Kalman)", color='#9467bd', linewidth=1.8) 
                ax1.set_title("G1 Robot Yaw Rate: Kalman Filter Ablation (Microscopic View)", fontsize=13, fontweight='bold')
                ax1.set_xlabel("Time Steps (0.02s / step)", fontsize=10)
                ax1.set_ylabel("Normalized Value", fontsize=10)
                ax1.legend(loc="upper right", framealpha=0.9)
                ax1.grid(True, linestyle='--', alpha=0.5)
                
                ax2.plot(data[:, 5], label="Noisy Linear Vx (Sensor)", color='#d62728', alpha=0.15, linewidth=0.8) 
                ax2.plot(data[:, 4], label="Raw Linear Vx (Ground Truth)", color='#2ca02c', alpha=0.4, linewidth=3.0) 
                ax2.plot(data[:, 6], label="Filtered Linear Vx (Kalman)", color='#1f77b4', linewidth=1.8)            
                if data.shape[1] > 7: 
                    ax2.plot(data[:, 7], label="Filtered Linear Vx (Adaptive_Kalman)", color='#9467bd', linestyle='--', linewidth=2.8)            
                ax2.set_title("G1 Robot Linear Velocity ($V_x$): Filter Comparison", fontsize=13, fontweight='bold')
                ax2.set_xlabel("Time Steps (0.02s / step)", fontsize=10)
                ax2.set_ylabel("Normalized Value", fontsize=10)
                ax2.legend(loc="upper right", framealpha=0.9)
                ax2.grid(True, linestyle='--', alpha=0.5)
                
                fig1.tight_layout()
                filter_path = "kalman_filter_ablation.png"
                fig1.savefig(filter_path, dpi=300, bbox_inches='tight')
                print(f"🟢 滤波图已保存: {filter_path}")
                
                # --- 航向与横向漂移：全部用物理真值，不用 PID 自己的反馈量 ---
                fig2, (ax3, ax5) = plt.subplots(2, 1, figsize=(12, 8))
                try:
                    if data_full.shape[1] < len(TRACE_COLUMNS):
                        raise RuntimeError(
                            f"日志只有 {data_full.shape[1]} 列，需要 {len(TRACE_COLUMNS)} 列。"
                            f"说明 global_velocity_log.append 还是旧版——真值那批改动没进去。")

                    n = np.arange(len(data_full))

                    # atan2 出来的航向缠绕在 (-pi, pi]，跨界会出现 6.28 的假断崖，必须 unwrap
                    gt_yaw = np.unwrap(data_full[:, 10])
                    pid_yaw = data_full[:, TRACE_COLUMNS.index("pid_feedback_yaw")]
                    # This is the actual pre-update PID feedback in either mode.

                    ax3.plot(n, gt_yaw - gt_yaw[0], label="Ground Truth Heading (physics)",
                             color='#e377c2', linewidth=2.5)
                    ax3.plot(n, pid_yaw - pid_yaw[0], label="PID feedback (actual control input)",
                             color='#7f7f7f', linewidth=1.5, linestyle='--')
                    ax3.axhline(0.0, color='k', linewidth=0.8, alpha=0.5)
                    ax3.set_title("Heading: physical ground truth vs actual PID feedback",
                                  fontsize=13, fontweight='bold')
                    ax3.set_ylabel("Heading change [rad]", fontsize=10)
                    ax3.legend(loc="upper left", framealpha=0.9)
                    ax3.grid(True, linestyle='--', alpha=0.5)

                    # 横向漂移：起点和复位朝向都是随机的，直接画世界系 Y 没有意义，
                    # 必须把位移投影到"起始航向"这个参考系里。
                    psi0 = data_full[0, 10]
                    dx = data_full[:, 11] - data_full[0, 11]
                    dy = data_full[:, 12] - data_full[0, 12]
                    forward = dx * np.cos(psi0) + dy * np.sin(psi0)
                    lateral = -dx * np.sin(psi0) + dy * np.cos(psi0)

                    ax5.plot(n, lateral, label="Lateral drift (perpendicular to start heading)",
                             color='#1f77b4', linewidth=2.2)
                    ax5.plot(n, forward, label="Forward travel", color='#2ca02c',
                             linewidth=1.2, alpha=0.6)
                    ax5.axhline(0.0, color='k', linewidth=0.8, alpha=0.5)
                    ax5.set_title(
                        f"Path in start frame  |  final lateral drift = {lateral[-1]:+.3f} m "
                        f"over {forward[-1]:.2f} m travelled",
                        fontsize=13, fontweight='bold')
                    ax5.set_ylabel("Distance [m]", fontsize=10)
                    ax5.set_xlabel("Time Steps (0.02s / step)", fontsize=10)
                    ax5.legend(loc="upper left", framealpha=0.9)
                    ax5.grid(True, linestyle='--', alpha=0.5)

                    fig2.tight_layout()
                    pid_path = "pid_heading_tracking.png"
                    fig2.savefig(pid_path, dpi=300, bbox_inches='tight')
                    print(f"🟢 PID 纠正图保存: {pid_path}")
                    print(f"   真值航向净变化 {gt_yaw[-1] - gt_yaw[0]:+.4f} rad | "
                          f"PID 自认为 {pid_yaw[-1] - pid_yaw[0]:+.4f} rad | "
                          f"反馈航向误差 {(pid_yaw[-1]-pid_yaw[0])-(gt_yaw[-1]-gt_yaw[0]):+.4f} rad")
                    print(f"   横向漂移 {lateral[-1]:+.3f} m / 前进 {forward[-1]:.2f} m")
                except Exception as inner_e:
                    print(f"[WARN] 计算PID角度失败: {inner_e}")
                
                fig3, ax4 = plt.subplots(1, 1, figsize=(12, 3))
                ax4.plot(data[:, 9], label="Adaptive R Value (Online Estimation)", color='#ff7f0e', linewidth=2.0)
                ax4.set_title("Real-time R-matrix Estimation (Sage-Husa)", fontsize=13, fontweight='bold')
                ax4.set_ylabel("R Value")
                ax4.grid(True, linestyle='--', alpha=0.5)
                fig3.tight_layout()
                
                plt.show()  
            else:
                print("[WARN] 内部数据列表为空。")
        except Exception as e:
            print(f"[ERROR] 画图失败: {e}")
            

def main():
    import mjlab.tasks  # noqa: F401
    import src.tasks

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    args = tyro.cli(
        PlayConfig,
        args=remaining_args,
        default=PlayConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )
    del remaining_args

    run_play(chosen_task, args)


if __name__ == "__main__":
    main()
