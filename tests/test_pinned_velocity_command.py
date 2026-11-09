"""Exercise actual command manager resampling and the project's phase function."""
import ast
from pathlib import Path
import os
import json
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from pinned_velocity_command import make_pinned_velocity_cfg, UniformVelocityCommandCfg
from pinned_input_audit import audit_inputs
import test_play_feedback


def config():
    return UniformVelocityCommandCfg(entity_name='robot', resampling_time_range=(.04, .06),
        heading_command=True, rel_standing_envs=1.,
        ranges=UniformVelocityCommandCfg.Ranges((-1, 1), (-1, 1), (-1, 1), (-3.14, 3.14)))


def environment():
    return SimpleNamespace(num_envs=2, device='cpu', step_dt=.02,
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        scene={'robot': SimpleNamespace(data=SimpleNamespace(heading_w=torch.zeros(2),
            root_link_lin_vel_b=torch.zeros(2, 3), root_link_ang_vel_b=torch.zeros(2, 3)))})


class PinnedCommandTests(unittest.TestCase):
    def phase(self, env):
        project = Path(os.environ.get('UNITREE_TEST_PROJECT_ROOT', str(Path(__file__).resolve().parents[1])))
        path = project / 'src/tasks/velocity/mdp/observations.py'
        node = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if getattr(n, 'name', None) == 'phase')
        ns = {'torch': torch, 'ManagerBasedRlEnv': object}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        return ns['phase'](env, .6, 'twist')

    def test_random_standing_resamples_never_mask_pinned_walking_phase(self):
        env = environment()
        cmd = make_pinned_velocity_cfg(config(), .5).build(env)
        env.command_manager = SimpleNamespace(get_command=lambda name: cmd.command)
        cmd.reset()
        cmd.set_velocity(.5, 0., .17)
        for _ in range(30):
            env.episode_length_buf += 1
            cmd.compute(.02)
            torch.testing.assert_close(cmd.command, torch.tensor([[.5, 0., .17]]).repeat(2, 1))
            self.assertFalse(cmd.is_standing_env.any())
            torch.testing.assert_close(torch.linalg.norm(self.phase(env), dim=1), torch.ones(2))
        self.assertTrue((cmd.command_counter > 2).all())
        cmd.reset(torch.tensor([0]))
        torch.testing.assert_close(cmd.command[:, 2], torch.tensor([0., .17]))

    def test_resampling_retains_original_rng_consumption_and_schedule(self):
        old, new = config().build(environment()), make_pinned_velocity_cfg(config(), .5).build(environment())
        torch.manual_seed(24002)
        start = torch.get_rng_state()
        old.reset(torch.arange(2))
        after = torch.get_rng_state()
        torch.set_rng_state(start)
        new.reset(torch.arange(2))
        torch.testing.assert_close(torch.get_rng_state(), after)
        torch.testing.assert_close(old.time_left, new.time_left)
        self.assertTrue(old.is_standing_env.all())
        self.assertFalse(new.is_standing_env.any())

    def test_stationary_command_keeps_real_standing_gate(self):
        env = environment()
        cmd = make_pinned_velocity_cfg(config(), 0.).build(env)
        env.command_manager = SimpleNamespace(get_command=lambda name: cmd.command)
        cmd.reset()
        self.assertTrue(cmd.is_standing_env.all())
        torch.testing.assert_close(self.phase(env), torch.zeros(2, 2))

    def test_reject_physical_velocity_randomization_without_mutating_source(self):
        source = config()
        converted = make_pinned_velocity_cfg(source, .5)
        self.assertIsNot(source.ranges, converted.ranges)
        self.assertEqual(source.rel_standing_envs, 1.)
        source.init_velocity_prob = .1
        with self.assertRaisesRegex(ValueError, 'init_velocity_prob'):
            make_pinned_velocity_cfg(source, .5)

    def test_all_three_feedback_modes_obey_input_contract_and_detect_corruption(self):
        for filt, mode in [('dirty', 'integrated'), ('fusion', 'integrated'), ('fusion', 'fusion_heading')]:
            helper = test_play_feedback.PlayFeedbackIntegrationTests()
            self.addCleanup(helper.doCleanups)
            run = helper.run_case(filt=filt, mode=mode, steps=30, real_metadata=True, stop_on_done=True)
            trace = run['traces'][0]
            result = audit_inputs(trace, {'dt': .02, 'pin_vx': .5})
            self.assertEqual(result['zero_phase_samples'], 0)
            corrupted = dict(trace, data=trace['data'].copy())
            corrupted['data'][3, 14:16] = 0.
            with self.assertRaises(AssertionError):
                audit_inputs(corrupted, {'dt': .02, 'pin_vx': .5})
            from analyze_heading_fixed_input import read_trial
            from experiment_record import episode_seed_metadata
            common = json.loads(trace['config_json'].item())
            common.update(steps=30, duration_s=.6)
            arm = 'A' if filt == 'dirty' else 'C' if mode == 'fusion_heading' else 'B'
            case = dict(case_id=arm+'_seed73', arm=arm, seed=73, filt=filt,
                        feedback_mode=mode, derived_seeds=episode_seed_metadata(73, 0))
            plan = dict(common=common, identity={k: trace[k].item() for k in ('checkpoint_sha256', 'source_tree_sha256')})
            row, _ = read_trial(run['trace_paths'][0], case, plan)
            self.assertTrue(row['full_duration'])
            self.assertEqual(row['input_checked_samples'], 30)


if __name__ == '__main__':
    unittest.main()
