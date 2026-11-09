"""Run the agreed nine-case heading pilot, sequentially, with a saved protocol.

Each case starts a fresh process. Failures are retained, never replaced by an
extra successful trial. This command does not run the later 90-case experiment.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from experiment_record import build_run_metadata, episode_seed_metadata

CHECKPOINT = 'logs/rsl_rl/g1_velocity/2026-05-20_14-06-36/model_12000.pt'
CHECKPOINT_SHA256 = '644b28242b551f9755e8458fe7600ba448bcce326766910f884b0c3ebd2a4582'
SOURCES = ['scripts/play.py', 'sensor_rig.py', 'heading_fusion.py',
           'heading_feedback.py', 'experiment_record.py', 'contact_settings.py', 'src']
MODES = {'A': ('dirty', 'integrated'), 'B': ('fusion', 'integrated'),
         'C': ('fusion', 'fusion_heading')}


def protocol(project):
    cases = []
    for seed, order in zip((12000, 12001, 12002), ('ABC', 'BCA', 'CAB')):
        for arm in order:
            filt, feedback = MODES[arm]
            cases.append({'case_id': f'{arm}_seed{seed}', 'arm': arm, 'seed': seed,
                          'filt': filt, 'feedback_mode': feedback,
                          'derived_seeds': episode_seed_metadata(seed, 0)})
    common = {'task_id': 'Unitree-G1-Flat', 'checkpoint': CHECKPOINT,
              'steps': 3000, 'dt': 0.02, 'duration_s': 60.0, 'repeats': 1,
              'pin_cmd': True, 'pin_vx': 0.5, 'fixed_body': True, 'friction': 0.8,
              'kp': 0.91, 'ki': 0.0, 'kd': 0.01, 'noise_std': 0.15,
              'gyro_bias': 0.02, 'abs_sigma': 0.05, 'abs_hz': 1.0, 'abs_dropout': 0.0,
              'batch_stop_on_done': True, 'no_terminations': False,
              'reference_delay_steps': 1, 'pid_feedback_delay_steps': 1}
    identity = build_run_metadata(project, CHECKPOINT, common, source_paths=SOURCES)
    if identity.get('checkpoint_sha256') != CHECKPOINT_SHA256:
        raise RuntimeError('Checkpoint bytes differ from the reviewed model_12000.pt')
    versions = {}
    for name in ('torch', 'numpy', 'mjlab', 'mujoco', 'mujoco-warp', 'warp-lang', 'rsl-rl-lib', 'tyro'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not_registered'
    return {'kind': 'three_seed_closed_loop_pilot', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'project': str(project), 'python': sys.executable, 'common': common, 'cases': cases,
            'identity': identity, 'dependency_versions': versions,
            'primary_outcome': 'RMS physical heading deviation from the starting heading over the recorded trajectory',
            'secondary_outcomes': ['absolute final heading deviation', 'RMS actual PID heading estimation error',
                                   'control RMS and saturation fraction', 'forward/lateral displacement',
                                   'failure and completed duration'],
            'failure_rule': 'Keep existing fell_over (>70 deg tilt) and time_out terms; stop at first done.',
            'analysis_rule': 'Paired descriptive B-C comparisons by seed; no significance/equivalence claim with n=3. '
                             'Keep all failures; compare full 60 s outcomes only for completed trials.',
            'sampling_note': 'Each trace row precedes one env.step. Completed duration is steps*dt; '
                             'last physical sample time is (steps-1)*dt; a terminal post-step state is not recorded.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = protocol(project)
    plan_file = output / 'protocol.json'
    if plan_file.exists():
        saved = json.loads(plan_file.read_text(encoding='utf-8'))
        for key in ('common', 'cases'):
            if saved[key] != plan[key]:
                raise RuntimeError('Existing protocol differs; use a separate output directory')
        if saved['identity']['source_tree_sha256'] != plan['identity']['source_tree_sha256']:
            raise RuntimeError('Experiment code changed since protocol creation')
        plan = saved
    else:
        with plan_file.open('x', encoding='utf-8') as stream:
            json.dump(plan, stream, ensure_ascii=False, indent=2)
    print(f'[PROTOCOL] {plan_file}', flush=True)
    if args.prepare_only:
        return
    env = {**os.environ, 'GIT_PYTHON_REFRESH': 'quiet', 'PYTHONIOENCODING': 'utf-8',
           'PYTHONUNBUFFERED': '1', 'PYTHONPATH': str(project) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    for number, case in enumerate(plan['cases'], 1):
        case_dir = output / case['case_id']
        # No automatic resampling or overwriting after a failed/partial case.
        case_dir.mkdir()
        command = [sys.executable, '-X', 'utf8', 'scripts/play.py', 'Unitree-G1-Flat',
            '--checkpoint-file', CHECKPOINT, '--device', 'cuda:0', '--pin-cmd', 'True',
            '--pin-vx', '0.5', '--fixed-body', 'True', '--friction', '0.8',
            '--kp', '0.91', '--ki', '0', '--kd', '0.01',
            '--rig.noise-std', '0.15', '--rig.gyro-bias', '0.02',
            '--rig.abs-sigma', '0.05', '--rig.abs-hz', '1.0', '--rig.abs-dropout', '0',
            '--rig.filt', case['filt'], '--feedback-mode', case['feedback_mode'],
            '--seed', str(case['seed']), '--batch-steps', '3000', '--repeats', '1',
            '--no-terminations', 'False', '--batch-stop-on-done', 'True',
            '--output-dir', str(case_dir)]
        record = {**case, 'command': command, 'started_utc': datetime.now(timezone.utc).isoformat()}
        print(f'[RUN] {number}/9 {case["case_id"]}', flush=True)
        started = time.monotonic()
        with (case_dir / 'console.txt').open('x', encoding='utf-8') as log:
            child = subprocess.Popen(command, cwd=project, env=env, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            for line in child.stdout:
                log.write(line)
                log.flush()
                if line.startswith(('[PROGRESS]', '[TRACE]')):
                    print(line.rstrip(), flush=True)
            returncode = child.wait()
        record.update(returncode=returncode, wall_seconds=time.monotonic() - started)
        traces = sorted(case_dir.glob('trace*.npz'))
        record['traces'] = [str(p.relative_to(output)) for p in traces]
        if returncode == 0 and len(traces) == 1:
            with np.load(traces[0], allow_pickle=False) as z:
                record.update(stop_reason=z['stop_reason'].item(), completed_steps=z['completed_steps'].item())
        (case_dir / 'run_record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'[END] {case["case_id"]} code={returncode} reason={record.get("stop_reason", "process_error")} '
              f'wall={record["wall_seconds"]:.1f}s', flush=True)
        if returncode != 0 or len(traces) != 1:
            raise RuntimeError(f'Infrastructure/recording error in {case["case_id"]}; retained console and partial data, stopping pilot')
    print('[COMPLETE] All 9 planned trials recorded.', flush=True)


if __name__ == '__main__':
    main()
