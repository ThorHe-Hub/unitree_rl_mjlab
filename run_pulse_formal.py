"""指令阶跃承受度正式批次（35 条）。预注册见 <输出目录>/PREREG.md。

    python run_pulse_formal.py --prepare-only
    python run_pulse_formal.py --run [--limit N]
    python run_pulse_formal.py --status

流程与 run_design_family_formal.py 相同：冻结协议与源码快照，每条之前核对指纹，每次尝试存在
<case>/attempt_NNN/，完成的尝试写 completed.json。摔倒是最终结果，不补抽。
Windows 下输出目录自动加长路径前缀（轨迹文件名很长，否则超过 260 字符，见正式批次 DEVIATIONS D1）。
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import numpy as np
import psutil

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
from experiment_record import build_run_metadata, episode_seed_metadata   # noqa: E402
from run_heading_pilot import CHECKPOINT, CHECKPOINT_SHA256               # noqa: E402
from run_heading_fixed_input import digest                                # noqa: E402
from run_heading_bias import batch_lock, stop_child                       # noqa: E402
from heading_bias_batch import write_json                                 # noqa: E402
from run_design_family_dev import COMMON, command                         # noqa: E402
from run_design_family_formal import play_source_paths, _deps             # noqa: E402


def long_path(p):
    p = Path(p).resolve()
    return Path('\\\\?\\' + str(p)) if os.name == 'nt' and not str(p).startswith('\\\\?\\') else p


SOURCES = play_source_paths()
# PULSE_FORMAL_OUT 只用于冒烟测试（build/ 下演练），正式批次不设
OUT = long_path(os.environ.get('PULSE_FORMAL_OUT', PROJECT / 'unitree_heading_results/pulse_formal_20260929'))
GUARDED = ['run_pulse_formal.py', 'run_design_family_dev.py', 'run_design_family_formal.py', 'sensor_profiles.py',
           'pinned_velocity_command.py']
FINAL_REASONS = {'budget_reached', 'terminated', 'time_out', 'done'}

ARM = dict(arm='Cs', feedback_mode='fusion_heading', q_b=1e-16, tau=None)
BASE = dict(filt='fusion', contact_friction=None, outage=(0.0, 0.0), pin_vx=0.5, variant='point', gyro_signal='point')
AMPS = [0.0, 0.05, 0.125, 0.25, 0.5, 1.0, 2.0]
SEEDS = list(range(72000, 72005))
PULSE_S = 1.0
STEPS = 2000


def pulse_at(j):
    """种子序号 j → 脉冲起点：把步态周期 0.6 s 五等分，各种子落在不同步态相位。"""
    return round(20.0 + 0.12 * j, 2)


def build_cases():
    """按种子分组、种子内按幅值排列；中途停下时，已完成的种子是完整配对。"""
    out = []
    for j, seed in enumerate(SEEDS):
        for a in AMPS:
            out.append(dict(BASE, **ARM, case_id='P_a%g_s%d' % (a, seed), amp=a, seed=seed, steps=STEPS,
                            pulse=a, pulse_at_s=pulse_at(j), pulse_s=PULSE_S,
                            derived_seeds=episode_seed_metadata(seed, 0)))
    ids = [c['case_id'] for c in out]
    if len(ids) != len(set(ids)):
        raise RuntimeError('duplicate case ids')
    return out


def protocol():
    cases = build_cases()
    for c in cases:
        c['command'] = command(c, Path('<attempt>'))[1:]
    prereg = OUT / 'PREREG.md'
    if not prereg.exists():
        raise RuntimeError('PREREG.md must exist before freezing')
    return dict(
        kind='pulse_formal', created_utc=datetime.now(timezone.utc).isoformat(),
        common=COMMON, arm=ARM, base=BASE, amps=AMPS, seeds=SEEDS, pulse_s=PULSE_S, steps=STEPS,
        cases=cases, trace_schema_version=6,
        identity=build_run_metadata(PROJECT, CHECKPOINT, {}, source_paths=SOURCES),
        guarded_sha256={n: digest(PROJECT / n) for n in GUARDED},
        prereg_sha256=digest(prereg), dependency_versions=_deps(),
        final_stop_reasons=sorted(FINAL_REASONS),
        resume_rule=('Falls and timeouts are final and never replaced. Process failures and '
                     'interrupted_or_error traces stay in numbered attempt directories and are retried '
                     'with the same case and seed. Any change to the source tree, checkpoint, guarded '
                     'files or PREREG.md stops the batch.'))


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'protocol.json'
    candidate = protocol()
    if path.exists():
        plan = load_plan()
        for key in ('cases', 'common', 'arm', 'base', 'amps', 'seeds', 'pulse_s', 'steps', 'trace_schema_version',
                    'guarded_sha256', 'prereg_sha256', 'final_stop_reasons'):
            if json.loads(json.dumps(plan[key])) != json.loads(json.dumps(candidate[key])):
                raise RuntimeError('Frozen protocol differs: %s' % key)
        verify(plan)
        return plan
    if any(OUT.glob('*/attempt_*')):
        raise RuntimeError('attempts exist without a protocol; use a new output directory')
    if candidate['identity']['checkpoint_sha256'] != CHECKPOINT_SHA256 \
            or not candidate['identity']['source_snapshot_complete']:
        raise RuntimeError('checkpoint mismatch or incomplete source snapshot')
    with path.open('x', encoding='utf-8') as f:
        json.dump(candidate, f, ensure_ascii=False, indent=1, allow_nan=False)
    (OUT / 'protocol.sha256').write_text(digest(path) + '\n', encoding='ascii')
    sources = json.loads(candidate['identity']['source_sha256_json'])
    sources.update(candidate['guarded_sha256'])
    with zipfile.ZipFile(OUT / 'experiment_source.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for name, expected in sorted(sources.items()):
            if digest(PROJECT / name) != expected:
                raise RuntimeError('source changed during snapshot: %s' % name)
            z.write(PROJECT / name, name)
        z.write(OUT / 'PREREG.md', 'PREREG.md')
    return candidate


def load_plan():
    path = OUT / 'protocol.json'
    if (OUT / 'protocol.sha256').read_text().strip() != digest(path):
        raise RuntimeError('protocol.json bytes changed')
    return json.loads(path.read_text(encoding='utf-8'))


def verify(plan):
    cur = build_run_metadata(PROJECT, CHECKPOINT, {}, source_paths=SOURCES)
    for key in ('source_tree_sha256', 'checkpoint_sha256'):
        if cur[key] != plan['identity'][key]:
            raise RuntimeError('experiment identity changed: %s' % key)
    for name, expected in plan['guarded_sha256'].items():
        if digest(PROJECT / name) != expected:
            raise RuntimeError('guarded file changed: %s' % name)
    if digest(OUT / 'PREREG.md') != plan['prereg_sha256']:
        raise RuntimeError('PREREG.md changed after freezing')


def inspect_attempt(attempt, case, plan):
    """合格的最终轨迹返回 (路径, stop_reason)；需要重试返回 None；元数据与协议不一致直接报错。"""
    traces = list(attempt.glob('trace*.npz'))
    if len(traces) > 1:
        raise RuntimeError('multiple traces in %s' % attempt)
    if not traces:
        return None
    path = traces[0]
    with np.load(path, allow_pickle=False) as z:
        meta = {k: z[k].item() for k in z.files if k not in ('data', 'columns')}
        n = len(z['data'])
    reason = meta['stop_reason']
    if reason not in FINAL_REASONS:
        return None
    expect = dict(seed=case['seed'], gyro_signal=case['gyro_signal'], feedback_mode=case['feedback_mode'],
                  fusion_q_bias=case['q_b'], pin_vx=case['pin_vx'], abs_hz=COMMON['abs_hz'],
                  noise_std=COMMON['noise_std'], gyro_bias=COMMON['gyro_bias'], abs_outage_start_s=0.0,
                  abs_outage_duration_s=0.0, trace_schema_version=plan['trace_schema_version'],
                  vyaw_pulse=case['pulse'], vyaw_pulse_at_s=case['pulse_at_s'], vyaw_pulse_s=case['pulse_s'],
                  source_tree_sha256=plan['identity']['source_tree_sha256'],
                  checkpoint_sha256=plan['identity']['checkpoint_sha256'])
    for key, value in expect.items():
        if meta.get(key) != value:
            raise RuntimeError('%s: %s = %r, expected %r' % (path, key, meta.get(key), value))
    if reason == 'budget_reached' and n != case['steps']:
        raise RuntimeError('%s: %d samples, expected %d' % (path, n, case['steps']))
    return path, reason


def case_state(case, plan):
    """返回 ('done', 路径) / ('todo', None)。已有最终尝试但缺标记时补写标记。"""
    d = OUT / case['case_id']
    marker = d / 'completed.json'
    finals = []
    for attempt in sorted(d.glob('attempt_*')) if d.exists() else []:
        r = inspect_attempt(attempt, case, plan)
        if r is not None:
            finals.append((attempt, r))
    if len(finals) > 1:
        raise RuntimeError('multiple final attempts for %s; inspect manually' % case['case_id'])
    if not finals:
        if marker.exists():
            raise RuntimeError('completed.json exists but no final attempt: %s' % case['case_id'])
        return 'todo', None
    attempt, (path, reason) = finals[0]
    expected = dict(attempt=attempt.name, trace_sha256=digest(path), stop_reason=reason)
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')) != expected:
            raise RuntimeError('completed.json changed: %s' % case['case_id'])
    else:
        write_json(marker, expected)
    return 'done', path


def run_case(case):
    d = OUT / case['case_id']
    d.mkdir(exist_ok=True)
    number = max([int(p.name.split('_')[-1]) for p in d.glob('attempt_*')] or [0]) + 1
    attempt = d / ('attempt_%03d' % number)
    attempt.mkdir()
    cmd = command(case, attempt)
    record = dict(case_id=case['case_id'], command=cmd, status='launching',
                  started_utc=datetime.now(timezone.utc).isoformat())
    write_json(attempt / 'launch_record.json', record)
    env = {**os.environ, 'GIT_PYTHON_REFRESH': 'quiet', 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1',
           'PYTHONPATH': str(PROJECT) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    child, t0 = None, time.monotonic()
    try:
        with (attempt / 'console.txt').open('x', encoding='utf-8') as log:
            child = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            record.update(child_pid=child.pid, child_created=psutil.Process(child.pid).create_time(),
                          status='running')
            write_json(attempt / 'launch_record.json', record)
            record.update(returncode=child.wait(), status='process_ended')
    except BaseException as exc:
        if child is not None:
            stop_child(child)
        record.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'infrastructure_error',
                      error=repr(exc))
        raise
    finally:
        record.update(wall_seconds=time.monotonic() - t0, ended_utc=datetime.now(timezone.utc).isoformat())
        write_json(attempt / 'run_record.json', record)
    return record


def run(limit=None):
    plan = freeze()
    with batch_lock(OUT):
        ran = 0
        for case in plan['cases']:
            if case_state(case, plan)[0] == 'done':
                continue
            if limit is not None and ran >= limit:
                break
            verify(plan)
            rec = run_case(case)
            ran += 1
            if case_state(case, plan)[0] != 'done':
                last = sorted((OUT / case['case_id']).glob('attempt_*'))[-1]
                tail = (last / 'console.txt').read_text(encoding='utf-8', errors='replace')[-3000:]
                raise SystemExit('[FAIL] %s rc=%s（现场保留，重新 --run 会用同一种子重试）\n%s'
                                 % (case['case_id'], rec.get('returncode'), tail))
            reason = json.loads((OUT / case['case_id'] / 'completed.json').read_text())['stop_reason']
            print('[ OK ] %-18s %-14s %.0fs' % (case['case_id'], reason, rec['wall_seconds']), flush=True)
    status(plan)


def status(plan=None):
    plan = plan or load_plan()
    final = {}
    for case in plan['cases']:
        if case_state(case, plan)[0] == 'done':
            r = json.loads((OUT / case['case_id'] / 'completed.json').read_text())['stop_reason']
            final[r] = final.get(r, 0) + 1
    print('完成 %d / %d：%s' % (sum(final.values()), len(plan['cases']), final))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    if args.prepare_only:
        plan = freeze()
        print('协议已冻结：%d 条，protocol SHA256 %s' % (len(plan['cases']), digest(OUT / 'protocol.json')))
    if args.run:
        run(args.limit)
    if args.status:
        status()


if __name__ == '__main__':
    main()