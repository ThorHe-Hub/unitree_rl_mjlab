"""设计族正式批次（208 条）。预注册见 <输出目录>/PREREG.md。

    python run_design_family_formal.py --prepare-only      # 冻结协议，不启动仿真
    python run_design_family_formal.py --run --limit 6     # 前台跑接下来 6 条未完成的
    python run_design_family_formal.py --status

每条之前重新核对源码树与权重指纹；不一致即停止。每次尝试保存在 <case>/attempt_NNN/，
完成的尝试写 completed.json（trace SHA256）。摔倒/超时是最终结果，不补抽；
进程失败或 interrupted_or_error 保留现场，下次 --run 用同一种子重试。
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


def play_source_paths():
    """play.py 自己做指纹时用的 source_paths 列表，直接从源码解析，避免两处手抄不一致。"""
    import ast
    tree = ast.parse((PROJECT / 'scripts/play.py').read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == 'source_paths' and isinstance(node.value, ast.List):
            return [ast.literal_eval(e) for e in node.value.elts]
    raise RuntimeError('source_paths not found in scripts/play.py')


SOURCES = play_source_paths()

OUT = Path(os.environ.get('DF_FORMAL_OUT', PROJECT / 'unitree_heading_results/design_family_formal_20260929'))
# DF_FORMAL_OUT 只用于冒烟测试（在 build/ 下演练整条流程），正式批次不设。
# 影响"跑什么"的文件：变了就不能续跑。分析脚本不在此列（其指纹在分析时另行记录）。
GUARDED = ['run_design_family_formal.py', 'run_design_family_dev.py', 'sensor_profiles.py',
           'pinned_velocity_command.py']
FINAL_REASONS = {'budget_reached', 'terminated', 'time_out', 'done'}

ARMS = {
    'Bs': dict(feedback_mode='integrated', q_b=1e-4, tau=None),
    'Cs': dict(feedback_mode='fusion_heading', q_b=1e-16, tau=None),
    'S1': dict(feedback_mode='fusion_offset_smoothed', q_b=1e-16, tau=1.0),
    'C8': dict(feedback_mode='fusion_heading', q_b=1e-8, tau=None),
}
VARIANTS = ('point', 'interval')
BASE = dict(filt='fusion', contact_friction=None, outage=(0.0, 0.0), pin_vx=0.5)
SCENES = {
    'N': dict(steps=3000, arms=['Bs', 'Cs', 'S1'], seeds=range(70000, 70010)),
    'O': dict(steps=4500, arms=['Bs', 'Cs', 'S1'], seeds=range(70000, 70010), outage=(20.0, 30.0)),
    'V': dict(steps=4500, arms=['Cs', 'S1', 'C8'], seeds=range(70000, 70005), vx_step_s=30.0, vx_after=0.8),
    'R': dict(steps=3000, arms=['Cs', 'S1'], seeds=range(70000, 70005), rates=[0.5, 2.0]),
    'G': dict(steps=3000, arms=['Cs'], seeds=range(71000, 71003), vxs=[0.2, 0.35, 0.5, 0.65, 0.8, 1.0]),
}


def _case(scene, arm, variant, seed, **extra):
    tag = {'point': 'p', 'interval': 'i'}[variant]
    parts = [scene] + (['f%g' % extra['abs_hz']] if 'abs_hz' in extra else []) \
        + (['vx%g' % extra['pin_vx']] if scene == 'G' else []) + [tag, arm, 's%d' % seed]
    c = dict(BASE, **ARMS[arm], case_id='_'.join(parts), scene=scene, arm=arm, variant=variant,
             gyro_signal=variant, seed=seed, steps=SCENES[scene]['steps'],
             derived_seeds=episode_seed_metadata(seed, 0))
    for key in ('outage', 'vx_step_s', 'vx_after'):
        if key in SCENES[scene]:
            c[key] = SCENES[scene][key]
    c.update(extra)
    if isinstance(c['outage'], list):
        c['outage'] = tuple(c['outage'])
    return c


def build_cases():
    """运行顺序：主要场景在前（N、O），每个种子内轮换臂的顺序；variant 相邻，保证中途停下时配对完整。"""
    out = []
    for scene in ('N', 'O', 'V', 'R', 'G'):
        sc = SCENES[scene]
        for j, seed in enumerate(sc['seeds']):
            arms = sc['arms'][j % len(sc['arms']):] + sc['arms'][:j % len(sc['arms'])]
            if scene == 'R':
                for hz in sc['rates']:
                    for arm in arms:
                        for v in VARIANTS:
                            out.append(_case(scene, arm, v, seed, abs_hz=hz))
            elif scene == 'G':
                for vx in sc['vxs']:
                    out.append(_case(scene, 'Cs', 'point', seed, pin_vx=vx))
            else:
                for arm in arms:
                    for v in VARIANTS:
                        out.append(_case(scene, arm, v, seed))
    ids = [c['case_id'] for c in out]
    if len(ids) != len(set(ids)):
        raise RuntimeError('duplicate case ids')
    return out


def _deps():
    names = ('numpy', 'scipy', 'torch', 'mujoco', 'mujoco-warp', 'mjlab', 'tyro', 'psutil')
    out = {}
    for n in names:
        try:
            out[n] = importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError:
            out[n] = 'not_registered'
    return out


def protocol():
    cases = build_cases()
    for c in cases:
        c['command'] = command(c, Path('<attempt>'))[1:]          # 去掉解释器路径，便于核对
    prereg = OUT / 'PREREG.md'
    if not prereg.exists():
        raise RuntimeError('PREREG.md must exist before freezing')
    return dict(
        kind='design_family_formal', created_utc=datetime.now(timezone.utc).isoformat(),
        common=COMMON, arms=ARMS, variants=list(VARIANTS),
        scenes={k: {kk: (list(vv) if isinstance(vv, range) else vv) for kk, vv in v.items()} for k, v in SCENES.items()},
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
        for key in ('cases', 'common', 'arms', 'scenes', 'trace_schema_version', 'guarded_sha256',
                    'prereg_sha256', 'final_stop_reasons'):
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
    """合格的最终轨迹返回 (路径, stop_reason)；需要重试返回 None；不一致直接报错。"""
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
                  fusion_q_bias=case['q_b'], pin_vx=case['pin_vx'],
                  abs_hz=case.get('abs_hz', COMMON['abs_hz']), noise_std=COMMON['noise_std'],
                  gyro_bias=COMMON['gyro_bias'], abs_outage_start_s=case['outage'][0],
                  abs_outage_duration_s=case['outage'][1], trace_schema_version=plan['trace_schema_version'],
                  pin_vx_step_s=case.get('vx_step_s', 0.0), source_tree_sha256=plan['identity']['source_tree_sha256'],
                  checkpoint_sha256=plan['identity']['checkpoint_sha256'])
    if case['tau'] is not None:
        expect['feedback_offset_tau'] = case['tau']
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


def run_case(case, plan):
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


def run(limit=None, only=None):
    plan = freeze()
    todo_total = 0
    with batch_lock(OUT):
        ran = 0
        for case in plan['cases']:
            if only and not case['case_id'].startswith(only):
                continue
            state, _ = case_state(case, plan)
            if state == 'done':
                continue
            if limit is not None and ran >= limit:
                todo_total += 1
                continue
            verify(plan)
            rec = run_case(case, plan)
            ran += 1
            state, path = case_state(case, plan)
            if state != 'done':
                tail = (OUT / case['case_id'] / sorted(p.name for p in (OUT / case['case_id']).glob('attempt_*'))[-1]
                        / 'console.txt').read_text(encoding='utf-8', errors='replace')[-3000:]
                raise SystemExit('[FAIL] %s rc=%s（现场保留，修复后重新 --run 会用同一种子重试）\n%s'
                                 % (case['case_id'], rec.get('returncode'), tail))
            reason = json.loads((OUT / case['case_id'] / 'completed.json').read_text())['stop_reason']
            print('[ OK ] %-28s %-14s %.0fs' % (case['case_id'], reason, rec['wall_seconds']), flush=True)
    status(plan)


def status(plan=None):
    plan = plan or load_plan()
    rows = {}
    for case in plan['cases']:
        state, path = case_state(case, plan)
        s = rows.setdefault(case['scene'], dict(done=0, total=0, final={}))
        s['total'] += 1
        if state == 'done':
            s['done'] += 1
            reason = json.loads((OUT / case['case_id'] / 'completed.json').read_text())['stop_reason']
            s['final'][reason] = s['final'].get(reason, 0) + 1
    done = sum(s['done'] for s in rows.values())
    print('完成 %d / %d：' % (done, len(plan['cases'])) +
          '  '.join('%s %d/%d %s' % (k, v['done'], v['total'], v['final'] or '') for k, v in rows.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--only', default=None)
    args = ap.parse_args()
    if args.prepare_only:
        plan = freeze()
        print('协议已冻结：%d 条，protocol SHA256 %s' % (len(plan['cases']), digest(OUT / 'protocol.json')))
    if args.run:
        run(args.limit, args.only)
    if args.status:
        status()


if __name__ == '__main__':
    main()
