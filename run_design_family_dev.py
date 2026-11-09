"""设计族的闭环抽点确认（开发集，不进论文证据链）。

目的：检查离线结论在闭环里是否成立。离线结论见 doc/design_family_offline.md：
  * 各取最优后 C 胜 B（BMI088：0.802° vs 2.395°）
  * 在 C 的最优 q_b=1e-16 上，S(τ=0.3) 精度几乎不损失（0.805°），最大跳变降为约 1/5
  * 中断恢复时 S(τ) 大幅削减恢复跳变
预言与判据写在 unitree_heading_results/design_family_dev_20260928/PREREG.md，先于任何结果。

每次只跑一条，便于在前台分段执行，避免长后台任务被系统回收：
    python run_design_family_dev.py --list
    python run_design_family_dev.py --run --limit 4     # 跑接下来 4 条未完成的
    python run_design_family_dev.py --analyze
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
from sensor_profiles import resolve                     # noqa: E402
from run_heading_pilot import CHECKPOINT                # noqa: E402

OUT = PROJECT / 'unitree_heading_results/design_family_dev_20260928'
DT = 0.02
SEEDS = [60000, 60001, 60002, 60003, 60004]
BMI = resolve('bmi088')
COMMON = dict(pin_vx=0.5, friction=0.8, kp=0.91, ki=0.0, kd=0.01,
              noise_std=float(BMI.noise_std(DT)), gyro_bias=0.02, abs_sigma=0.05, abs_hz=1.0)
# (臂名, filt, feedback_mode, q_b, τ)
ARMS = [('B', 'fusion', 'integrated', 1e-4, None),
        ('C', 'fusion', 'fusion_heading', 1e-16, None),
        ('S0.3', 'fusion', 'fusion_offset_smoothed', 1e-16, 0.3),
        ('S1', 'fusion', 'fusion_offset_smoothed', 1e-16, 1.0)]
OUTAGE_ARMS = ['B', 'C', 'S0.3']
SCENES = {'nominal': dict(steps=3000, contact_friction=None, outage=(0.0, 0.0)),
          'outage30': dict(steps=4500, contact_friction=0.6, outage=(20.0, 30.0))}


def cases():
    out = []
    for scene, sc in SCENES.items():
        for arm, filt, mode, q, tau in ARMS:
            if scene == 'outage30' and arm not in OUTAGE_ARMS:
                continue
            for s in SEEDS:
                out.append(dict(case_id='%s_%s_seed%d' % (scene, arm, s), scene=scene, arm=arm,
                                filt=filt, feedback_mode=mode, q_b=q, tau=tau, seed=s, **sc))
    return out


def command(c, case_dir):
    cmd = [sys.executable, '-X', 'utf8', 'scripts/play.py', 'Unitree-G1-Flat',
           '--checkpoint-file', CHECKPOINT, '--device', 'cuda:0',
           '--pin-cmd', 'True', '--pin-vx', str(c.get('pin_vx', COMMON['pin_vx'])),
           '--fixed-body', 'True', '--friction', str(COMMON['friction']),
           '--kp', str(COMMON['kp']), '--ki', str(COMMON['ki']), '--kd', str(COMMON['kd']),
           '--rig.noise-std', repr(COMMON['noise_std']), '--rig.gyro-bias', str(COMMON['gyro_bias']),
           '--rig.abs-sigma', str(COMMON['abs_sigma']), '--rig.abs-hz', str(c.get('abs_hz', COMMON['abs_hz'])),
           '--rig.abs-dropout', '0.0', '--rig.fusion-q-bias', repr(c['q_b']),
           '--rig.abs-outage-start-s', str(c['outage'][0]),
           '--rig.abs-outage-duration-s', str(c['outage'][1]),
           '--rig.filt', c['filt'], '--feedback-mode', c['feedback_mode'],
           '--seed', str(c['seed']), '--batch-steps', str(c['steps']), '--repeats', '1',
           '--no-terminations', 'False', '--batch-stop-on-done', 'True',
           '--output-dir', str(case_dir)]
    if c['tau'] is not None:
        cmd += ['--feedback-offset-tau', str(c['tau'])]
    if c['contact_friction'] is not None:
        cmd += ['--contact-friction', str(c['contact_friction'])]
    for flag, key in (('--pin-vx-step-s', 'vx_step_s'), ('--pin-vx-after', 'vx_after'),
                      ('--vyaw-pulse', 'pulse'), ('--vyaw-pulse-at-s', 'pulse_at_s'),
                      ('--vyaw-pulse-s', 'pulse_s'), ('--rig.gyro-signal', 'gyro_signal')):
        if c.get(key) is not None:
            cmd += [flag, str(c[key])]
    return cmd


def done(case_dir):
    return case_dir.exists() and len(list(case_dir.glob('*.npz'))) == 1


def execute(case_list, out_dir, limit=None, only=None):
    """逐条前台运行未完成的条目。fell/terminated 也算完成（轨迹照常保存），只有进程失败才停。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'cases.json').write_text(json.dumps(case_list, indent=1), encoding='utf-8')
    env = {**os.environ, 'GIT_PYTHON_REFRESH': 'quiet', 'PYTHONIOENCODING': 'utf-8',
           'PYTHONUNBUFFERED': '1',
           'PYTHONPATH': str(PROJECT) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    todo = [c for c in case_list if not done(out_dir / c['case_id'])]
    if only:
        todo = [c for c in todo if c['case_id'].startswith(only)]
    todo = todo[:limit] if limit else todo
    for c in todo:
        d = out_dir / c['case_id']
        for stale in d.glob('*') if d.exists() else []:   # 未完成的残留现场
            stale.unlink()
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(d / 'console.txt', 'w', encoding='utf-8') as log:
            rc = subprocess.call(command(c, d), cwd=str(PROJECT), env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
        if rc != 0 or not done(d):
            raise SystemExit('[FAIL] %s rc=%d（保留现场）' % (c['case_id'], rc))
        print('[ OK ] %s %.0fs' % (c['case_id'], time.time() - t0), flush=True)
    left = sum(not done(out_dir / c['case_id']) for c in case_list)
    print('剩余 %d / %d' % (left, len(case_list)))


def run(args):
    execute(cases(), OUT, args.limit, args.only)


def metrics(c, path):
    """一条闭环轨迹的指标。反馈量用 feedback_heading（本步 advance 之后的值）。"""
    import design_family as df
    with np.load(path, allow_pickle=False) as z:
        a = z['data']
        col = {str(n): i for i, n in enumerate(z['columns'])}
        stop = z['stop_reason'].item()
    psi = np.unwrap(a[:, col['gt_yaw']])
    psi -= psi[0]
    fb = a[:, col['feedback_heading']]
    e = fb - psi
    cmd = a[:, col['cmd_vyaw']]
    deg = lambda x: float(np.rad2deg(x))
    rms = lambda lo, hi: deg(np.sqrt(np.mean(e[int(lo / DT):int(hi / DT)] ** 2)))
    out = dict(stop_reason=stop, n=len(a),
               tilt_mean=deg(a[250:, col['base_tilt']].mean()),
               tilt_p99=deg(np.percentile(a[250:, col['base_tilt']], 99)),
               phys_late=deg(np.sqrt(np.mean(psi[-1500:] ** 2))))
    t0, dur = c['outage']
    if dur == 0:
        out.update(err=rms(30, 60),
                   fb_jump=deg(np.abs(np.diff(fb[250:])).max()),
                   cmd_jump=float(np.abs(np.diff(cmd[250:])).max()))
    else:
        k1 = int(round((t0 + dur) / DT))
        out.update(err_before=rms(10, t0), err_during=rms(t0, t0 + dur),
                   err_after10=rms(t0 + dur, t0 + dur + 10), err_late=rms(t0 + dur + 10, len(a) * DT),
                   fb_jump_reacq=deg(np.abs(np.diff(fb[k1 - 1:k1 + 100])).max()),
                   cmd_jump_reacq=float(np.abs(np.diff(cmd[k1 - 1:k1 + 100])).max()),
                   fb_jump_steady=deg(np.abs(np.diff(fb[250:int(t0 / DT)])).max()))
    # 在这条闭环路径上重放离线模型：smooth(f, i, τ) 必须复现记录的反馈量
    tr = df.load_trace(path)
    parts = df.decompose([tr], tr.meta['noise_std'], c['q_b'], COMMON['abs_sigma'])
    tau = 0.0 if c['feedback_mode'] == 'fusion_heading' else (math.inf if c['tau'] is None else c['tau'])
    s = sum(df.smooth(parts[k][0], parts[k][1], tau) for k in df.SOURCES)
    out['offline_replay_max_err'] = float(np.abs(s[0] - e).max())
    return out


def paired(rows, scene, a, b, key):
    """同种子配对差 b−a。返回 (均值, 标准差, b 更小的种子数, n)。"""
    x = {r['seed']: r[key] for r in rows if r['scene'] == scene and r['arm'] == a}
    y = {r['seed']: r[key] for r in rows if r['scene'] == scene and r['arm'] == b}
    s = sorted(set(x) & set(y))
    d = np.array([y[k] - x[k] for k in s])
    return (float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else float('nan'),
            int((d < 0).sum()), len(d)) if len(d) else None


def analyze(_args):
    rows = []
    for c in cases():
        d = OUT / c['case_id']
        if done(d):
            rows.append({**c, **metrics(c, next(d.glob('*.npz')))})
    if not rows:
        raise SystemExit('没有完成的轨迹')
    bad = [r['case_id'] for r in rows if r['stop_reason'] != 'budget_reached']
    worst = max(r['offline_replay_max_err'] for r in rows)
    L = ['# 设计族闭环抽点确认：结果（开发集）', '',
         '由 `run_design_family_dev.py --analyze` 生成。预言与判据见 `PREREG.md`。', '',
         '- 完成 %d / %d 条；非 budget_reached：%s' % (len(rows), len(cases()), bad or '无'),
         '- 离线模型在闭环路径上复现记录反馈量的最大误差：%.1e rad' % worst, '']

    def table(scene, keys):
        arms = [a for a, *_ in ARMS if any(r['scene'] == scene and r['arm'] == a for r in rows)]
        out = ['| 臂 | n | ' + ' | '.join(keys) + ' |', '|---|---:|' + '---:|' * len(keys)]
        for a in arms:
            rs = [r for r in rows if r['scene'] == scene and r['arm'] == a]
            out.append('| %s | %d | ' % (a, len(rs)) + ' | '.join(
                '%.3f ± %.3f' % (np.mean([r[k] for r in rs]), np.std([r[k] for r in rs], ddof=1) if len(rs) > 1 else 0)
                for k in keys) + ' |')
        return out

    L += ['## 名义工况（60 s，[30,60) s）', '',
          '误差/跳变单位为度，cmd_jump 为 rad/s，tilt 为度；均值 ± 种子间标准差。', '']
    L += table('nominal', ['err', 'fb_jump', 'cmd_jump', 'tilt_mean', 'tilt_p99', 'phys_late']) + ['']
    L += ['配对差（同种子，后者减前者；"更小"为后者更小的种子数）：', '',
          '| 对比 | 指标 | 均值 | 标准差 | 更小 / n |', '|---|---|---:|---:|---:|']
    for a, b, k in [('B', 'C', 'err'), ('C', 'S0.3', 'err'), ('C', 'S1', 'err'),
                    ('C', 'S0.3', 'fb_jump'), ('C', 'S1', 'fb_jump'), ('C', 'S0.3', 'tilt_p99')]:
        p = paired(rows, 'nominal', a, b, k)
        if p:
            L.append('| %s → %s | %s | %+.3f | %.3f | %d / %d |' % (a, b, k, *p))
    L += ['', '## 中断 30 s（[20,50) s 无参考）', '']
    L += table('outage30', ['err_before', 'err_during', 'err_after10', 'err_late',
                            'fb_jump_reacq', 'cmd_jump_reacq', 'fb_jump_steady', 'tilt_p99']) + ['']
    L += ['| 对比 | 指标 | 均值 | 标准差 | 更小 / n |', '|---|---|---:|---:|---:|']
    for a, b, k in [('B', 'C', 'err_late'), ('C', 'S0.3', 'err_after10'),
                    ('C', 'S0.3', 'fb_jump_reacq'), ('C', 'S0.3', 'cmd_jump_reacq')]:
        p = paired(rows, 'outage30', a, b, k)
        if p:
            L.append('| %s → %s | %s | %+.3f | %.3f | %d / %d |' % (a, b, k, *p))
    (OUT / 'RESULTS.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    (OUT / 'results.json').write_text(json.dumps(rows, indent=1, default=str), encoding='utf-8')
    print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--only', default=None, help='只跑 case_id 以此开头的条目')
    args = ap.parse_args()
    if args.list:
        for c in cases():
            print('%-28s %s' % (c['case_id'], 'done' if done(OUT / c['case_id']) else ''))
    if args.run:
        run(args)
    if args.analyze:
        analyze(args)


if __name__ == '__main__':
    main()


