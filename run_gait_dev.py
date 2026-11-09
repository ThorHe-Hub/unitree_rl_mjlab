"""步态运动学项（G）、变速（V）、指令跳变预算（P）的开发集。预言见 PREREG.md。

复用 run_design_family_dev 的命令构造与逐条前台执行：
    python run_gait_dev.py --list
    python run_gait_dev.py --run --limit 5 [--only G_]
    python run_gait_dev.py --analyze
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
import design_family as df                                         # noqa: E402
from design_family import interval_rate                            # noqa: E402
from run_design_family_dev import COMMON, DT, execute, done        # noqa: E402

OUT = PROJECT / 'unitree_heading_results/gait_dev_20260928'
BASE = dict(filt='fusion', q_b=1e-16, contact_friction=None, outage=(0.0, 0.0))
C_STAR = dict(arm='C', feedback_mode='fusion_heading', tau=None)
S15 = dict(arm='S1.5', feedback_mode='fusion_offset_smoothed', tau=1.5)
VX_LEVELS = [0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
PULSES = [0.25, 0.5, 1.0, 1.5, 2.0]


def cases():
    out = []
    for vx in VX_LEVELS:
        for s in (61000, 61001, 61002):
            out.append(dict(BASE, **C_STAR, case_id='G_vx%g_seed%d' % (vx, s), group='G',
                            seed=s, steps=3000, pin_vx=vx))
    for arm in (C_STAR, S15):
        for s in range(62000, 62005):
            out.append(dict(BASE, **arm, case_id='V_%s_seed%d' % (arm['arm'], s), group='V',
                            seed=s, steps=4500, pin_vx=0.5, vx_step_s=30.0, vx_after=0.8))
    for a in PULSES:
        for s in (63000, 63001, 63002):
            out.append(dict(BASE, **C_STAR, case_id='P_a%g_seed%d' % (a, s), group='P',
                            seed=s, steps=2000, pin_vx=0.5, pulse=a, pulse_at_s=20.0, pulse_s=1.0))
    return out


def load(path):
    with np.load(path, allow_pickle=False) as z:
        a = z['data']
        c = {str(n): i for i, n in enumerate(z['columns'])}
        meta = {k: z[k].item() for k in z.files if k not in ('data', 'columns')}
    if meta.get('trace_schema_version', 0) < 5:
        raise ValueError('need schema >= v5: %s' % path)
    return a, c, meta


def gait_metrics(a, c):
    """G：e = mean(ω_z − ψ̇) 分解为采样项与欧拉项；三种积分的末端误差。

    rec[k+1] 是区间 k→k+1 末端的点采样（帧判定见 PREREG 中途补记），与 ψ̇_k = Δψ_k/DT 对齐。"""
    psi = np.unwrap(a[:, c['gt_yaw']])
    phi, th = a[:, c['gt_roll']], a[:, c['gt_pitch']]
    rate = np.diff(psi) / DT
    rec = a[1:, [c['gt_wx'], c['gt_wy'], c['gt_wz']]]
    est = interval_rate(phi, th, psi)
    p0, t0 = phi[:-1], th[:-1]
    euler_rec = (rec[:, 1] * np.sin(p0) + rec[:, 2] * np.cos(p0)) / np.cos(t0)
    euler_est = (est[:, 1] * np.sin(p0) + est[:, 2] * np.cos(p0)) / np.cos(t0)
    s = slice(250, None)
    m = lambda x: float(x[s].mean())
    e = m(rec[:, 2] - rate)
    sampling, euler = m(rec[:, 2] - est[:, 2]), m(est[:, 2] - rate)
    loop = -m(p0 * est[:, 1])                           # 一阶近似的欧拉项
    T = (len(rate) - 250) * DT
    end = lambda x: float(np.rad2deg(abs(m(x - rate)) * T))
    return dict(e=e, sampling=sampling, euler=euler, loop=loop,
                identity_resid_rec=m(euler_rec - rate), identity_resid_est=m(euler_est - rate),
                loop_explains_e=float(1 - abs(e - loop) / abs(e)) if e else float('nan'),
                loop_explains_euler=float(1 - abs(euler - loop) / abs(euler)) if euler else float('nan'),
                end_err_wz_deg=end(rec[:, 2]), end_err_euler_rec_deg=end(euler_rec),
                end_err_euler_est_deg=end(euler_est),
                roll_amp_deg=float(np.rad2deg(p0[s].std() * math.sqrt(2))),
                pitch_amp_deg=float(np.rad2deg(t0[s].std() * math.sqrt(2))))


def est_err(a, c, lo, hi):
    psi = np.unwrap(a[:, c['gt_yaw']])
    e = a[:, c['feedback_heading']] - (psi - psi[0])
    return float(np.rad2deg(np.sqrt(np.mean(e[int(lo / DT):int(hi / DT)] ** 2))))


def pulse_metrics(a, c, meta, case):
    t0, t1 = case['pulse_at_s'], case['pulse_at_s'] + case['pulse_s']
    k0, k1 = int(round(t0 / DT)), int(round(t1 / DT))
    n = len(a)
    tilt = np.rad2deg(a[:, c['base_tilt']])
    psi = np.rad2deg(np.unwrap(a[:, c['gt_yaw']]))
    base = psi[max(0, k0 - 250):k0].mean()
    dev = np.abs(psi - base)
    over = np.where(dev[k1:] > 5.0)[0] if k1 < n else np.array([])
    # 脉冲结束后最后一次超出 5° 的时刻：此后一直在 5° 以内
    settle = float((over[-1] + 1) * DT) if len(over) else 0.0
    return dict(stop_reason=meta['stop_reason'], completed_s=n * DT,
                tilt_max_pre=float(tilt[max(0, k0 - 250):k0].max()),
                tilt_max_post=float(tilt[k0:min(n, k1 + 250)].max()) if k0 < n else float('nan'),
                yaw_dev_max_deg=float(dev[k0:].max()) if k0 < n else float('nan'),
                settle_5deg_s=settle if k1 < n else float('nan'))



def v_offline(paths):
    """V3：在变速闭环路径上离线重放 q_b 网格（τ=0 与 τ=1.5），比较恒速段与变速后段的最优 q_b。"""
    from analyze_design_family import evaluate, best
    tr = [df.load_trace(p) for p in paths]
    qs = [1e-16, 1e-12, 1e-10, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    rows = evaluate(tr, COMMON['noise_std'], COMMON['abs_sigma'], qs, [0.0, 1.5],
                    [(10, 30), (60, 90)])
    out = {}
    for w in ('10-30', '60-90'):
        for tau in (0.0, 1.5):
            b = best(rows, w, tau=tau)
            out['%s_tau%g' % (w, tau)] = dict(q_b=b['q_b'], rms=b['windows'][w]['expected_rms'],
                                               rms_at_1e16=next(r['windows'][w]['expected_rms'] for r in rows
                                                                if r['q_b'] == 1e-16 and r['tau'] == tau))
    return out


def analyze(_args):
    rows, L = [], ['# 步态运动学项 / 变速 / 指令跳变预算：结果（开发集）', '',
                   '由 `run_gait_dev.py --analyze` 生成。预言见 `PREREG.md`。', '']
    for c in cases():
        d = OUT / c['case_id']
        if not done(d):
            continue
        a, col, meta = load(next(d.glob('*.npz')))
        r = dict(c, stop_reason=meta['stop_reason'])
        if c['group'] == 'G':
            r.update(gait_metrics(a, col))
        elif c['group'] == 'V':
            r.update(err_pre=est_err(a, col, 10, 30), err_post=est_err(a, col, 60, 90),
                     err_trans=est_err(a, col, 30, 40))
        else:
            r.update(pulse_metrics(a, col, meta, c))
        rows.append(r)
    L.append('完成 %d / %d 条。' % (len(rows), len(cases())))

    g = [r for r in rows if r['group'] == 'G']
    if g:
        L += ['', '## G　运动学项分解', '',
              'e = mean(ω_z,点采样 − ψ̇) = 采样项 mean(ω_z − est_z) + 欧拉项 mean(est_z − ψ̇)；est 为相邻姿态反推的区间平均角速度。', '',
              '| vx | n | e [rad/s] | 采样项 | 欧拉项 | 环面积近似 | 恒等式残差 rec / est | 环面积解释 e / 欧拉项 | 末端误差 ω_z / 欧拉(rec) / 欧拉(est) | 横滚/俯仰幅值 [°] |',
              '|---:|---:|---:|---:|---:|---:|---|---|---|---|']
        for vx in VX_LEVELS:
            rs = [r for r in g if r['pin_vx'] == vx]
            if not rs:
                continue
            m = lambda k: float(np.mean([r[k] for r in rs]))
            L.append('| %g | %d | %+.5f | %+.5f | %+.5f | %+.5f | %+.1e / %+.1e | %.0f%% / %.0f%% | %.2f° / %.2f° / %.3f° | %.2f / %.2f |'
                     % (vx, len(rs), m('e'), m('sampling'), m('euler'), m('loop'),
                        m('identity_resid_rec'), m('identity_resid_est'),
                        100 * m('loop_explains_e'), 100 * m('loop_explains_euler'),
                        m('end_err_wz_deg'), m('end_err_euler_rec_deg'), m('end_err_euler_est_deg'),
                        m('roll_amp_deg'), m('pitch_amp_deg')))
    v = [r for r in rows if r['group'] == 'V']
    if v:
        L += ['', '## V　变速 0.5→0.8 @30 s（估计误差 RMS）', '',
              '| 臂 | n | [10,30) 恒速 | [30,40) 变速后 10 s | [60,90) 稳定后 |', '|---|---:|---:|---:|---:|']
        for arm in ('C', 'S1.5'):
            rs = [r for r in v if r['arm'] == arm]
            if rs:
                f = lambda k: '%.3f ± %.3f' % (np.mean([r[k] for r in rs]), np.std([r[k] for r in rs], ddof=1) if len(rs) > 1 else 0)
                L.append('| %s | %d | %s | %s | %s |' % (arm, len(rs), f('err_pre'), f('err_trans'), f('err_post')))
        paths = [next((OUT / r['case_id']).glob('*.npz')) for r in v if r['arm'] == 'C']
        if paths:
            off = v_offline(paths)
            L += ['', 'V3 离线（C 臂的闭环路径，期望 RMS）：', '', '| 窗口 | τ | 最优 q_b | 最优 RMS | q_b=1e-16 时 |', '|---|---:|---:|---:|---:|']
            L += ['| %s | %s | %.0e | %.3f° | %.3f° |' % (k.split('_')[0], k.split('_tau')[1], x['q_b'], x['rms'], x['rms_at_1e16'])
                  for k, x in off.items()]
    p = [r for r in rows if r['group'] == 'P']
    if p:
        L += ['', '## P　指令跳变预算（C*，脉冲 1 s @20 s）', '',
              '| 幅值 [rad/s] | n | 摔倒 | tilt 最大 前 5 s | tilt 最大 脉冲起 6 s | 航向最大偏离 | 回到 5° 内 [s] |',
              '|---:|---:|---:|---:|---:|---:|---:|']
        for amp in PULSES:
            rs = [r for r in p if r['pulse'] == amp]
            if rs:
                fell = sum(r['stop_reason'] != 'budget_reached' for r in rs)
                m = lambda k: float(np.nanmean([r[k] for r in rs]))
                L.append('| %g | %d | %d | %.2f° | %.2f° | %.1f° | %.2f |'
                         % (amp, len(rs), fell, m('tilt_max_pre'), m('tilt_max_post'), m('yaw_dev_max_deg'), m('settle_5deg_s')))
    (OUT / 'RESULTS.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    (OUT / 'results.json').write_text(json.dumps(rows, indent=1, default=str), encoding='utf-8')
    print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--only', default=None)
    args = ap.parse_args()
    if args.list:
        for c in cases():
            print('%-24s %s' % (c['case_id'], 'done' if done(OUT / c['case_id']) else ''))
    if args.run:
        execute(cases(), OUT, args.limit, args.only)
    if args.analyze:
        analyze(args)


if __name__ == '__main__':
    main()
