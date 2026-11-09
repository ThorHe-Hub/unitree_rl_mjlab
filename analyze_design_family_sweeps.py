"""设计族的两组离线扫描：参考更新率 × σa，以及中断时长。

方法与 analyze_design_family.py 相同（冻结路径 + 按来源叠加 + 噪声理论期望），
区别是用 design_family.load_trace 的反事实参数在同一物理路径上换参考设置。
这一步逐位等价于 SensorRig 在该设置下的输出（tests/test_design_family.py 断言）。

回答两个问题（预言写在 doc/design_family_sweeps_prereg.md，先于本脚本的结果）：
  1. τ 的可用窗口与参考周期 T_ref 是什么关系？
  2. 按名义精度选出的整定，在更长的中断下代价多大？是否存在兼顾两者的整定？

用法：
    python analyze_design_family_sweeps.py [--quick]
"""
import argparse
import glob
import json
import math
from pathlib import Path
import time

import numpy as np

import design_family as df
from analyze_design_family import BMI088, INF, evaluate, best, fmt_tau, rad2deg

PROJECT = Path(__file__).resolve().parent
OUT_DIR = PROJECT / 'unitree_heading_results/design_family_offline_20260928'
DOC = PROJECT / 'doc/design_family_sweeps.md'

RATE_PATHS = ('BMI088 · 零偏 0.02', 'unitree_heading_results/bias_002/[BC]_seed*/**/*.npz')
RATES = [0.25, 0.5, 1.0, 2.0, 5.0]
SIGMA_AS = [0.02, 0.05, 0.10]
Q_RATE = [1e-16, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
TAU_FINE = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0,
            15.0, 20.0, 30.0, 50.0, 100.0, INF]
ACC_TOL = 1.05     # 精度容差：期望 RMS ≤ 1.05 × C*
JUMP_TOL = 1.10    # 平滑容差：最大跳变 ≤ 1.10 × 该 q_b 上的最小跳变

OUTAGE_PATHS = ('300 s 路径', 'unitree_heading_results/duration_300s_20260908/**/*.npz')
OUTAGE_START = 60.0
OUTAGE_DURS = [10.0, 30.0, 60.0, 120.0]
Q_OUT = [1e-16, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3]
TAU_OUT = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, INF]


def tau_window(rows, q, w, c_star):
    """在给定 q_b 上：精度不劣于 ACC_TOL×C* 且跳变不超过 JUMP_TOL×最小跳变 的 τ 区间。"""
    prof = sorted([r for r in rows if r['q_b'] == q], key=lambda r: r['tau'])
    jmin = min(r['jump_deg'] for r in prof)
    ok_acc = [r['tau'] for r in prof if r['windows'][w]['expected_rms'] <= ACC_TOL * c_star]
    ok_jump = [r['tau'] for r in prof if r['jump_deg'] <= JUMP_TOL * jmin]
    both = sorted(set(ok_acc) & set(ok_jump))
    return dict(tau_jump_min=min(ok_jump) if ok_jump else None,
                tau_acc_max=max(ok_acc) if ok_acc else None,
                window=(both[0], both[-1]) if both else None,
                jump_floor=jmin)


def rate_sweep(qs, taus):
    name, pat = RATE_PATHS
    w, win = '30-60', (30, 60)
    out, L = [], ['## 一、参考更新率 × σa（%s，窗口 %s s）' % (name, w), '',
                  '精度容差 %.2f×C*，平滑容差 %.2f×最小跳变。T_ref = 1/f。' % (ACC_TOL, JUMP_TOL), '',
                  '| f [Hz] | σa [rad] | B* | C*（q_b） | 2σa/√N 近似 | τ 窗口 [s]（C* 的 q_b 上） | 窗口/T_ref | 最小跳变 |',
                  '|---:|---:|---:|---:|---:|---|---|---:|']
    for hz in RATES:
        tr = [df.load_trace(p, abs_hz=hz) for p in sorted(glob.glob(pat, recursive=True))]
        for sa in SIGMA_AS:
            t0 = time.time()
            rows = evaluate(tr, BMI088, sa, qs, taus, [win])
            b, c = best(rows, w, tau=INF), best(rows, w, tau=0.0)
            cs = c['windows'][w]['expected_rms']
            tw = tau_window(rows, c['q_b'], w, cs)
            n_mid = 45 * hz                      # 窗口中点 45 s 处的参考点数
            approx = rad2deg(2 * sa / math.sqrt(n_mid))
            win_s = ('%s–%s' % (fmt_tau(tw['window'][0]), fmt_tau(tw['window'][1]))) if tw['window'] else '空'
            ratio = ('%.2g–%.2g' % (tw['window'][0] * hz, tw['window'][1] * hz)
                     if tw['window'] and not math.isinf(tw['window'][1]) else '—')
            L.append('| %g | %g | %.3f° | %.3f°（%.0e） | %.3f° | %s | %s | %.3f° |'
                     % (hz, sa, b['windows'][w]['expected_rms'], cs, c['q_b'], approx,
                        win_s, ratio, tw['jump_floor']))
            out.append(dict(hz=hz, sigma_a=sa, b_star=b['windows'][w]['expected_rms'], b_q=b['q_b'],
                            c_star=cs, c_q=c['q_b'], approx=approx, **tw, rows=rows))
            print('rate %g Hz sa %g: B* %.3f C* %.3f window %s  %.0fs'
                  % (hz, sa, b['windows'][w]['expected_rms'], cs, tw['window'], time.time() - t0), flush=True)
    return out, L


def _exp_rms(det, var_g, var_r, a, b):
    sl = slice(int(round(a / df.DT)), int(round(b / df.DT)))
    return rad2deg(math.sqrt(float((det[:, sl] ** 2).mean()) + var_g[sl].mean() + var_r[sl].mean()))


def outage_sweep(qs, taus):
    name, pat = OUTAGE_PATHS
    paths = sorted(glob.glob(pat, recursive=True))
    t0s = OUTAGE_START
    res, L = {}, ['## 二、中断时长（%s，%d 条，中断从 %g s 开始，σg=BMI088，σa=0.05）' % (name, len(paths), t0s), '',
                  'J_nom = [30,60) s 期望 RMS（中断前，各时长相同）；J_out = [中断开始, 结束) 期望 RMS（含中断与恢复）。',
                  '"兼顾"= 使 max(J_nom/min J_nom, J_out/min J_out) 最小的整定。', '']
    for dur in OUTAGE_DURS:
        t1 = t0s + dur
        tr = [df.load_trace(p, outage=(t0s, dur)) for p in paths]
        T = len(tr[0].psi) * df.DT
        k1 = int(round(t1 / df.DT))
        rows = []
        for q in qs:
            parts = df.decompose(tr, BMI088, q, 0.05)
            var = df.noise_variance(tr[0].upd, BMI088, q, 0.05, taus)
            for j, tau in enumerate(taus):
                s = {src: df.smooth(parts[src][0], parts[src][1], tau) for src in df.SOURCES}
                det = s['kin'] + s['bias']
                total = sum(s.values())
                vg, vr = var['gyro'][j], var['ref'][j]
                d = np.abs(np.diff(total, axis=1))
                rows.append(dict(q_b=q, tau=tau,
                                 J_nom=_exp_rms(det, vg, vr, 30, 60),
                                 during=_exp_rms(det, vg, vr, t0s, t1),
                                 after10=_exp_rms(det, vg, vr, t1, t1 + 10),
                                 late=_exp_rms(det, vg, vr, t1 + 10, T),
                                 J_out=_exp_rms(det, vg, vr, t0s, T),
                                 reacq_jump=rad2deg(d[:, k1 - 1:k1 + 100].max(axis=1).mean())))
        nmin = min(r['J_nom'] for r in rows)
        omin = min(r['J_out'] for r in rows)
        for r in rows:
            r['score'] = max(r['J_nom'] / nmin, r['J_out'] / omin)
        picks = [('B*（名义最优）', min((r for r in rows if math.isinf(r['tau'])), key=lambda r: r['J_nom'])),
                 ('C*（名义最优）', min((r for r in rows if r['tau'] == 0), key=lambda r: r['J_nom'])),
                 ('S*（名义最优）', min(rows, key=lambda r: r['J_nom'])),
                 ('中断最优', min(rows, key=lambda r: r['J_out'])),
                 ('兼顾', min(rows, key=lambda r: r['score']))]
        L += ['### 中断 %g s（%g–%g s）' % (dur, t0s, t1), '',
              '| 选择 | q_b | τ | J_nom | 中断中 | 恢复后 10 s | 其后 | J_out | 恢复跳变 | score |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        L += ['| %s | %.0e | %s | %.3f° | %.3f° | %.3f° | %.3f° | %.3f° | %.3f° | %.2f |'
              % (lab, r['q_b'], fmt_tau(r['tau']), r['J_nom'], r['during'], r['after10'],
                 r['late'], r['J_out'], r['reacq_jump'], r['score']) for lab, r in picks]
        L.append('')
        res[str(dur)] = rows
        print('outage %gs: C* J_out %.3f  B* J_out %.3f  robust (%g, %s) score %.2f'
              % (dur, picks[1][1]['J_out'], picks[0][1]['J_out'], picks[4][1]['q_b'],
                 fmt_tau(picks[4][1]['tau']), picks[4][1]['score']), flush=True)
    return res, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--only', choices=('rate', 'outage'), default=None)
    args = ap.parse_args()
    global RATES, SIGMA_AS, OUTAGE_DURS
    qr, tf, qo, to = Q_RATE, TAU_FINE, Q_OUT, TAU_OUT
    if args.quick:
        RATES, SIGMA_AS, OUTAGE_DURS = [1.0], [0.05], [30.0]
        qr, tf, qo, to = [1e-16, 1e-4], [0.0, 1.0, INF], [1e-16, 1e-4], [0.0, 1.0, INF]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    L = ['# 设计族离线扫描：参考更新率与中断时长', '',
         '由 `analyze_design_family_sweeps.py` 生成。预言见 `doc/design_family_sweeps_prereg.md`。'
         '**离线结果，需闭环抽点确认。**', '']
    result = {}
    if args.only in (None, 'rate'):
        r, lines = rate_sweep(qr, tf)
        result['rate'] = r
        L += lines + ['']
    if args.only in (None, 'outage'):
        o, lines = outage_sweep(qo, to)
        result['outage'] = o
        L += lines
    suffix = '_quick' if args.quick else ''
    (OUT_DIR / ('sweeps%s.json' % suffix)).write_text(
        json.dumps(result, ensure_ascii=False, default=str, indent=1), encoding='utf-8')
    (DOC.with_name(DOC.stem + suffix + '.md')).write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('written', DOC.with_name(DOC.stem + suffix + '.md'))


if __name__ == '__main__':
    main()
