"""设计族 (q_b, τ) 平面的离线分析：误差预算、最优点、跳变、时长与中断。

方法见 design_family.py。本脚本把"期望 RMS"作为主指标：

    E[e²] = mean_{路径,窗口}(e_kin + e_bias)²  +  Var_gyro(理论)  +  Var_ref(理论)

确定性部分（步态运动学项 + 注入零偏）用已记录路径重放；两个噪声来源
用精确协方差递推给期望，不依赖 20 条路径上恰好抽到的那组噪声。
这样选最优点时，噪声部分没有"在同一批样本上挑最好"的乐观偏差。
同时报告用真实噪声实现重放的经验 RMS，二者应当接近。

用法：
    python analyze_design_family.py            # 全部场景，写报告与 JSON
    python analyze_design_family.py --quick    # 粗网格，冒烟用
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

import design_family as df

PROJECT = Path(__file__).resolve().parent
OUT_DIR = PROJECT / 'unitree_heading_results/design_family_offline_20260928'
DOC = PROJECT / 'doc/design_family_offline.md'
INF = float('inf')

BMI088 = 1.728e-3          # sensor_profiles.resolve('bmi088').noise_std(0.02)，取 4 位
Q_GRID = [1e-16, 1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
TAU_GRID = [0.0, 0.1, 0.3, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0, 300.0, INF]
QUICK_Q = [1e-16, 1e-8, 1e-5, 1e-4]
QUICK_TAU = [0.0, 1.0, 10.0, 30.0, INF]

# 场景：(名称, 路径 glob, 估计器用的 σg, 评价窗口 [s))
SCENARIOS = [
    ('BMI088 · 零偏 0.02', 'unitree_heading_results/bias_002/[BC]_seed*/**/*.npz', BMI088, (30, 60)),
    ('BMI088 · 零偏 0',    'build/heading_fixed_input_20260906/[BC]_seed*/*.npz',   BMI088, (30, 60)),
    ('stress · 零偏 0.02', 'unitree_heading_results/bias_002/[BC]_seed*/**/*.npz', 0.15,   (30, 60)),
    ('stress · 零偏 0',    'build/heading_fixed_input_20260906/[BC]_seed*/*.npz',   0.15,   (30, 60)),
]
DURATION = ('300 s · 零偏 0.02', 'unitree_heading_results/duration_300s_20260908/**/*.npz')
DURATION_WINDOWS = [(30, 60), (120, 180), (240, 300)]
OUTAGE = ('中断 30 s', 'unitree_heading_results/reference_outage_20260912/outage30/**/*.npz')


def rad2deg(x):
    return float(np.rad2deg(x))


def evaluate(traces, sigma_g, sigma_a, qs, taus, windows):
    """对 (q, τ) 网格算各窗口的误差预算与跳变。返回嵌套 dict 列表。"""
    rows = []
    upd = traces[0].upd
    for q in qs:
        parts = df.decompose(traces, sigma_g, q, sigma_a)
        var = df.noise_variance(upd, sigma_g, q, sigma_a, taus)
        for j, tau in enumerate(taus):
            s = {src: df.smooth(parts[src][0], parts[src][1], tau) for src in df.SOURCES}
            det = s['kin'] + s['bias']
            total = det + s['gyro'] + s['ref']
            jump = np.abs(np.diff(total[:, 250:], axis=1)).max(axis=1).mean()
            row = dict(q_b=q, tau=tau, jump_deg=rad2deg(jump), windows={})
            for (w0, w1) in windows:
                sl = slice(int(w0 / df.DT), int(w1 / df.DT))
                mse_det = float((det[:, sl] ** 2).mean())
                mse_kin = float((s['kin'][:, sl] ** 2).mean())
                mse_bias = float((s['bias'][:, sl] ** 2).mean())
                v_g = float(var['gyro'][j, sl].mean())
                v_r = float(var['ref'][j, sl].mean())
                row['windows']['%d-%d' % (w0, w1)] = dict(
                    expected_rms=rad2deg(math.sqrt(mse_det + v_g + v_r)),
                    empirical_rms=rad2deg(math.sqrt(float((total[:, sl] ** 2).mean()))),
                    mse_det=mse_det, mse_kin=mse_kin, mse_bias=mse_bias,
                    var_gyro=v_g, var_ref=v_r)
            rows.append(row)
    return rows


def outage_eval(traces, sigma_g, sigma_a, q, taus):
    """中断场景：中断期、恢复瞬间、恢复后的误差。时刻取自元数据。"""
    m = traces[0].meta
    t0 = float(m['abs_outage_start_s'])
    t1 = t0 + float(m['abs_outage_duration_s'])
    k1 = int(round(t1 / df.DT))
    parts = df.decompose(traces, sigma_g, q, sigma_a)
    var = df.noise_variance(traces[0].upd, sigma_g, q, sigma_a, taus)
    out = []
    for j, tau in enumerate(taus):
        s = {src: df.smooth(parts[src][0], parts[src][1], tau) for src in df.SOURCES}
        total = sum(s.values())
        det = s['kin'] + s['bias']

        def exp_rms(a, b):
            sl = slice(int(a / df.DT), int(b / df.DT))
            return rad2deg(math.sqrt(float((det[:, sl] ** 2).mean())
                                     + var['gyro'][j, sl].mean() + var['ref'][j, sl].mean()))
        d = np.abs(np.diff(total, axis=1))
        out.append(dict(tau=tau,
                        before=exp_rms(10, t0), during=exp_rms(t0, t1),
                        after_10s=exp_rms(t1, t1 + 10), after_late=exp_rms(t1 + 10, len(total[0]) * df.DT),
                        reacq_jump_deg=rad2deg(d[:, k1 - 1:k1 + 100].max(axis=1).mean()),
                        steady_jump_deg=rad2deg(d[:, 500:int(t0 / df.DT)].max(axis=1).mean())))
    return out


def best(rows, window, **fix):
    """在满足 fix 条件的行里取期望 RMS 最小者。"""
    cand = [r for r in rows if all(r[k] == v for k, v in fix.items())]
    return min(cand, key=lambda r: r['windows'][window]['expected_rms'])


def fmt_tau(t):
    return '∞' if math.isinf(t) else ('%g' % t)


def budget_line(label, r, w):
    x = r['windows'][w]
    g = lambda v: rad2deg(math.sqrt(max(v, 0.0)))
    return ('| %s | %.0e | %s | **%.3f°** | %.3f° | %.3f° | %.3f° | %.3f° | %.3f° | %.3f° |'
            % (label, r['q_b'], fmt_tau(r['tau']), x['expected_rms'], x['empirical_rms'],
               g(x['mse_kin']), g(x['mse_bias']), g(x['var_gyro']), g(x['var_ref']), r['jump_deg']))


BUDGET_HEAD = ['| 方案 | q_b | τ [s] | 期望 RMS | 经验 RMS | 运动学 | 零偏 | 陀螺噪声 | 参考噪声 | 最大跳变 |',
               '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    qs, taus = (QUICK_Q, QUICK_TAU) if args.quick else (Q_GRID, TAU_GRID)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = dict(q_grid=qs, tau_grid=[fmt_tau(t) for t in taus], scenarios={})
    L = ['# 设计族 (q_b, τ) 离线分析', '',
         '由 `analyze_design_family.py` 生成。冻结已记录路径，按误差来源叠加重放；'
         '陀螺与参考噪声用精确协方差递推给期望。**离线结果，需闭环抽点确认。**', '',
         '- 期望 RMS = √(确定性部分均方 + 陀螺噪声方差 + 参考噪声方差)，是主指标。',
         '- 经验 RMS：用路径上真实抽到的那组噪声重放，作对照。',
         '- 各来源列是该来源**单独存在**时的 RMS。运动学与零偏之间有交叉项，所以各列平方和不等于总量。',
         '- 最大跳变：[5 s, 末尾) 内反馈量单步变化绝对值的最大值，按路径取平均。',
         '- B = (q_b 最优, τ=∞)，C = (q_b 最优, τ=0)，S* = 整个平面的最优点。', '']

    for name, pat, sg, win in SCENARIOS:
        t0 = time.time()
        tr = df.load_glob(pat)
        w = '%d-%d' % win
        rows = evaluate(tr, sg, 0.05, qs, taus, [win])
        b = best(rows, w, tau=INF)
        c = best(rows, w, tau=0.0)
        s = best(rows, w)
        result['scenarios'][name] = dict(sigma_g=sg, n_paths=len(tr), window=w, rows=rows)
        L += ['## %s（σg=%g，%d 条路径，窗口 %s s）' % (name, sg, len(tr), w), ''] + BUDGET_HEAD
        L += [budget_line('B', b, w), budget_line('C', c, w), budget_line('S*', s, w), '']
        # τ 剖面：在 C 的最优 q_b 与 B 的最优 q_b 上
        for lab, q in (('C 的最优 q_b', c['q_b']), ('B 的最优 q_b', b['q_b'])):
            prof = [r for r in rows if r['q_b'] == q]
            L += ['%s = %.0e 上的 τ 剖面（期望 RMS / 最大跳变）：' % (lab, q), '',
                  '| τ | ' + ' | '.join(fmt_tau(r['tau']) for r in prof) + ' |',
                  '|---|' + '---:|' * len(prof),
                  '| RMS | ' + ' | '.join('%.3f' % r['windows'][w]['expected_rms'] for r in prof) + ' |',
                  '| 跳变 | ' + ' | '.join('%.3f' % r['jump_deg'] for r in prof) + ' |', '']
        print('%s: B %.3f C %.3f S* %.3f (q=%g tau=%s)  %.0fs'
              % (name, b['windows'][w]['expected_rms'], c['windows'][w]['expected_rms'],
                 s['windows'][w]['expected_rms'], s['q_b'], fmt_tau(s['tau']), time.time() - t0), flush=True)

    # 时长：300 s 路径，σg 取记录值（stress）与 BMI088 两档
    name, pat = DURATION
    tr = df.load_glob(pat)
    L += ['## 时长：%s（%d 条路径）' % (name, len(tr)), '',
          '| σg | 窗口 [s] | B | C | S* | S* 的 (q_b, τ) |', '|---|---|---:|---:|---:|---|']
    result['duration'] = {}
    for sg in (0.15, BMI088):
        wins = DURATION_WINDOWS
        rows = evaluate(tr, sg, 0.05, qs, taus, wins)
        result['duration'][str(sg)] = rows
        for w0, w1 in wins:
            w = '%d-%d' % (w0, w1)
            b, c, s = best(rows, w, tau=INF), best(rows, w, tau=0.0), best(rows, w)
            L.append('| %g | %s | %.3f° | %.3f° | %.3f° | (%.0e, %s) |'
                     % (sg, w, b['windows'][w]['expected_rms'], c['windows'][w]['expected_rms'],
                        s['windows'][w]['expected_rms'], s['q_b'], fmt_tau(s['tau'])))
    L.append('')
    print('duration done', flush=True)

    # 中断：原记录 σg=0.15、q_b=1e-8；另给 BMI088 档
    name, pat = OUTAGE
    tr = df.load_glob(pat)
    result['outage'] = {}
    L += ['## %s（%d 条路径，中断 %g–%g s）' % (name, len(tr), tr[0].meta['abs_outage_start_s'],
                                           tr[0].meta['abs_outage_start_s'] + tr[0].meta['abs_outage_duration_s']), '']
    for sg in (0.15, BMI088):
        o = outage_eval(tr, sg, 0.05, 1e-8, taus)
        result['outage'][str(sg)] = o
        L += ['σg=%g，q_b=1e-8：' % sg, '',
              '| τ | 中断前 | 中断中 | 恢复后 10 s | 其后 | 恢复跳变 | 平时跳变 |',
              '|---:|---:|---:|---:|---:|---:|---:|']
        L += ['| %s | %.3f° | %.3f° | %.3f° | %.3f° | %.3f° | %.3f° |'
              % (fmt_tau(r['tau']), r['before'], r['during'], r['after_10s'], r['after_late'],
                 r['reacq_jump_deg'], r['steady_jump_deg']) for r in o]
        L.append('')
    print('outage done', flush=True)

    (OUT_DIR / 'grid.json').write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    DOC.write_text('\n'.join(L), encoding='utf-8')
    print('写入 %s 与 %s' % (DOC, OUT_DIR / 'grid.json'))


if __name__ == '__main__':
    main()

