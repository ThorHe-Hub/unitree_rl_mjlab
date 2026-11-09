"""指令阶跃承受度正式批次的分析：PH1–PH3（Holm）与描述性预言 D1–D3。定义见 PREREG.md。

    python analyze_pulse_formal.py            # 35 条齐全后写 RESULTS.md、results.json
    python analyze_pulse_formal.py --partial  # 只把已完成部分打印到屏幕，不写文件
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
import run_pulse_formal as rp                       # noqa: E402
from run_heading_fixed_input import digest          # noqa: E402

DT = 0.02
DEG = 180 / math.pi
MARGIN = 0.10          # PH1、PH2 的非劣界 [°]
WIN = 6.0              # Δtilt 的前后窗口 [s]
PRIMARY = [('PH1', 0.05, 'noninf'), ('PH2', 0.125, 'noninf'), ('PH3', 0.5, 'sup')]
PREDICTED = {'PH1': '成立', 'PH2': '不确定', 'PH3': '成立'}
D2_RANGES = {1.0: (0.7, 1.4), 2.0: (1.7, 3.3)}
D3_RATIO = (26 * 0.8, 26 * 1.2)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        a = z['data'].copy()
        c = {str(n): i for i, n in enumerate(z['columns'])}
        meta = {k: z[k].item() for k in z.files if k not in ('data', 'columns')}
    return a, c, meta


def metrics(a, c, meta, case):
    n = len(a)
    k0 = int(round(case['pulse_at_s'] / DT))
    k1 = int(round((case['pulse_at_s'] + case['pulse_s']) / DT))
    w = int(round(WIN / DT))
    tilt = DEG * a[:, c['base_tilt']]
    psi = DEG * np.unwrap(a[:, c['gt_yaw']])
    out = dict(case_id=case['case_id'], amp=case['amp'], seed=case['seed'], t0=case['pulse_at_s'],
               stop_reason=meta['stop_reason'], fell=meta['stop_reason'] != 'budget_reached', n=n,
               tilt_pre=float(tilt[k0 - w:k0].max()) if n >= k0 else None, tilt_post=None, d_tilt=None)
    if n >= k0 + w:
        out['tilt_post'] = float(tilt[k0:k0 + w].max())
        out['d_tilt'] = out['tilt_post'] - out['tilt_pre']
    if n > k0:
        dev = np.abs(psi - psi[k0 - 250:k0].mean())
        over = np.nonzero(dev[k1:] > 5.0)[0]
        out.update(yaw_dev=float(dev[k0:].max()),
                   settle_5deg_s=float((over[-1] + 1) * DT) if len(over) else 0.0,
                   cmd_peak=float(np.abs(a[k0:min(n, k1 + 250), c['cmd_vyaw']]).max()))
    return out


def collect(partial):
    plan = rp.load_plan()
    rows = []
    for case in plan['cases']:
        state, path = rp.case_state(case, plan)
        if state != 'done':
            if partial:
                continue
            raise SystemExit('未完成：%s（用 --partial 看中间结果）' % case['case_id'])
        rows.append(metrics(*load(path), case))
    return plan, rows


def excess(rows, amp):
    """同种子配对：excess = Δtilt(A) − Δtilt(0)。返回 (数组, 是否有摔倒或缺失)。"""
    by = {(r['amp'], r['seed']): r for r in rows}
    vals, bad = [], False
    for s in rp.SEEDS:
        r, r0 = by.get((amp, s)), by.get((0.0, s))
        if r is None or r0 is None:
            continue
        if r['fell'] or r0['fell'] or r['d_tilt'] is None or r0['d_tilt'] is None:
            bad = True
            continue
        vals.append(r['d_tilt'] - r0['d_tilt'])
    return np.array(vals), bad


def tests(rows):
    out = []
    for hid, amp, kind in PRIMARY:
        x, bad = excess(rows, amp)
        t = dict(id=hid, amp=amp, kind=kind, n=len(x), values=[float(v) for v in x], fell_or_missing=bad,
                 mean=float(x.mean()) if len(x) else None, sd=float(x.std(ddof=1)) if len(x) > 1 else None)
        if bad:                                     # PREREG：摔倒时 PH1/PH2 不成立，PH3 成立
            t.update(p=1.0 if kind == 'noninf' else 0.0, bound=None)
        elif len(x) > 1:
            q = stats.t.ppf(0.95, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
            if kind == 'noninf':
                t.update(p=float(stats.ttest_1samp(x, MARGIN, alternative='less').pvalue),
                         bound=float(x.mean() + q), bound_kind='单侧 95% 上界')
            else:
                t.update(p=float(stats.ttest_1samp(x, 0.0, alternative='greater').pvalue),
                         bound=float(x.mean() - q), bound_kind='单侧 95% 下界')
        else:
            t.update(p=None, bound=None)
        out.append(t)
    ps = [(t['p'], k) for k, t in enumerate(out) if t['p'] is not None]
    run_max = 0.0
    for rank, (p, k) in enumerate(sorted(ps)):     # Holm：按 p 从小到大，乘以剩余个数，取累积最大
        run_max = max(run_max, min(1.0, p * (len(PRIMARY) - rank)))
        out[k]['p_holm'] = run_max
        out[k]['pass'] = run_max < 0.05
    return out


def describe(rows):
    by_amp = {}
    for a in rp.AMPS:
        rs = [r for r in rows if r['amp'] == a]
        x, bad = excess(rows, a) if a != 0.0 else (np.zeros(0), False)
        dev = [r['yaw_dev'] for r in rs if 'yaw_dev' in r]
        by_amp[a] = dict(
            n=len(rs), falls=sum(r['fell'] for r in rs),
            excess_mean=float(x.mean()) if len(x) else None, excess_sd=float(x.std(ddof=1)) if len(x) > 1 else None,
            d_tilt_mean=float(np.mean([r['d_tilt'] for r in rs if r['d_tilt'] is not None])) if rs else None,
            tilt_pre_mean=float(np.mean([r['tilt_pre'] for r in rs])) if rs else None,
            yaw_dev_mean=float(np.mean(dev)) if dev else None,
            yaw_ratio=float(np.mean(dev) / a) if dev and a > 0 else None,
            settle_mean=float(np.mean([r['settle_5deg_s'] for r in rs if 'settle_5deg_s' in r])) if rs else None,
            cmd_peak_mean=float(np.mean([r['cmd_peak'] for r in rs if 'cmd_peak' in r])) if rs else None)
    means = [by_amp[a]['excess_mean'] for a in rp.AMPS[1:] if by_amp[a]['excess_mean'] is not None]
    checks = dict(
        D1=dict(text='35 条都不摔倒', ok=sum(v['falls'] for v in by_amp.values()) == 0),
        D2=dict(text='超额均值随 A 单调不减；A=1.0 在 0.7–1.4°，A=2.0 在 1.7–3.3°',
                ok=all(b >= a for a, b in zip(means, means[1:]))
                and all(by_amp[a]['excess_mean'] is not None and lo <= by_amp[a]['excess_mean'] <= hi
                        for a, (lo, hi) in D2_RANGES.items())),
        D3=dict(text='A ≥ 0.25 时 yaw_dev/A 在 26 ±20% °/(rad/s)',
                ok=all(by_amp[a]['yaw_ratio'] is not None and D3_RATIO[0] <= by_amp[a]['yaw_ratio'] <= D3_RATIO[1]
                       for a in rp.AMPS if a >= 0.25)))
    return by_amp, checks


def fmt(x, f='%.3f'):
    return '—' if x is None else f % x


def report(plan, rows, ts, by_amp, checks):
    L = ['# 指令阶跃承受度正式批次　结果', '',
         '由 analyze_pulse_formal.py 生成（SHA256 %s）。协议 SHA256 %s，预注册 SHA256 %s。' % (
             digest(PROJECT / 'analyze_pulse_formal.py'), digest(rp.OUT / 'protocol.json'), plan['prereg_sha256']),
         '完成 %d / %d 条；摔倒 %d 条。' % (len(rows), len(plan['cases']), sum(r['fell'] for r in rows)), '',
         '## 主要假设（Holm，族 α = 0.05）', '',
         '| 编号 | A [rad/s] | 检验 | n | 超额均值 ± SD [°] | 单侧 95% 界 [°] | p | Holm p | 结论 | 预言 |',
         '|---|---|---|---|---|---|---|---|---|---|']
    for t in ts:
        L.append('| %s | %g | %s | %d | %s ± %s | %s | %s | %s | %s | %s |' % (
            t['id'], t['amp'], '均值 < %.2f°' % MARGIN if t['kind'] == 'noninf' else '均值 > 0', t['n'],
            fmt(t['mean']), fmt(t['sd']), fmt(t.get('bound')), fmt(t['p'], '%.2g'), fmt(t.get('p_holm'), '%.2g'),
            '未检验' if t.get('p_holm') is None else ('成立' if t['pass'] else '不成立'), PREDICTED[t['id']]))
    L += ['', '## 描述性预言', '']
    L += ['- %s：%s — %s' % (k, v['text'], '符合' if v['ok'] else '不符合') for k, v in checks.items()]
    L += ['', '## 各幅值汇总（5 个种子）', '',
          '| A [rad/s] | 摔倒 | Δtilt 均值 [°] | 超额均值 ± SD [°] | 航向最大偏离 [°] | 偏离/A [°/(rad/s)] | '
          '回到 5° 内 [s] | 指令峰值 [rad/s] |', '|---|---|---|---|---|---|---|---|']
    for a in rp.AMPS:
        d = by_amp[a]
        L.append('| %g | %d/%d | %s | %s ± %s | %s | %s | %s | %s |' % (
            a, d['falls'], d['n'], fmt(d['d_tilt_mean']), fmt(d['excess_mean']), fmt(d['excess_sd']),
            fmt(d['yaw_dev_mean'], '%.2f'), fmt(d['yaw_ratio'], '%.1f'), fmt(d['settle_mean'], '%.2f'),
            fmt(d['cmd_peak_mean'])))
    by = {(r['amp'], r['seed']): r for r in rows}
    L += ['', '## 超额按种子（检查步态相位的影响）', '',
          '| A [rad/s] | ' + ' | '.join('%d（t0=%.2f s）' % (s, rp.pulse_at(j)) for j, s in enumerate(rp.SEEDS)) + ' |',
          '|---|' + '---|' * len(rp.SEEDS)]
    for a in rp.AMPS[1:]:
        cells = []
        for s in rp.SEEDS:
            r, r0 = by.get((a, s)), by.get((0.0, s))
            ok = r and r0 and r['d_tilt'] is not None and r0['d_tilt'] is not None
            cells.append('%.3f' % (r['d_tilt'] - r0['d_tilt']) if ok else '—')
        L.append('| %g | %s |' % (a, ' | '.join(cells)))
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--partial', action='store_true')
    args = ap.parse_args()
    plan, rows = collect(args.partial)
    ts = tests(rows)
    by_amp, checks = describe(rows)
    text = report(plan, rows, ts, by_amp, checks)
    if args.partial:
        print(text)
        return
    (rp.OUT / 'RESULTS.md').write_text(text, encoding='utf-8')
    (rp.OUT / 'results.json').write_text(json.dumps(dict(
        rows=rows, tests=ts, by_amp={str(k): v for k, v in by_amp.items()}, checks=checks,
        analyzer_sha256=digest(PROJECT / 'analyze_pulse_formal.py'),
        protocol_sha256=digest(rp.OUT / 'protocol.json')), ensure_ascii=False, indent=1), encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
