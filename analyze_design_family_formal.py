"""设计族正式批次的分析：严格按 PREREG.md 第 3–7 节。

    python analyze_design_family_formal.py            # 写 RESULTS.md 与 results.json
    python analyze_design_family_formal.py --partial  # 批次未完成时，只分析已完成的配对（标注为中期）

本脚本不在运行时的指纹守护列表内；运行时记录其 SHA256，任何修改都会在 RESULTS.md 里体现。
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
# 部分轨迹路径达 260 字符（DEVIATIONS.md D1），用扩展长度前缀读取同一目录
import os                                                         # noqa: E402
if os.name == 'nt' and 'DF_FORMAL_OUT' not in os.environ:
    os.environ['DF_FORMAL_OUT'] = '\\\\?\\' + str(PROJECT / 'unitree_heading_results' / 'design_family_formal_20260929')
import design_family as df                                        # noqa: E402
import run_design_family_formal as rf                             # noqa: E402

DT = 0.02
DEG = 180.0 / math.pi


def rms(x):
    return float(DEG * np.sqrt(np.mean(x ** 2)))


def trace_metrics(case, path):
    with np.load(path, allow_pickle=False) as z:
        a = z['data']
        c = {str(n): i for i, n in enumerate(z['columns'])}
        stop = z['stop_reason'].item()
    n = len(a)
    psi = np.unwrap(a[:, c['gt_yaw']])
    psi -= psi[0]
    fb = a[:, c['feedback_heading']]
    e = fb - psi
    cmd = a[:, c['cmd_vyaw']]
    tilt = a[250:, c['base_tilt']] * DEG
    win = lambda lo, hi: rms(e[int(round(lo / DT)):min(n, int(round(hi / DT)))]) if lo / DT < n else float('nan')
    out = dict(stop_reason=stop, n=n, complete=stop == 'budget_reached',
               fb_jump=float(DEG * np.abs(np.diff(fb[250:])).max()),
               cmd_jump=float(np.abs(np.diff(cmd[250:])).max()),
               tilt_p99=float(np.percentile(tilt, 99)), tilt_max=float(tilt.max()))
    sc = case['scene']
    if sc in ('N', 'R', 'G'):
        out['err'] = win(30, 60)
    elif sc == 'O':
        k1 = int(round(50.0 / DT))
        out.update(err_before=win(10, 20), err_during=win(20, 50), err_after10=win(50, 60),
                   err_late=win(60, 90),
                   reacq_jump=float(DEG * np.abs(np.diff(fb[k1 - 1:k1 + 100])).max()) if n > k1 + 100 else float('nan'),
                   reacq_cmd_jump=float(np.abs(np.diff(cmd[k1 - 1:k1 + 100])).max()) if n > k1 + 100 else float('nan'))
    elif sc == 'V':
        out.update(err_pre=win(10, 30), err_post=win(60, 90))
    # 有效性 1：离线模型复现记录的反馈量
    tr = df.load_trace(path)
    parts = df.decompose([tr], tr.meta['noise_std'], case['q_b'], rf.COMMON['abs_sigma'])
    tau = 0.0 if case['feedback_mode'] == 'fusion_heading' else (math.inf if case['tau'] is None else case['tau'])
    s = sum(df.smooth(parts[k][0], parts[k][1], tau) for k in df.SOURCES)[0]
    out['replay_err'] = float(np.abs(s - e).max())
    # 有效性 2：四元数区间角速度 vs 欧拉角重建
    rec = df.interval_rate(a[:, c['gt_roll']], a[:, c['gt_pitch']], np.unwrap(a[:, c['gt_yaw']]))[:, 2]
    out['interval_check'] = float(np.abs(a[1:, c['raw_yaw_interval']] - rec).max())
    if sc == 'G':
        rate = np.diff(np.unwrap(a[:, c['gt_yaw']])) / DT
        pt, iv = a[1:, c['raw_yaw']], a[1:, c['raw_yaw_interval']]
        s_ = slice(250, None)
        e_ = float((pt - rate)[s_].mean())
        samp, eul = float((pt - iv)[s_].mean()), float((iv - rate)[s_].mean())
        T = (len(rate) - 250) * DT
        out.update(e=e_, sampling=samp, euler=eul, decomp_resid=abs(e_ - samp - eul),
                   end_err_point=float(DEG * abs(e_) * T), end_err_interval=float(DEG * abs(eul) * T))
    return out


def paired(rows, sel_a, sel_b, key):
    """同种子配对差 b − a。sel 为 (scene, variant, arm, extra)。只用两边都完整的种子。"""
    def pick(sel):
        scene, variant, arm, extra = sel
        return {r['seed']: r for r in rows if r['scene'] == scene and r['variant'] == variant
                and r['arm'] == arm and all(r.get(k) == v for k, v in extra.items())}
    A, B = pick(sel_a), pick(sel_b)
    seeds = sorted(s for s in set(A) & set(B) if A[s]['complete'] and B[s]['complete'])
    excluded = sorted((set(A) | set(B)) - set(seeds))
    d = np.array([B[s][key] - A[s][key] for s in seeds])
    if len(d) < 2:
        return dict(n=len(d), excluded=excluded)
    m, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / math.sqrt(len(d))
    tcrit = stats.t.ppf(0.975, len(d) - 1)
    p = float(stats.ttest_1samp(d, 0.0).pvalue) if sd > 0 else (0.0 if m != 0 else 1.0)
    return dict(n=len(d), mean=m, sd=sd, ci=(m - tcrit * se, m + tcrit * se),
                upper95_one_sided=m + stats.t.ppf(0.95, len(d) - 1) * se,
                lower95_one_sided=m - stats.t.ppf(0.95, len(d) - 1) * se,
                p=p, n_negative=int((d < 0).sum()), excluded=excluded, d=d.tolist())


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj.tolist()


# PREREG 第 4、5 节（编号、场景、variant、比较、指标、界值）
PRIMARY = [
    ('H1', 'N', 'point', 'Bs', 'Cs', 'err'), ('H2', 'N', 'interval', 'Bs', 'Cs', 'err'),
    ('H3', 'N', 'point', 'Cs', 'S1', 'fb_jump'), ('H4', 'N', 'interval', 'Cs', 'S1', 'fb_jump'),
    ('H5', 'O', 'point', 'Bs', 'Cs', 'err_late'), ('H6', 'O', 'interval', 'Bs', 'Cs', 'err_late'),
    ('H7', 'O', 'point', 'Cs', 'S1', 'reacq_jump'), ('H8', 'O', 'interval', 'Cs', 'S1', 'reacq_jump'),
]
NONINF = [
    ('NI1', 'N', 'point', 'Cs', 'S1', 'err', 0.10), ('NI2', 'N', 'interval', 'Cs', 'S1', 'err', 0.10),
    ('NI3', 'O', 'point', 'Cs', 'S1', 'err_after10', 0.30), ('NI4', 'O', 'interval', 'Cs', 'S1', 'err_after10', 0.30),
]
# 离线预言（PREREG 表中数值），用于有效性检查 4：偏离 >50% 列出
PREDICTED = {('N', 'point', 'Bs', 'err'): 2.390, ('N', 'point', 'Cs', 'err'): 0.797, ('N', 'point', 'S1', 'err'): 0.809,
             ('N', 'interval', 'Bs', 'err'): 2.380, ('N', 'interval', 'Cs', 'err'): 0.769,
             ('N', 'interval', 'S1', 'err'): 0.781,
             ('O', 'point', 'Cs', 'err_late'): 0.741, ('O', 'point', 'S1', 'err_late'): 0.753,
             ('O', 'point', 'Bs', 'err_late'): 62.13,
             ('V', 'point', 'Cs', 'err_post'): 1.627, ('V', 'point', 'C8', 'err_post'): 1.072,
             ('V', 'interval', 'Cs', 'err_post'): 0.624, ('V', 'interval', 'C8', 'err_post'): 1.018}


def fmt(r, digits=3):
    if 'mean' not in r:
        return '| n=%d（不足） |' % r['n']
    f = '%%+.%df' % digits
    return ('%d | ' + f + ' | %.' + str(digits) + 'f | [' + f + ', ' + f + '] | %d/%d') % (
        r['n'], r['mean'], r['sd'], r['ci'][0], r['ci'][1], r['n_negative'], r['n'])


def group(rows, **sel):
    return [r for r in rows if all(r.get(k) == v for k, v in sel.items())]


def mean_sd(rs, key):
    v = [r[key] for r in rs if r['complete'] and np.isfinite(r.get(key, np.nan))]
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v)) if v else (np.nan, np.nan, 0)


def analyze(partial):
    plan = rf.load_plan()
    rows, missing = [], []
    for case in plan['cases']:
        state, path = rf.case_state(case, plan)
        if state != 'done':
            missing.append(case['case_id'])
            continue
        rows.append({**{k: case[k] for k in ('case_id', 'scene', 'arm', 'variant', 'seed', 'q_b', 'tau',
                                             'feedback_mode', 'pin_vx')},
                     'abs_hz': case.get('abs_hz', 1.0), **trace_metrics(case, path)})
    if missing and not partial:
        raise SystemExit('%d 条未完成；用 --partial 做中期分析' % len(missing))
    me = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    L = ['# 设计族正式批次：结果' + ('（**中期，%d/%d 条**）' % (len(rows), len(plan['cases'])) if missing else ''), '',
         '由 `analyze_design_family_formal.py` 生成（SHA256 `%s`）。预注册：`PREREG.md`（SHA256 `%s`），协议 `protocol.json`。'
         % (me[:16], plan['prereg_sha256'][:16]), '']
    # 完成率与失败
    fails = [r for r in rows if not r['complete']]
    L += ['## 0. 完成情况', '', '完成 %d / %d 条；未到预算（摔倒/超时）%d 条%s。' % (
        len(rows), len(plan['cases']), len(fails),
        ('：' + '、'.join('%s（%s，%.1f s）' % (r['case_id'], r['stop_reason'], r['n'] * DT) for r in fails)) if fails else ''), '']
    # 有效性
    worst_replay = max(r['replay_err'] for r in rows)
    worst_int = max(r['interval_check'] for r in rows if r['variant'] == 'interval') if any(r['variant'] == 'interval' for r in rows) else float('nan')
    L += ['## 1. 有效性检查（PREREG §7）', '',
          '| 检查 | 结果 | 判据 | 通过 |', '|---|---|---|---|',
          '| 1 离线重放复现反馈量 | 最大 %.1e rad | < 1e-9 | %s |' % (worst_replay, '是' if worst_replay < 1e-9 else '**否**'),
          '| 2 interval：四元数 vs 欧拉角重建 | 最大 %.1e rad/s | < 1e-4 | %s |' % (worst_int, '是' if not worst_int >= 1e-4 else '**否**'),
          '| 3 源码/权重指纹 | 每条在 case_state 中逐条核对 | 与协议一致 | 是 |', '']
    if worst_replay >= 1e-9 or worst_int >= 1e-4:
        L.append('**有效性检查不通过，以下结果不可用。**')
    # 各臂描述
    L += ['## 2. 各臂描述（均值 ± SD，完整轨迹）', '']
    for scene, keys in (('N', ['err', 'fb_jump', 'cmd_jump', 'tilt_p99']),
                        ('O', ['err_before', 'err_during', 'err_after10', 'err_late', 'reacq_jump', 'reacq_cmd_jump', 'tilt_p99']),
                        ('V', ['err_pre', 'err_post', 'fb_jump', 'tilt_p99']),
                        ('R', ['err', 'fb_jump'])):
        L += ['### %s' % scene, '', '| variant | 臂 | 参考 [Hz] | n | ' + ' | '.join(keys) + ' |',
              '|---|---|---:|---:|' + '---:|' * len(keys)]
        for hz in sorted({r['abs_hz'] for r in group(rows, scene=scene)}):
            for v in rf.VARIANTS:
                for arm in rf.SCENES[scene]['arms']:
                    rs = group(rows, scene=scene, variant=v, arm=arm, abs_hz=hz)
                    if rs:
                        cells = ['%.3f ± %.3f' % mean_sd(rs, k)[:2] for k in keys]
                        L.append('| %s | %s | %g | %d | %s |' % (v, arm, hz, mean_sd(rs, keys[0])[2], ' | '.join(cells)))
        L.append('')
    return plan, rows, L, missing


def hypotheses(plan, rows, L):
    res = {}
    prim = [(hid, paired(rows, (sc, v, a, {}), (sc, v, b, {}), key), sc, v, a, b, key)
            for hid, sc, v, a, b, key in PRIMARY]
    adj = holm([r.get('p', 1.0) for _, r, *_ in prim])
    L += ['## 3. 主要假设（PREREG §4；双侧配对 t，族内 Holm，α=0.05）', '',
          'd = 后者 − 前者（同种子）。"更小"为 d<0 的种子数。', '',
          '| 编号 | 场景/variant | 比较 | 指标 | n | 均值 d | SD | 95% CI | 更小 | p | Holm p | 结论 |',
          '|---|---|---|---|---:|---:|---:|---|---:|---:|---:|---|']
    for (hid, r, sc, v, a, b, key), p_adj in zip(prim, adj):
        ok = 'mean' in r and r['mean'] < 0 and p_adj < 0.05
        L.append('| %s | %s/%s | %s → %s | %s | %s | %.2e | %.2e | %s |'
                 % (hid, sc, v, a, b, key, fmt(r), r.get('p', float('nan')), p_adj,
                    '成立' if ok else '**不成立**'))
        res[hid] = dict(r, p_holm=p_adj, supported=bool(ok))
    L.append('')
    L += ['## 4. 非劣效（PREREG §5；单侧 95% 上界 < 界值）', '',
          '| 编号 | 场景/variant | 比较 | 指标 | n | 均值 d | SD | 95% CI | 更小 | 单侧界 | 界值 | 结论 |',
          '|---|---|---|---|---:|---:|---:|---|---:|---|---:|---|']
    for hid, sc, v, a, b, key, margin in NONINF:
        r = paired(rows, (sc, v, a, {}), (sc, v, b, {}), key)
        ok = 'mean' in r and r['upper95_one_sided'] < margin
        L.append('| %s | %s/%s | %s → %s | %s | %s | %+.3f | %+.2f | %s |'
                 % (hid, sc, v, a, b, key, fmt(r), r.get('upper95_one_sided', float('nan')), margin,
                    '成立' if ok else '**不成立**'))
        res[hid] = dict(r, margin=margin, supported=bool(ok))
    for v in rf.VARIANTS:                                   # NI5：tilt p99 等效（TOST ±0.10°）
        r = paired(rows, ('N', v, 'Cs', {}), ('N', v, 'S1', {}), 'tilt_p99')
        ok = 'mean' in r and r['upper95_one_sided'] < 0.10 and r['lower95_one_sided'] > -0.10
        band = '[%+.3f, %+.3f]' % (r.get('lower95_one_sided', float('nan')), r.get('upper95_one_sided', float('nan')))
        L.append('| NI5 | N/%s | Cs → S1 | tilt_p99 | %s | %s | ±0.10 | %s |'
                 % (v, fmt(r), band, '成立' if ok else '**不成立**'))
        res['NI5_' + v] = dict(r, supported=bool(ok))
    L.append('')
    return res


def secondary(plan, rows, L, res):
    L += ['## 5. 次要假设（PREREG §6；描述性，不校正）', '',
          '### variant 效应（同种子，d = interval − point）', '',
          '| 场景 | 臂 | 指标 | n | 均值 d | SD | 95% CI | 更小 |', '|---|---|---|---:|---:|---:|---|---:|']
    for sc, arm, key in (('N', 'Cs', 'err'), ('N', 'S1', 'err'), ('N', 'S1', 'fb_jump'), ('N', 'Bs', 'err'),
                         ('O', 'Cs', 'err_after10'), ('O', 'S1', 'reacq_jump')):
        r = paired(rows, (sc, 'point', arm, {}), (sc, 'interval', arm, {}), key)
        L.append('| %s | %s | %s | %s |' % (sc, arm, key, fmt(r)))
        res['variant_%s_%s_%s' % (sc, arm, key)] = r
    L += ['', '### V：变速', '',
          '| 编号 | variant | 比较 | 指标 | n | 均值 d | SD | 95% CI | 更小 | 预言 | 符合 |',
          '|---|---|---|---|---:|---:|---:|---|---:|---|---|']
    for hid, v, a, b, key, want in (('V1', 'point', 'Cs', 'C8', 'err_post', '<0'),
                                    ('V2', 'interval', 'Cs', 'C8', 'err_post', '>0')):
        r = paired(rows, ('V', v, a, {}), ('V', v, b, {}), key)
        good = 'mean' in r and ((r['mean'] < 0) if want == '<0' else (r['mean'] > 0))
        L.append('| %s | %s | %s → %s | %s | %s | d%s | %s |'
                 % (hid, v, a, b, key, fmt(r), want, '是' if good else '**否**'))
        res[hid] = dict(r, consistent=bool(good))
    for v in rf.VARIANTS:                                   # V3：C* 变速前 → 变速后（同条轨迹）
        rs = [x for x in group(rows, scene='V', variant=v, arm='Cs') if x['complete']]
        d = np.array([x['err_post'] - x['err_pre'] for x in rs])
        if len(d):
            want_le0 = v == 'interval'
            good = (d.mean() <= 0) if want_le0 else (d.mean() > 0)
            L.append('| V3 | %s | Cs [10,30) → [60,90) | err | %d | %+.3f | %.3f | — | %d/%d | %s | %s |'
                     % (v, len(d), d.mean(), d.std(ddof=1) if len(d) > 1 else 0.0, (d < 0).sum(), len(d),
                        'd≤0' if want_le0 else 'd>0', '是' if good else '**否**'))
            res['V3_' + v] = d.tolist()
    L += ['', '### R：参考率', '',
          '| f [Hz] | variant | C* 误差 | S1 − C* 误差 [95% CI] | 最大跳变 C* → S1 | R2 预言 C* | 偏差 |',
          '|---:|---|---:|---|---|---:|---:|']
    for hz in (0.5, 2.0):
        for v in rf.VARIANTS:
            r = paired(rows, ('R', v, 'Cs', dict(abs_hz=hz)), ('R', v, 'S1', dict(abs_hz=hz)), 'err')
            cs = mean_sd(group(rows, scene='R', variant=v, arm='Cs', abs_hz=hz), 'err')[0]
            j0 = mean_sd(group(rows, scene='R', variant=v, arm='Cs', abs_hz=hz), 'fb_jump')[0]
            j1 = mean_sd(group(rows, scene='R', variant=v, arm='S1', abs_hz=hz), 'fb_jump')[0]
            pred = math.sqrt(0.2 ** 2 + (0.765 / math.sqrt(hz)) ** 2)
            ci = ('%+.3f [%+.3f, %+.3f]' % (r['mean'], r['ci'][0], r['ci'][1])) if 'mean' in r else '—'
            L.append('| %g | %s | %.3f | %s | %.3f → %.3f | %.3f | %+.0f%% |'
                     % (hz, v, cs, ci, j0, j1, pred, 100 * (cs / pred - 1)))
            res['R_%g_%s' % (hz, v)] = dict(r, c_star=cs, pred=pred)
    g = [x for x in rows if x['scene'] == 'G' and x['complete']]
    if g:
        L += ['', '### G：运动学项分解（point）', '',
              '| vx | n | e [rad/s] | 采样项 | 欧拉项 | 分解残差（最大） | 60 s 末端误差 点采样 / 区间 |',
              '|---:|---:|---:|---:|---:|---:|---|']
        for vx in rf.SCENES['G']['vxs']:
            rs = [x for x in g if x['pin_vx'] == vx]
            if rs:
                m = lambda k: float(np.mean([x[k] for x in rs]))
                L.append('| %g | %d | %+.5f | %+.5f | %+.5f | %.1e | %.2f° / %.3f° |'
                         % (vx, len(rs), m('e'), m('sampling'), m('euler'), max(x['decomp_resid'] for x in rs),
                            m('end_err_point'), m('end_err_interval')))
    L += ['', '## 6. 闭环与离线预言（PREREG §7.4，偏离 >50% 加粗）', '',
          '| 场景 | variant | 臂 | 指标 | 闭环 | 离线预言 | 偏离 |', '|---|---|---|---|---:|---:|---:|']
    for (sc, v, arm, key), pred in PREDICTED.items():
        m = mean_sd(group(rows, scene=sc, variant=v, arm=arm), key)[0]
        if np.isfinite(m):
            dev = m / pred - 1
            mark = '**' if abs(dev) > 0.5 else ''
            L.append('| %s | %s | %s | %s | %.3f | %.3f | %s%+.0f%%%s |' % (sc, v, arm, key, m, pred, mark, 100 * dev, mark))
    L.append('')


def posthoc(plan, rows, L):
    """事后分析（**不在预注册内**），解释第 5、6 节中与预言不符或需要澄清的地方。"""
    cases = {c['case_id']: c for c in plan['cases']}
    path = lambda cid: rf.case_state(cases[cid], plan)[1]

    def load(cid):
        with np.load(path(cid), allow_pickle=False) as z:
            return z['data'].copy(), {str(n): i for i, n in enumerate(z['columns'])}

    L += ['## 7. 事后分析（非预注册，见 DEVIATIONS.md D2–D5）', '',
          '### 7.1 两种跳变口径（D2）', '',
          'PREREG 中 H3/H4 与 variant 效应引用的离线预言值（0.188°、0.036°）是**估计误差 fb−ψ** 的跳变；'
          '终点指标 fb_jump 是**反馈量 fb** 的跳变，后者还包含步态本身的航向摆动。两种口径并列如下（N，10 种子均值）：', '',
          '| variant | 臂 | max\\|Δfb\\|（终点） | max\\|Δ(fb−ψ)\\|（离线预言口径） |', '|---|---|---:|---:|']
    for v in rf.VARIANTS:
        for arm in ('Cs', 'S1'):
            fbj, ej = [], []
            for s in rf.SCENES['N']['seeds']:
                a, c = load('N_%s_%s_s%d' % (v[0], arm, s))
                psi = np.unwrap(a[:, c['gt_yaw']]); psi -= psi[0]
                fb = a[:, c['feedback_heading']]
                fbj.append(DEG * np.abs(np.diff(fb[250:])).max())
                ej.append(DEG * np.abs(np.diff((fb - psi)[250:])).max())
            L.append('| %s | %s | %.3f° | %.3f° |' % (v, arm, np.mean(fbj), np.mean(ej)))
    L += ['', '### 7.2 闭环误差低于离线预言约 30% 的来源（D4）', '',
          '在闭环路径上把误差精确拆成确定性部分（运动学 + 零偏）与噪声部分（陀螺 + 参考），'
          '并与噪声的理论期望比较（[30,60) s）：', '',
          '| 场景 | variant | 参考 [Hz] | det² | noise² 实测 | noise² 期望 | 比值 | 比值 < 1 的种子 | 2·det·noise |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    from analyze_design_family import BMI088
    for scene, v, hz in (('N', 'point', None), ('N', 'interval', None), ('R', 'point', 0.5), ('R', 'point', 2.0)):
        tag = ('f%g_' % hz) if hz else ''
        tr = [df.load_trace(path('%s_%s%s_Cs_s%d' % (scene, tag, v[0], s))) for s in rf.SCENES[scene]['seeds']]
        parts = df.decompose(tr, BMI088, 1e-16, rf.COMMON['abs_sigma'])
        var = df.noise_variance(tr[0].upd, BMI088, 1e-16, rf.COMMON['abs_sigma'], [0.0])
        sl = slice(1500, 3000)
        det = (parts['kin'][0] + parts['bias'][0])[:, sl]
        noise = (parts['gyro'][0] + parts['ref'][0])[:, sl]
        expn = var['gyro'][0, sl].mean() + var['ref'][0, sl].mean()
        ratio = (noise ** 2).mean(axis=1) / expn
        L.append('| %s | %s | %g | %.4f | %.4f | %.4f | %.2f | %d/%d | %+.4f |'
                 % (scene, v, hz or 1, DEG ** 2 * (det ** 2).mean(), DEG ** 2 * (noise ** 2).mean(), DEG ** 2 * expn,
                    ratio.mean(), (ratio < 1).sum(), len(ratio), DEG ** 2 * 2 * (det * noise).mean()))
    L += ['', '单位 deg²。交叉项很小，说明闭环反馈没有让确定性部分与噪声明显相关；差距来自本批种子上**参考噪声的实现偏小**。'
          'C* 的误差主要是一个"水平估计误差"随机变量（有效自由度约 1–2），单个种子波动很大，'
          '所以 10 个种子的绝对误差水平不稳，**配对差**（同种子共享同一份参考噪声）才稳健。'
          'N、O、V、R 共用种子 70000–70009，参考噪声实现相同，四个场景一起偏低不是独立证据。', '',
          '### 7.3 G：两种"区间积分"口径的 60 s 末端误差（D3）', '',
          '| vx | 点采样 ω_z | 单轴区间 ω_z（本脚本第 5 节） | 区间三轴 + 欧拉率公式（开发集 G4 口径） |',
          '|---:|---:|---:|---:|']
    for vx in rf.SCENES['G']['vxs']:
        z0, z1, z3 = [], [], []
        for s in rf.SCENES['G']['seeds']:
            a, c = load('G_vx%g_p_Cs_s%d' % (vx, s))
            phi, th, ps = a[:, c['gt_roll']], a[:, c['gt_pitch']], np.unwrap(a[:, c['gt_yaw']])
            rate = np.diff(ps) / DT
            est = df.interval_rate(phi, th, ps)
            eul = (est[:, 1] * np.sin(phi[:-1]) + est[:, 2] * np.cos(phi[:-1])) / np.cos(th[:-1])
            T = (len(rate) - 250) * DT
            end = lambda x: DEG * abs((x - rate)[250:].mean()) * T
            z0.append(end(a[1:, c['raw_yaw']])); z1.append(end(a[1:, c['raw_yaw_interval']])); z3.append(end(eul))
        L.append('| %g | %.2f° | %.3f° | %.3f° |' % (vx, np.mean(z0), np.mean(z1), np.mean(z3)))
    L += ['', '单轴区间积分只去掉采样项，欧拉（倾斜环）项仍在；三轴区间角速度经欧拉率公式换算才同时去掉两项。', '',
          '### 7.4 同种子的物理路径不逐位相同（D5）', '']
    for cid_n, cid_o in (('N_p_Cs_s70000', 'O_p_Cs_s70000'), ('N_i_S1_s70003', 'O_i_S1_s70003')):
        a, c = load(cid_n)
        b, _ = load(cid_o)
        d = np.abs(a[:1000, c['gt_yaw']] - b[:1000, c['gt_yaw']])
        nz = np.nonzero(d)[0]
        L.append('- %s 与 %s：前 20 s 输入完全相同，gt_yaw 在第 %s 步开始不同（差 %.0e rad），到第 1000 步最大差 %.1e rad。'
                 % (cid_n, cid_o, nz[0] if len(nz) else '—', d[nz[0]] if len(nz) else 0, d.max()))
    L += ['', '仿真在 GPU 上不是逐位确定的：同样的输入在第 2–3 步出现 1e-9 量级的差异，随后被步态动力学放大。'
          '传感器噪声（numpy 按种子生成）在各臂之间逐位相同，所以配对仍然有效；但物理路径只是近似相同，'
          '重跑同一条不会逐位复现轨迹。', '']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--partial', action='store_true')
    args = ap.parse_args()
    plan, rows, L, missing = analyze(args.partial)
    res = hypotheses(plan, rows, L)
    secondary(plan, rows, L, res)
    posthoc(plan, rows, L)
    suffix = '_interim' if missing else ''
    (rf.OUT / ('RESULTS%s.md' % suffix)).write_text('\n'.join(L) + '\n', encoding='utf-8')
    (rf.OUT / ('results%s.json' % suffix)).write_text(
        json.dumps(dict(rows=rows, tests=res), indent=1, default=str), encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
