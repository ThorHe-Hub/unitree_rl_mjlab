"""论文补充分析：正文里不由正式批次分析器（analyze_design_family_formal.py、analyze_pulse_formal.py）直接给出的数字，
都由本脚本生成。只读取数据，不改动任何批次。

    python analyze_thesis_supplement.py                        # 全部，约 20 分钟
    python analyze_thesis_supplement.py --only budget nominal  # 只重算指定各节，其余沿用已有结果

输出 unitree_heading_results/thesis_supplement_20260929/{supplement.json, SUPPLEMENT.md}。
标"事后"的各节是在正式结果出来之后才做的分析，不属于任何预注册。
"""
import argparse
import glob
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
FORMAL = PROJECT / 'unitree_heading_results/design_family_formal_20260929'
if os.name == 'nt':                  # 正式批次有超过 260 字符的轨迹路径（DEVIATIONS D1）
    os.environ.setdefault('DF_FORMAL_OUT', '\\\\?\\' + str(FORMAL))
import design_family as df                                            # noqa: E402
import run_design_family_formal as rf                                 # noqa: E402
from analyze_design_family import BMI088, evaluate, best              # noqa: E402
from analyze_design_family_sweeps import TAU_FINE, tau_window         # noqa: E402
from run_heading_fixed_input import digest                            # noqa: E402

OUT = PROJECT / 'unitree_heading_results/thesis_supplement_20260929'
OFFLINE = PROJECT / 'unitree_heading_results/design_family_offline_20260928'
RES = PROJECT / 'unitree_heading_results'
DEG = 180 / math.pi
DT = 0.02
SEEDS = range(70000, 70010)
DEV = {'bias_002': 'bias_002/[BC]_seed*/**/*.npz', 'duration_300s': 'duration_300s_20260908/**/*.npz',
       'outage30': 'reference_outage_20260912/outage30/**/*.npz'}
ARM = {'Bs': 'B\\*', 'Cs': 'C\\*', 'S1': 'S1', 'C8': 'C8'}
VAR = {'p': '点采样', 'i': '区间平均', 'point': '点采样', 'interval': '区间平均'}
_PLAN = {}


def formal(case_id):
    """正式批次一条已完成轨迹的路径（逐条核对完成标记与轨迹 SHA256）。"""
    if not _PLAN:
        _PLAN['plan'] = rf.load_plan()
        _PLAN['cases'] = {c['case_id']: c for c in _PLAN['plan']['cases']}
    state, path = rf.case_state(_PLAN['cases'][case_id], _PLAN['plan'])
    if state != 'done':
        raise RuntimeError('not complete: ' + case_id)
    return path


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return (z['data'].copy(), {str(n): i for i, n in enumerate(z['columns'])},
                {k: z[k].item() for k in z.files if k not in ('data', 'columns')})


def formal_rows():
    return json.loads((FORMAL / 'results.json').read_text(encoding='utf-8'))['rows']


def ms(x, d=3):
    x = np.asarray(x, dtype=float)
    return '%.*f ± %.*f' % (d, x.mean(), d, x.std(ddof=1))


def ftrace(case_id, **kw):
    return df.load_trace(formal(case_id), **kw)


# ---------------------------------------------------------------------------
# 1 开发集路径的记录条件
# ---------------------------------------------------------------------------
def sec_dev_conditions():
    L = ['## 1 开发集路径的记录条件', '',
         '正文的离线 BMI088 结果是在这些路径上的反事实重放：只换噪声项，运动学项与跳变下限仍反映记录时的闭环运动。', '',
         '| 路径集 | 条数 | 种子 | 步数 | 记录时 σg [rad/s] | 记录时方案 | 记录时 q_b | 真实航向最大单步变化 |',
         '|---|---:|---|---:|---:|---|---|---:|']
    out = {}
    for name, pat in DEV.items():
        paths = sorted(glob.glob(str(RES / pat), recursive=True))
        seeds, sg, lens, modes, qb, jumps = set(), set(), set(), {}, set(), []
        for p in paths:
            a, c, m = load(p)
            seeds.add(int(m['seed'])); sg.add(float(m['noise_std'])); lens.add(len(a))
            modes[m['feedback_mode']] = modes.get(m['feedback_mode'], 0) + 1
            qb.add(str(m.get('fusion_q_bias', '元数据未记录')))
            psi = np.unwrap(a[:, c['gt_yaw']])
            jumps.append(DEG * np.abs(np.diff(psi[250:])).max())
        out[name] = dict(n=len(paths), seeds=[min(seeds), max(seeds)], steps=sorted(lens), sigma_g=sorted(sg),
                         modes=modes, q_b=sorted(qb), yaw_jump_deg=float(np.mean(jumps)))
        L.append('| %s | %d | %d–%d | %s | %s | %s | %s | %.2f° |' % (
            name, len(paths), min(seeds), max(seeds), '/'.join(map(str, sorted(lens))), '/'.join(map(str, sorted(sg))),
            '、'.join('%s %d' % kv for kv in modes.items()), '、'.join(sorted(qb)), np.mean(jumps)))
    jf = []
    for s in SEEDS:
        a, c, _ = load(formal('N_p_Cs_s%d' % s))
        jf.append(DEG * np.abs(np.diff(np.unwrap(a[:, c['gt_yaw']])[250:])).max())
    out['formal_N'] = dict(yaw_jump_deg=float(np.mean(jf)))
    L += ['| 对照：正式批次 N（C\\*，点采样） | 10 | 70000–70009 | 3000 | 0.001728 | fusion_heading 10 | 1e-16 | %.2f° |'
          % np.mean(jf), '']
    return dict(data=out, md=L)


# ---------------------------------------------------------------------------
# 2 C* 参考噪声项：近似与精确递推（正文表 3-2）
# ---------------------------------------------------------------------------
def sec_approx_ref():
    sw = json.loads((OFFLINE / 'sweeps.json').read_text(encoding='utf-8'))['rate']
    L = ['## 2 C\\* 的参考噪声项：近似与精确递推（开发集路径，[30, 60) s）', '',
         '| f_ref [Hz] | σa [rad] | 精确 | √3·σa/√N | 2σa/√N |', '|---:|---:|---:|---:|---:|']
    out = []
    t = np.arange(1500, 3000) * DT
    for r in sorted(sw, key=lambda r: (r['hz'], r['sigma_a'])):
        row = next(x for x in r['rows'] if x['q_b'] == r['c_q'] and float(x['tau']) == 0.0)
        ex = DEG * math.sqrt(row['windows']['30-60']['var_ref'])
        N = np.floor(t * r['hz'])
        a3 = DEG * math.sqrt(np.mean(3 * r['sigma_a'] ** 2 / N))
        a2 = DEG * math.sqrt(np.mean(4 * r['sigma_a'] ** 2 / N))
        out.append(dict(hz=r['hz'], sigma_a=r['sigma_a'], exact=ex, sqrt3=a3, two=a2))
        L.append('| %g | %g | %.3f° | %.3f°（%+.0f%%） | %.3f°（%+.0f%%） |' % (
            r['hz'], r['sigma_a'], ex, a3, 100 * (a3 / ex - 1), a2, 100 * (a2 / ex - 1)))
    return dict(data=out, md=L + ['', '注：N = ⌊f_ref·t⌋，对窗口内各时刻取均方后开方。', ''])


# ---------------------------------------------------------------------------
# 3 协方差递推与蒙特卡洛（正文表 3-4）
# ---------------------------------------------------------------------------
def sec_monte_carlo():
    n, M = 3000, 2000
    k = np.arange(n)
    upd = (k % 50 == 0) & (k > 0)
    rng = np.random.default_rng(1)
    sl = slice(1500, 3000)
    L = ['## 3 噪声项 RMS：协方差递推与蒙特卡洛（2000 次，1 Hz，[30, 60) s）', '',
         '| q_b | τ | 噪声项 | 递推 | 蒙特卡洛 | 相对差 |', '|---:|---:|---|---:|---:|---:|']
    out = []
    for q, taus in ((1e-16, [0.0, 1.0, math.inf]), (1e-4, [math.inf])):
        K0, K1 = df.gains(upd, BMI088, q, 0.05)
        var = df.noise_variance(upd, BMI088, q, 0.05, taus)
        for src in ('gyro', 'ref'):
            u = BMI088 * rng.standard_normal((M, n)) if src == 'gyro' else np.zeros((M, n))
            y = 0.05 * rng.standard_normal((M, n)) if src == 'ref' else np.zeros((M, n))
            f, _, i = df.replay(u, y, upd, K0, K1)
            for j, tau in enumerate(taus):
                s = df.smooth(f, i, tau)
                mc = DEG * math.sqrt((s[:, sl] ** 2).mean())
                th = DEG * math.sqrt(var[src][j, sl].mean())
                out.append(dict(q_b=q, tau=tau, src=src, theory=th, mc=mc))
                L.append('| %g | %s | %s | %.4f° | %.4f° | %+.1f%% |' % (
                    q, '∞' if math.isinf(tau) else '%g' % tau, {'gyro': '陀螺', 'ref': '参考'}[src], th, mc,
                    100 * (mc / th - 1)))
    return dict(data=out, md=L + [''])


# ---------------------------------------------------------------------------
# 4 误差预算（正文表 4-2）
# ---------------------------------------------------------------------------
def sec_budget():
    L = ['## 4 误差预算（正式批次 N，10 个种子，[30, 60) s）', '',
         '| 陀螺信号 | 方案 | 期望 RMS J | 闭环实测（合并） | 运动学 | 零偏 | 陀螺噪声 | 参考噪声 | 估计误差最大单步变化 |',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    out = {}
    for v in ('p', 'i'):
        for arm, q, tau in (('Bs', 1e-4, math.inf), ('Cs', 1e-16, 0.0), ('S1', 1e-16, 1.0)):
            tr = [ftrace('N_%s_%s_s%d' % (v, arm, s)) for s in SEEDS]
            r = evaluate(tr, BMI088, 0.05, [q], [tau], [(30, 60)])[0]
            w = r['windows']['30-60']
            d = dict(J=w['expected_rms'], emp=w['empirical_rms'], kin=DEG * math.sqrt(w['mse_kin']),
                     bias=DEG * math.sqrt(w['mse_bias']), gyro=DEG * math.sqrt(w['var_gyro']),
                     ref=DEG * math.sqrt(w['var_ref']), jump=r['jump_deg'])
            if arm == 'Cs':     # 同一增益下去掉陀螺噪声项
                d['J_no_gyro'] = math.sqrt(d['J'] ** 2 - d['gyro'] ** 2)
            out['%s_%s' % (v, arm)] = d
            L.append('| %s | %s | %.3f° | %.3f° | %.3f° | %.3f° | %.3f° | %.3f° | %.3f° |' % (
                VAR[v], ARM[arm], d['J'], d['emp'], d['kin'], d['bias'], d['gyro'], d['ref'], d['jump']))
    L += ['', '点采样 C\\* 在同一增益下去掉陀螺噪声项：%.3f° → %.3f°。' % (out['p_Cs']['J'], out['p_Cs']['J_no_gyro']), '']
    return dict(data=out, md=L)


# ---------------------------------------------------------------------------
# 5 名义工况：平滑指标的两个窗口、物理航向（事后）；参考率场景的稳态跳变
# ---------------------------------------------------------------------------
def _jumps(a, c):
    dfb = DEG * np.abs(np.diff(a[:, c['feedback_heading']]))
    dcm = np.abs(np.diff(a[:, c['cmd_vyaw']]))
    return dict(fb5=dfb[250:].max(), fb30=dfb[1500:3000].max(), cmd5=dcm[250:].max(), cmd30=dcm[1500:3000].max(),
                t_fb=(250 + int(dfb[250:].argmax())) * DT)


def sec_nominal():
    L = ['## 5 名义工况的平滑指标与物理航向（正式批次 N，10 个种子；事后）', '',
         '预注册的跳变指标取 [5 s, 末尾)，其最大值多出现在滤波器收敛初期；这里同时给出与精度指标相同的 [30, 60) s。', '',
         '| 陀螺信号 | 方案 | 反馈最大单步变化 [5 s, 末尾) | 同 [30, 60) s | 指令最大单步变化 [5 s, 末尾) [rad/s] | '
         '同 [30, 60) s | [5 s, 末尾) 最大值所在时刻 [s] | S 均值 [°] | 平均转向指令 [rad/s] | 物理航向 RMS [°] | 估计误差 RMS [°] |',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    out = {}
    for v in ('p', 'i'):
        for arm in ('Bs', 'Cs', 'S1'):
            rs = []
            for s in SEEDS:
                a, c, _ = load(formal('N_%s_%s_s%d' % (v, arm, s)))
                psi = np.unwrap(a[:, c['gt_yaw']]); psi -= psi[0]
                fb = a[:, c['feedback_heading']]
                w = slice(1500, 3000)
                r = _jumps(a, c)
                r.update(S=DEG * fb[w].mean(), cmd=a[w, c['cmd_vyaw']].mean(), phys=DEG * math.sqrt((psi[w] ** 2).mean()),
                         est=DEG * math.sqrt(((fb - psi)[w] ** 2).mean()))
                rs.append(r)
            agg = {k: [float(r[k]) for r in rs] for k in rs[0]}
            out['%s_%s' % (v, arm)] = agg
            L.append('| %s | %s | %s° | %s° | %s | %s | %s | %s | %s | %s | %s |' % (
                VAR[v], ARM[arm], ms(agg['fb5']), ms(agg['fb30']), ms(agg['cmd5']), ms(agg['cmd30']),
                ms(agg['t_fb'], 1), ms(agg['S']), ms(agg['cmd'], 4), ms(agg['phys']), ms(agg['est'])))
    L += ['', '### 参考率场景 R（C\\*，点采样，种子 70000–70004）', '',
          '| f_ref [Hz] | 反馈最大单步变化 [5 s, 末尾) | 所在时刻 [s] | 同 [30, 60) s |', '|---:|---:|---:|---:|']
    out['R'] = {}
    for hz, ids in ((0.5, ['R_f0.5_p_Cs_s%d' % s for s in range(70000, 70005)]),
                    (1.0, ['N_p_Cs_s%d' % s for s in range(70000, 70005)]),
                    (2.0, ['R_f2_p_Cs_s%d' % s for s in range(70000, 70005)])):
        rs = [_jumps(*load(formal(i))[:2]) for i in ids]
        agg = {k: [float(r[k]) for r in rs] for k in rs[0]}
        out['R'][str(hz)] = agg
        L.append('| %g | %s° | %s | %s° |' % (hz, ms(agg['fb5']), ms(agg['t_fb'], 1), ms(agg['fb30'])))
    return dict(data=out, md=L + [''])


# ---------------------------------------------------------------------------
# 6 τ 的精度上界与平滑下界：开发集路径与正式路径（事后；正文表 3-3、图 3-2）
# ---------------------------------------------------------------------------
TAU_Q = [1e-16, 1e-12, 1e-10, 1e-8]
TAU_COMBOS = [(hz, sa) for hz in (0.25, 0.5, 1.0, 2.0, 5.0) for sa in (0.02, 0.05, 0.1)]
STRESS_COMBOS = [(0.5, 0.05), (1.0, 0.05), (2.0, 0.05)]


def _tw(rows):
    c = best(rows, '30-60', tau=0.0)
    w = tau_window(rows, c['q_b'], '30-60', c['windows']['30-60']['expected_rms'])
    prof = sorted((r for r in rows if r['q_b'] == c['q_b']), key=lambda r: float(r['tau']))
    return dict(c_star=c['windows']['30-60']['expected_rms'], c_q=c['q_b'], acc_max=w['tau_acc_max'],
                jump_min=w['tau_jump_min'], floor=w['jump_floor'], window=w['window'],
                taus=[float(r['tau']) for r in prof], jump=[r['jump_deg'] for r in prof],
                rms_ratio=[r['windows']['30-60']['expected_rms'] / c['windows']['30-60']['expected_rms'] for r in prof])


def sec_tau_window():
    sw = json.loads((OFFLINE / 'sweeps.json').read_text(encoding='utf-8'))['rate']
    out = {'dev_bmi': {}, 'formal_p_bmi': {}, 'formal_i_bmi': {}, 'formal_p_stress': {}, 'dev_stress': {}}
    for r in sw:
        prof = sorted((x for x in r['rows'] if x['q_b'] == r['c_q']), key=lambda x: float(x['tau']))
        out['dev_bmi']['%g_%g' % (r['hz'], r['sigma_a'])] = dict(
            c_star=r['c_star'], c_q=r['c_q'], acc_max=r['tau_acc_max'], jump_min=r['tau_jump_min'],
            floor=r['jump_floor'], window=r['window'], taus=[float(x['tau']) for x in prof],
            jump=[x['jump_deg'] for x in prof],
            rms_ratio=[x['windows']['30-60']['expected_rms'] / r['c_star'] for x in prof])
    for v in ('p', 'i'):
        paths = [formal('N_%s_Cs_s%d' % (v, s)) for s in SEEDS]
        for hz, sa in TAU_COMBOS:
            tr = [df.load_trace(p, abs_hz=hz) for p in paths]
            out['formal_%s_bmi' % v]['%g_%g' % (hz, sa)] = _tw(evaluate(tr, BMI088, sa, TAU_Q, TAU_FINE, [(30, 60)]))
            print('   τ 窗口 formal %s f=%g σa=%g' % (v, hz, sa), flush=True)
    paths = [formal('N_p_Cs_s%d' % s) for s in SEEDS]
    dev = sorted(glob.glob(str(RES / DEV['bias_002']), recursive=True))
    for hz, sa in STRESS_COMBOS:
        out['formal_p_stress']['%g_%g' % (hz, sa)] = _tw(evaluate(
            [df.load_trace(p, abs_hz=hz) for p in paths], 0.15, sa, TAU_Q, TAU_FINE, [(30, 60)]))
        out['dev_stress']['%g_%g' % (hz, sa)] = _tw(evaluate(
            [df.load_trace(p, abs_hz=hz) for p in dev], 0.15, sa, TAU_Q, TAU_FINE, [(30, 60)]))
    g = lambda x: '—' if x is None else '%g' % x
    L = ['## 6 τ 的精度上界与平滑下界（事后，[30, 60) s）', '',
         '精度上界：期望 RMS ≤ 1.05 × C\\* 的最大 τ。平滑下界：估计误差最大单步变化 ≤ 1.10 × 其最小值（跳变下限）的最小 τ。'
         '跳变按 [5 s, 末尾) 计。', '',
         '| 陀螺档 | f_ref [Hz] | σa [rad] | 精度上界：开发集 / 正式点采样 / 正式区间 [s] | 跳变下限：开发集 / 正式点采样 / 正式区间 | '
         '平滑下界：开发集 / 正式点采样 / 正式区间 [s] |', '|---|---:|---:|---|---|---|']
    for hz, sa in TAU_COMBOS:
        key = '%g_%g' % (hz, sa)
        d, p, i = out['dev_bmi'][key], out['formal_p_bmi'][key], out['formal_i_bmi'][key]
        L.append('| BMI088 | %g | %g | %s / %s / %s | %.3f° / %.3f° / %.3f° | %s / %s / %s |' % (
            hz, sa, g(d['acc_max']), g(p['acc_max']), g(i['acc_max']), d['floor'], p['floor'], i['floor'],
            g(d['jump_min']), g(p['jump_min']), g(i['jump_min'])))
    for hz, sa in STRESS_COMBOS:
        key = '%g_%g' % (hz, sa)
        d, p = out['dev_stress'][key], out['formal_p_stress'][key]
        L.append('| 压力档 | %g | %g | %s / %s / — | %.3f° / %.3f° / — | %s / %s / — |' % (
            hz, sa, g(d['acc_max']), g(p['acc_max']), d['floor'], p['floor'], g(d['jump_min']), g(p['jump_min'])))
    return dict(data=out, md=L + [''])


# ---------------------------------------------------------------------------
# 7 中断与恢复（正文 5.2、5.3、表 5-2 注）
# ---------------------------------------------------------------------------
def _mask(n, t0=None, dur=30.0):
    k = np.arange(n)
    upd = (k % 50 == 0) & (k > 0)
    if t0 is not None:
        upd &= ~((k * DT >= t0 - 1e-9) & (k * DT < t0 + dur - 1e-9))
    return upd


def _bias_sd(q, k_end=1000):
    """按实际增益传播陀螺与参考噪声，得到第 k_end 步零偏估计误差的真实 SD（不是滤波器自认的 P）。"""
    upd = _mask(k_end)
    K0, K1 = df.gains(upd, BMI088, q, 0.05)
    F = np.array([[1.0, -DT], [0.0, 1.0]])
    S = np.zeros((2, 2))
    for j in range(k_end):
        S = F @ S @ F.T + np.diag([(BMI088 * DT) ** 2, 0.0])
        if upd[j]:
            K = np.array([K0[j], K1[j]])
            IKH = np.array([[1 - K[0], 0.0], [-K[1], 1.0]])
            S = IKH @ S @ IKH.T + np.outer(K, K) * 0.05 ** 2
    kk = np.nonzero(upd)[0]
    return math.sqrt(S[1, 1]), float(K0[kk[-1]]), float(K1[kk[-1]])


def _reacq(tr, t1, tau):
    parts = df.decompose(tr, BMI088, 1e-16, 0.05)
    tot = sum(df.smooth(parts[k][0], parts[k][1], tau) for k in df.SOURCES)
    k1 = int(round(t1 / DT))
    return DEG * np.abs(np.diff(tot, axis=1))[:, k1 - 1:k1 + 100].max(axis=1)


def sec_recovery():
    out = {}
    L = ['## 7 中断与恢复', '', '### 中断开始时零偏估计误差的真实 SD（1 Hz，BMI088，第 1000 步即 20 s）', '',
         '| q_b | SD [rad/s] | ×30 s [°] | 最后一次更新的 K0 | K1 [1/s] |', '|---:|---:|---:|---:|---:|']
    for q in (1e-4, 1e-16):
        sd, k0, k1 = _bias_sd(q)
        out['bias_sd_%g' % q] = dict(sd=sd, drift30=DEG * sd * 30, K0=k0, K1=k1)
        L.append('| %g | %.4f | %.1f | %.3f | %.4f |' % (q, sd, DEG * sd * 30, k0, k1))
    L += ['', '### 闭环漂移率（正式批次 O，点采样，(e(50 s) − e(20 s))/30，10 个种子的 RMS）', '']
    for arm in ('Bs', 'Cs'):
        rates = []
        for s in SEEDS:
            a, c, _ = load(formal('O_p_%s_s%d' % (arm, s)))
            psi = np.unwrap(a[:, c['gt_yaw']]); psi -= psi[0]
            e = DEG * (a[:, c['feedback_heading']] - psi)
            rates.append((e[2499] - e[1000]) / 30.0)
        rates = np.array(rates)
        out['drift_%s' % arm] = dict(rms=float(math.sqrt((rates ** 2).mean())), per_seed=rates.tolist())
        L.append('- %s：%.3f °/s（%.4f rad/s）' % (ARM[arm], math.sqrt((rates ** 2).mean()),
                                                math.sqrt((rates ** 2).mean()) / DEG))
    L += ['', '### 恢复增益与中断开始的时刻（q_b = 1e-16）', '',
          '| 中断 | 中断前最后一次 K0 | 恢复后第一次 K0 |', '|---|---:|---:|']
    for t0, n in ((20.0, 4500), (60.0, 15000)):
        upd = _mask(n, t0)
        K0, _ = df.gains(upd, BMI088, 1e-16, 0.05)
        kk = np.nonzero(upd)[0]
        b, a_ = kk[kk < int(t0 / DT)][-1], kk[kk >= int((t0 + 30) / DT)][0]
        out['K0_%g' % t0] = dict(before=float(K0[b]), after=float(K0[a_]))
        L.append('| [%g, %g) s | %.3f | %.3f |' % (t0, t0 + 30, K0[b], K0[a_]))
    L += ['', '### 恢复跳变（估计误差的最大单步变化，[恢复前 1 步, 恢复后 2 s)）', '',
          '| 路径 | 中断 | τ = 0 | τ = 1 s |', '|---|---|---:|---:|']
    tr = [ftrace('O_p_Cs_s%d' % s) for s in SEEDS]
    e0, e1 = _reacq(tr, 50.0, 0.0), _reacq(tr, 50.0, 1.0)
    fbj = [r['reacq_jump'] for r in formal_rows() if r['scene'] == 'O' and r['variant'] == 'point' and r['arm'] == 'Cs']
    out['formal_O'] = dict(tau0=e0.tolist(), tau1=e1.tolist(), fb_recorded=fbj)
    L.append('| 正式批次 O（C\\* 路径，点采样） | [20, 50) s | %s° | %s° |' % (ms(e0, 2), ms(e1, 2)))
    L.append('| 同上，闭环记录的反馈量口径 | [20, 50) s | %s° | — |' % ms(fbj, 2))
    d300 = sorted(glob.glob(str(RES / DEV['duration_300s']), recursive=True))
    for t0 in (20.0, 60.0):
        tr = [df.load_trace(p, outage=(t0, 30.0)) for p in d300]
        e0, e1 = _reacq(tr, t0 + 30, 0.0), _reacq(tr, t0 + 30, 1.0)
        out['dev300_%g' % t0] = dict(tau0=e0.tolist(), tau1=e1.tolist())
        L.append('| 开发集 300 s 路径（反事实中断） | [%g, %g) s | %s° | %s° |' % (t0, t0 + 30, ms(e0, 2), ms(e1, 2)))
    return dict(data=out, md=L + [''])


# ---------------------------------------------------------------------------
# 8 零偏先验与静止标定（事后；正文 6.3）
# ---------------------------------------------------------------------------
PRIOR_SETTINGS = (('初始零偏未知', 0.05, None), ('标定残差 1e-3', 1e-3, 1e-3), ('标定残差 2e-4', 2e-4, 2e-4),
                  ('理想标定', 2e-4, 0.0))


def _prior_scan(tr, win, qs):
    """返回 {(prior, q, tau): (det 的三个均方分量, 噪声方差)}；det = kin + β·bias，β 由标定残差决定。"""
    sl = slice(int(win[0] / DT), int(win[1] / DT))
    res = {}
    for pr in sorted({p for _, p, _ in PRIOR_SETTINGS}):
        for q in qs:
            parts = df.decompose(tr, BMI088, q, 0.05, prior=pr)
            var = df.noise_variance(tr[0].upd, BMI088, q, 0.05, [0.0, math.inf], prior=pr)
            for j, tau in enumerate((0.0, math.inf)):
                s = {k: df.smooth(parts[k][0], parts[k][1], tau) for k in df.SOURCES}
                kin, bias = s['kin'][:, sl], s['bias'][:, sl]
                res[(pr, q, tau)] = (float((kin ** 2).mean()), float((kin * bias).mean()), float((bias ** 2).mean()),
                                     float(var['gyro'][j, sl].mean() + var['ref'][j, sl].mean()))
    return res


def _J(m, b0, resid):
    kk, kb, bb, nv = m
    if resid is None:                 # 零偏未知：记录的 b0 全部留给估计器
        det = kk + 2 * kb + bb
    else:                             # 残差 ~ N(0, resid²)：交叉项期望为 0
        det = kk + bb * (resid / b0) ** 2
    return DEG * math.sqrt(det + nv)


def sec_bias_prior():
    L = ['## 8 零偏先验与静止标定（事后）', '',
         '"标定残差 r"：开机前静止标定，零偏先验 SD 取 r，初始零偏估计误差按 N(0, r²) 计；"理想标定"：残差为 0，先验 SD 取 2e-4。'
         '标定只能消除常值零偏 b0，消除不了步态造成的等效零偏。', '',
         '| 路径与窗口 | 设定 | B 族最优（q_b） | C 族最优（q_b） |', '|---|---|---|---|']
    out = {}
    cases = (('正式 N 点采样 [30, 60) s', [ftrace('N_p_Cs_s%d' % s) for s in SEEDS], (30, 60),
              [1e-16, 1e-10, 1e-8, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4]),
             ('正式 N 区间平均 [30, 60) s', [ftrace('N_i_Cs_s%d' % s) for s in SEEDS], (30, 60),
              [1e-16, 1e-10, 1e-8, 1e-6, 1e-5, 3e-5, 1e-4, 3e-4]),
             ('开发集 300 s 点采样 [240, 300) s',
              [df.load_trace(p) for p in sorted(glob.glob(str(RES / DEV['duration_300s']), recursive=True))], (240, 300),
              [1e-16, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4]))
    for name, tr, win, qs in cases:
        b0 = float(tr[0].meta['gyro_bias'])
        m = _prior_scan(tr, win, qs)
        out[name] = {}
        for lab, pr, resid in PRIOR_SETTINGS:
            bb = min(((q, _J(m[(pr, q, math.inf)], b0, resid)) for q in qs), key=lambda x: x[1])
            cc = min(((q, _J(m[(pr, q, 0.0)], b0, resid)) for q in qs), key=lambda x: x[1])
            out[name][lab] = dict(B=bb[1], B_q=bb[0], C=cc[1], C_q=cc[0])
            L.append('| %s | %s | %.3f°（%g） | %.3f°（%g） |' % (name, lab, bb[1], bb[0], cc[1], cc[0]))
        print('   零偏先验：%s' % name, flush=True)
    L += ['', '### 参考中断后 [60, 90) s（正式批次 O 的 C\\* 路径；各设定取上表正式 N 的族内最优 q_b）', '',
          '| 陀螺信号 | 设定 | B 族 | C 族 |', '|---|---|---:|---:|']
    for v, nkey in (('p', '正式 N 点采样 [30, 60) s'), ('i', '正式 N 区间平均 [30, 60) s')):
        tr = [ftrace('O_%s_Cs_s%d' % (v, s)) for s in SEEDS]
        b0 = float(tr[0].meta['gyro_bias'])
        out['O_' + v] = {}
        m = _prior_scan(tr, (60, 90), sorted({out[nkey][lab][k] for lab, _, _ in PRIOR_SETTINGS for k in ('B_q', 'C_q')}))
        for lab, pr, resid in PRIOR_SETTINGS:
            sel = out[nkey][lab]
            jb, jc = _J(m[(pr, sel['B_q'], math.inf)], b0, resid), _J(m[(pr, sel['C_q'], 0.0)], b0, resid)
            out['O_' + v][lab] = dict(B=jb, C=jc)
            L.append('| %s | %s | %.2f° | %.2f° |' % (VAR[v], lab, jb, jc))
    return dict(data=out, md=L + [''])


# ---------------------------------------------------------------------------
# 9 非参数稳健性（事后）；10 区间平均对策略观测的影响（事后）
# ---------------------------------------------------------------------------
H_ROWS = (('H1', 'N', 'point', 'Bs', 'Cs', 'err'), ('H2', 'N', 'interval', 'Bs', 'Cs', 'err'),
          ('H3', 'N', 'point', 'Cs', 'S1', 'fb_jump'), ('H4', 'N', 'interval', 'Cs', 'S1', 'fb_jump'),
          ('H5', 'O', 'point', 'Bs', 'Cs', 'err_late'), ('H6', 'O', 'interval', 'Bs', 'Cs', 'err_late'),
          ('H7', 'O', 'point', 'Cs', 'S1', 'reacq_jump'), ('H8', 'O', 'interval', 'Cs', 'S1', 'reacq_jump'))


def sec_nonparam():
    from scipy import stats
    rows = formal_rows()
    res = []
    for hid, sc, v, a, b, key in H_ROWS:
        g = lambda arm: {r['seed']: r[key] for r in rows if r['scene'] == sc and r['variant'] == v and r['arm'] == arm}
        A, B = g(a), g(b)
        d = np.array([B[s] - A[s] for s in sorted(A)])
        res.append(dict(id=hid, n=len(d), neg=int((d < 0).sum()), p=float(stats.wilcoxon(d).pvalue)))
    order = sorted(range(len(res)), key=lambda k: res[k]['p'])
    run = 0.0
    for rank, k in enumerate(order):
        run = max(run, min(1.0, res[k]['p'] * (len(res) - rank)))
        res[k]['p_holm'] = run
    L = ['## 9 主要假设的非参数检验（事后）', '', '| 编号 | n | d < 0 的种子 | Wilcoxon 精确 p | Holm p |',
         '|---|---:|---:|---:|---:|']
    L += ['| %s | %d | %d/%d | %.4f | %.4f |' % (r['id'], r['n'], r['neg'], r['n'], r['p'], r['p_holm']) for r in res]
    L += ['', 'n = 5 时符号检验与 Wilcoxon 精确检验能达到的最小单侧 p 为 1/32 = 0.031，经 3 项 Holm 校正后为 0.094，'
          '所以指令阶跃批次的 PH1/PH3 只能依靠 t 检验。', '']
    return dict(data=res, md=L)


def sec_variant_obs():
    d = {k: [] for k in ('bias', 'S', 'cmd')}
    for s in SEEDS:
        (ap, cp, _), (ai, ci, _) = load(formal('N_p_Cs_s%d' % s)), load(formal('N_i_Cs_s%d' % s))
        w = slice(1500, 3000)
        d['bias'].append(ai[w, ci['fusion_bias']].mean() - ap[w, cp['fusion_bias']].mean())
        d['S'].append(DEG * (ai[w, ci['feedback_heading']].mean() - ap[w, cp['feedback_heading']].mean()))
        d['cmd'].append(ai[w, ci['cmd_vyaw']].mean() - ap[w, cp['cmd_vyaw']].mean())
    L = ['## 10 区间平均陀螺对策略观测的影响（正式批次 N，C\\*，同种子 interval − point，[30, 60) s；事后）', '',
         '- 零偏估计之差：%s rad/s。策略观测 = 点采样读数 − 零偏估计，所以观测到的角速度相应偏移 %.4f rad/s。'
         % (ms(d['bias'], 5), -np.mean(d['bias'])),
         '- 反馈量 S 的均值之差：%s°；平均转向指令之差：%s rad/s。' % (ms(d['S']), ms(d['cmd'], 4)), '']
    return dict(data={k: [float(x) for x in v] for k, v in d.items()}, md=L)


def sec_seed_overlap():
    """正式批次的种子是否在其他结果目录中出现过（只读 npz 里的 seed 字段）。"""
    formal_dirs = ('design_family_formal_20260929', 'pulse_formal_20260929', 'thesis_supplement_20260929')
    used, failed = {}, set()
    for p in glob.glob(str(RES / '**' / '*.npz'), recursive=True):
        top = Path(p).relative_to(RES).parts[0]
        if top in formal_dirs:
            continue
        try:
            with np.load(p, allow_pickle=False) as z:
                if 'seed' in z.files:
                    used.setdefault(int(z['seed'].item()), set()).add(top)
        except (OSError, ValueError):
            failed.add(top)
    fs = set(range(70000, 70010)) | set(range(71000, 71003)) | set(range(72000, 72005))
    hit = sorted(s for s in fs if s in used)
    L = ['## 11 种子隔离', '',
         '- 其他结果目录共用过 %d 个不同种子；与正式批次的 %d 个种子重叠：%s。' % (len(used), len(fs), hit or '无'),
         '- 读取失败的目录：%s（均早于正式批次）。' % ('、'.join(sorted(failed)) or '无'), '']
    return dict(data=dict(n_other=len(used), n_formal=len(fs), overlap=hit, failed=sorted(failed)), md=L)


SECTIONS = dict(dev_conditions=sec_dev_conditions, approx_ref=sec_approx_ref, monte_carlo=sec_monte_carlo,
                budget=sec_budget, nominal=sec_nominal, tau_window=sec_tau_window, recovery=sec_recovery,
                bias_prior=sec_bias_prior, nonparam=sec_nonparam, variant_obs=sec_variant_obs,
                seed_overlap=sec_seed_overlap)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', nargs='*', choices=sorted(SECTIONS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'supplement.json'
    res = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    for name in (args.only or list(SECTIONS)):
        t0 = time.time()
        res[name] = SECTIONS[name]()
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
        print('[%s] %.0f s' % (name, time.time() - t0), flush=True)
    L = ['# 论文补充分析', '', '由 analyze_thesis_supplement.py 生成（SHA256 %s）。标"事后"的各节不属于任何预注册。'
         % digest(Path(__file__)), '']
    for name in SECTIONS:
        if name in res:
            L += res[name]['md']
    (OUT / 'SUPPLEMENT.md').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('wrote', OUT / 'SUPPLEMENT.md')


if __name__ == '__main__':
    main()
