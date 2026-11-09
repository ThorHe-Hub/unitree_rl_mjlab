"""v3 论文图（机理示意 + 实验结果）。输出 figs/v3/<名>.pdf 与 .png（300 dpi）。

    python make_figures_v3.py            # 全部
    python make_figures_v3.py f01 f08    # 指定

图宽按 A4 版心（约 16 cm）设计，正文字号 9 pt，放进论文时不缩放。图内不写标题，标题由论文图注给出。
数据来源写在每个函数的文档字符串里。f09、f13 读取 analyze_thesis_supplement.py 的输出，先运行它。
"""
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
OUT = PROJECT / 'figs/v3'
FORMAL = PROJECT / 'unitree_heading_results/design_family_formal_20260929'
OFFLINE = PROJECT / 'unitree_heading_results/design_family_offline_20260928'
PULSE = PROJECT / 'unitree_heading_results/pulse_formal_20260929'
SUPP = PROJECT / 'unitree_heading_results/thesis_supplement_20260929'
if os.name == 'nt':                               # 部分正式轨迹路径达 260 字符（DEVIATIONS D1）
    os.environ.setdefault('DF_FORMAL_OUT', '\\\\?\\' + str(FORMAL))

W = 6.3                                            # 图宽 [in]，约 16 cm
DEG = 180 / math.pi
DT = 0.02
# Okabe-Ito 色盲友好配色；方案固定配色，全文一致
COL = dict(B='#D55E00', C='#0072B2', S='#009E73', C8='#CC79A7', truth='#333333', ref='#E69F00',
           gray='#8c8c8c', light='#f2f2f2', ink='#1a1a1a', ink2='#555555')
BOX = dict(sensor='#e8f1f8', est='#eaf5ef', ctrl='#fdf1e4', plant='#f3f3f3', note='#fbfbfb')

for _fp in ('/mnt/c/Windows/Fonts/msyh.ttc', '/mnt/c/Windows/Fonts/simhei.ttf'):
    if Path(_fp).exists():
        matplotlib.font_manager.fontManager.addfont(_fp)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'DejaVu Sans'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'savefig.dpi': 300, 'figure.dpi': 110,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight', pad_inches=0.03)
    fig.savefig(OUT / (name + '.png'), bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print('wrote', OUT / (name + '.png'))


def panel(ax, label, x=-0.12, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9.5, fontweight='bold', va='bottom', ha='left')


# ---------------------------------------------------------------------------
# 示意图工具
# ---------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fc=BOX['note'], ec='#7a8a99', fs=8.5, lw=0.9, bold=False, color=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', color=color or COL['ink'], linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, p0, p1, color='#4a5a6a', text=None, tpos=0.5, toff=(0, 0.12), lw=1.0, ls='-', fs=8, tcolor=None,
          rad=0.0, head=6):
    ax.annotate('', xy=p1, xytext=p0, arrowprops=dict(arrowstyle='-|>,head_length=0.35,head_width=0.18',
                color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0, connectionstyle='arc3,rad=%g' % rad,
                mutation_scale=head * 1.6))
    if text:
        tx = p0[0] + (p1[0] - p0[0]) * tpos + toff[0]
        ty = p0[1] + (p1[1] - p0[1]) * tpos + toff[1]
        ax.text(tx, ty, text, ha='center', va='center', fontsize=fs, color=tcolor or color)


def blank(figsize):
    """示意图画布：坐标轴铺满整张图，数据单位与版面尺寸一一对应，方框宽度才可控。"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    return fig, ax


# ---------------------------------------------------------------------------
# F01 研究路线（第 1 章）
# ---------------------------------------------------------------------------
def f01_route():
    """研究路线示意。无数据。"""
    fig, ax = blank((W, 2.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.5)
    steps = [
        ('问题', '固定策略下\n航向反馈的\n形成方式', BOX['plant']),
        ('建模', '统一设计族\n' + r'$S(q_b,\tau)$', BOX['est']),
        ('理论与离线', '误差按来源分解\n协方差递推\n开发集离线扫描', BOX['sensor']),
        ('闭环验证', '预注册\n正式批次\n208 + 35 条', BOX['ctrl']),
        ('结论', '精度、平滑\n与鲁棒的折中\n设计规则', BOX['plant']),
    ]
    w, h, gap = 1.62, 1.6, 0.4
    x0 = (10 - (5 * w + 4 * gap)) / 2
    for k, (head, body, fc) in enumerate(steps):
        x = x0 + k * (w + gap)
        box(ax, x, 1.15, w, h, body, fc=fc, fs=8)
        ax.text(x + w / 2, 1.15 + h + 0.14, head, ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=COL['ink'])
        if k < 4:
            arrow(ax, (x + w + 0.05, 1.15 + h / 2), (x + w + gap - 0.05, 1.15 + h / 2))
    xa = x0 + 2 * (w + gap) + w / 2
    xb = x0 + 3 * (w + gap) + w / 2
    arrow(ax, (xb, 1.1), (xa, 1.1), rad=-0.35, color=COL['gray'], ls='--', lw=0.9)
    ax.text((xa + xb) / 2, 0.28, r'闭环轨迹检验离线模型（重放误差 < $10^{-12}$ rad）', ha='center', fontsize=7.5,
            color=COL['ink2'])
    save(fig, 'f01_route')


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------
_PLAN = {}


def formal(case_id):
    """正式批次的一条轨迹：(data, 列名->下标, meta, 路径)。"""
    import run_design_family_formal as rf
    if not _PLAN:
        _PLAN['plan'] = rf.load_plan()
        _PLAN['cases'] = {c['case_id']: c for c in _PLAN['plan']['cases']}
    state, path = rf.case_state(_PLAN['cases'][case_id], _PLAN['plan'])
    if state != 'done':
        raise RuntimeError('not complete: ' + case_id)
    with np.load(path, allow_pickle=False) as z:
        return (z['data'].copy(), {str(n): i for i, n in enumerate(z['columns'])},
                {k: z[k].item() for k in z.files if k not in ('data', 'columns')}, path)


def formal_rows():
    return json.loads((FORMAL / 'results.json').read_text(encoding='utf-8'))['rows']


def heading_err(a, c):
    psi = np.unwrap(a[:, c['gt_yaw']])
    return a[:, c['feedback_heading']] - (psi - psi[0])


def phase_of(a, c):
    return np.mod(np.arctan2(a[:, c['phase_sin']], a[:, c['phase_cos']]), 2 * np.pi)


# ---------------------------------------------------------------------------
# F02 系统结构与设计族（第 2、3 章）
# ---------------------------------------------------------------------------
def f02_system():
    r"""系统框图：同一估计器，只改航向反馈的接法；S(τ) 把 B、C 统一成一族。无数据。"""
    fig, ax = blank((W, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.08)
    box(ax, 0.55, 3.55, 1.75, 1.05, '陀螺（50 Hz）\n' + r'$z=\omega+\varepsilon+w$', fc=BOX['sensor'])
    box(ax, 0.55, 1.95, 1.75, 1.05, '绝对航向参考\n（1 Hz）' + r'$y=\psi+v$', fc=BOX['sensor'])
    box(ax, 2.75, 2.4, 1.7, 1.75, '两状态\n卡尔曼滤波\n' + r'状态 $[\psi,\ b]$' + '\n' + r'参数 $q_b$', fc=BOX['est'])
    box(ax, 5.0, 3.75, 2.05, 0.95, '扣零偏积分\n' + r'$i=\int(z-\hat b)\,dt$', fc=BOX['est'])
    box(ax, 5.0, 1.95, 2.05, 1.15, '反馈量（设计族）\n' + r'$S=i+\mathrm{LPF}_{\tau}(f-i)$', fc='#dff0e6',
        ec=COL['S'], lw=1.5)
    box(ax, 7.55, 1.95, 1.0, 1.15, 'PD\n航向保持', fc=BOX['ctrl'])
    box(ax, 7.55, 3.75, 1.0, 0.95, '固定 RL\n运动策略', fc=BOX['ctrl'])
    box(ax, 8.95, 2.85, 0.95, 1.2, 'G1\n物理\n仿真', fc=BOX['plant'])
    arrow(ax, (2.3, 4.45), (5.0, 4.45), text=r'$z$', toff=(0, 0.13))
    arrow(ax, (2.3, 3.8), (2.75, 3.6))
    arrow(ax, (2.3, 2.6), (2.75, 2.85))
    arrow(ax, (4.45, 3.95), (5.0, 4.05), text=r'$\hat b$', toff=(-0.05, 0.15))
    arrow(ax, (4.45, 2.6), (5.0, 2.45), text=r'$f$', toff=(-0.05, -0.15))
    arrow(ax, (6.03, 3.75), (6.03, 3.1), text=r'$i$', toff=(0.15, 0))
    arrow(ax, (7.05, 2.52), (7.55, 2.52))
    arrow(ax, (8.05, 3.1), (8.05, 3.75), text=r'$v_{\mathrm{yaw}}$', toff=(0.3, 0))
    arrow(ax, (8.55, 4.3), (9.1, 4.05))
    ax.text(8.05, 4.83, r'观测含 $z-\hat b$', ha='center', fontsize=7.5, color=COL['ink2'])
    ax.text(3.6, 2.2, r'$f$：融合航向', ha='center', va='top', fontsize=7.5, color=COL['ink2'])
    # 物理回路（虚线）：机器人运动决定陀螺与参考"看到"的 ω、ψ
    loop = dict(ls='--', lw=0.9, color=COL['gray'])
    ax.plot([9.42, 9.42, 0.22, 0.22], [2.85, 0.35, 0.35, 4.08], **loop)
    for yy in (4.08, 2.48):
        ax.annotate('', xy=(0.55, yy), xytext=(0.22, yy),
                    arrowprops=dict(arrowstyle='-|>', color=COL['gray'], lw=0.9, mutation_scale=9))
    ax.plot([0.22, 0.22], [2.48, 2.48], **loop)
    ax.text(4.8, 0.47, r'机体真实运动 $\omega,\ \psi$（只能经传感器间接得到）', ha='center', va='bottom',
            fontsize=7.5, color=COL['ink2'])
    ax.text(6.03, 1.78, r'$\tau\to 0$：$S=f$（方案 C）      $\tau\to\infty$：$S=i$（方案 B）',
            ha='center', va='top', fontsize=7.8, color=COL['S'])
    ax.text(6.03, 1.38, r'中间的 $\tau$：只对 $f-i$ 低通，陀螺增量原样保留', ha='center', va='top',
            fontsize=7.5, color=COL['ink2'])
    save(fig, 'f02_system')


# ---------------------------------------------------------------------------
# F03 点采样与步态锁相的混叠（第 2 章）
# ---------------------------------------------------------------------------
def f03_aliasing():
    """(a) 示意：窄冲击 + 宽回摆，一个周期内积分为 0；固定相位点采样的均值不为 0。
    (b) 实测：正式批次 G_vx0.5 三个种子，按 30 个采样相位折叠的 (点采样 − 区间平均) ω_z。"""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw=dict(width_ratios=[1.15, 1]))
    T, n = 0.6, 30
    t = np.linspace(0, 2 * T, 2000)

    def omega(tt):
        ph = np.mod(tt, T)
        spike = 1.0 * np.exp(-0.5 * ((ph - 0.08) / 0.004) ** 2)
        back = -1.0 * 0.004 / 0.03 * np.exp(-0.5 * ((ph - 0.33) / 0.03) ** 2)
        return 0.25 * np.sin(2 * np.pi * tt / T) + spike + back
    tk = np.arange(0, 2 * T - 1e-9, DT)
    a1.plot(t, omega(t), color=COL['gray'], lw=1.0, label='连续角速度')
    a1.plot(tk, omega(tk), 'o', ms=2.6, color=COL['C'], label='50 Hz 点采样')
    a1.axhline(0, color=COL['ink'], lw=0.7)
    sm = omega(tk[:n]).mean()
    a1.axhline(sm, color=COL['B'], lw=1.1, ls='--')
    # 标签放在正弦波谷之上的空白处（0.33–0.57 s 曲线低于 −0.15）
    a1.text(0.45, sm + 0.025, '点采样平均', color=COL['B'], ha='center', va='bottom', fontsize=7.3)
    a1.text(0.45, -0.025, '区间平均 0', color=COL['ink'], ha='center', va='top', fontsize=7.3)
    kpk = int(round(0.08 / DT))
    a1.annotate('每周期同相位采到冲击峰', xy=(tk[kpk], omega(tk[kpk])), xytext=(0.13, 1.3),
                fontsize=7.3, color=COL['ink2'], va='center', ha='left',
                arrowprops=dict(arrowstyle='-|>', color=COL['ink2'], lw=0.7, mutation_scale=8))
    for x in (0, T, 2 * T):
        a1.axvline(x, color=COL['light'], lw=4, zorder=0)
    a1.set_xlabel('时间 [s]（步态周期 0.6 s = 30 个控制步）')
    a1.set_ylabel(r'$\omega_z$（示意单位）')
    a1.legend(loc='upper right', fontsize=7, handlelength=1.5, bbox_to_anchor=(1.0, 0.9))
    a1.set_ylim(-0.38, 1.45)
    panel(a1, '(a) 示意')
    prof = []
    for s in (71000, 71001, 71002):
        a, c, _, _ = formal('G_vx0.5_p_Cs_s%d' % s)
        ph = phase_of(a, c)[250:]
        d = (a[:, c['raw_yaw']] - a[:, c['raw_yaw_interval']])[250:]
        b = np.floor(ph / (2 * np.pi) * n + 0.5).astype(int) % n
        prof.append([d[b == j].mean() for j in range(n)])
    prof = np.array(prof)
    m = prof.mean(0)
    a2.bar(np.arange(n), m, width=0.8, color=[COL['B'] if v < 0 else COL['C'] for v in m])
    a2.axhline(0, color=COL['ink'], lw=0.6)
    overall = prof.mean()
    a2.axhline(overall, color=COL['ink'], lw=1.0, ls='--')
    a2.text(8.5, 0.105, '虚线：30 个相位的平均 %.4f rad/s\n（相位固定，所以不会随时间平均掉）' % overall,
            ha='left', va='top', fontsize=7.3, color=COL['ink'])
    a2.set_xlabel('采样相位（第几个控制步，共 30）')
    a2.set_ylabel('点采样 − 区间平均 [rad/s]')
    a2.set_xlim(-0.8, n - 0.2)
    panel(a2, '(b) 实测（vx = 0.5 m/s，3 个种子）')
    fig.tight_layout(w_pad=1.2)
    save(fig, 'f03_aliasing')


FIGS = {}   # 名称 -> 函数，文件末尾汇总


# ---------------------------------------------------------------------------
# F04 步态运动学项：倾斜环、两成分随步速、三种积分（第 2 章）
# ---------------------------------------------------------------------------
def _phase_mean(x, ph, n=30):
    b = np.floor(ph / (2 * np.pi) * n + 0.5).astype(int) % n
    return np.array([x[b == j].mean() for j in range(n)])


def f04_gait():
    """正式批次 G（point，C*，vx 6 档 × 种子 71000–71002）。
    (a) 按 30 个相位平均的横滚-俯仰环；(b) e = 采样项 + 欧拉项；(c) 60 s 末端误差，三种积分方式。"""
    import design_family as df
    VX = (0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
    rows = {vx: [] for vx in VX}
    loops = {}
    for vx in VX:
        for s in (71000, 71001, 71002):
            a, c, _, _ = formal('G_vx%g_p_Cs_s%d' % (vx, s))
            phi, th, ps = a[:, c['gt_roll']], a[:, c['gt_pitch']], np.unwrap(a[:, c['gt_yaw']])
            rate = np.diff(ps) / DT
            pt, iv = a[1:, c['raw_yaw']], a[1:, c['raw_yaw_interval']]
            est = df.interval_rate(phi, th, ps)
            eul = (est[:, 1] * np.sin(phi[:-1]) + est[:, 2] * np.cos(phi[:-1])) / np.cos(th[:-1])
            sl = slice(250, None)
            T = (len(rate) - 250) * DT
            rows[vx].append(dict(samp=(pt - iv)[sl].mean(), euler=(iv - rate)[sl].mean(), e=(pt - rate)[sl].mean(),
                                 end_pt=DEG * abs((pt - rate)[sl].mean()) * T,
                                 end_z=DEG * abs((iv - rate)[sl].mean()) * T,
                                 end_eu=DEG * abs((eul - rate)[sl].mean()) * T))
            if vx in (0.2, 1.0) and s == 71000:
                ph = phase_of(a, c)[250:]
                loops[vx] = (_phase_mean(DEG * th[250:], ph), _phase_mean(DEG * phi[250:], ph))
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(W, 2.25), gridspec_kw=dict(width_ratios=[0.9, 1.1, 1]))
    for vx, col in ((0.2, COL['B']), (1.0, COL['C'])):
        x, y = loops[vx]
        x, y = x - x.mean(), y - y.mean()
        xx, yy = np.r_[x, x[0]], np.r_[y, y[0]]
        a1.fill(xx, yy, color=col, alpha=0.15, lw=0)
        a1.plot(xx, yy, color=col, lw=1.1, label='vx = %g' % vx)
        for j in (4, 19):
            a1.annotate('', xy=(x[j + 1], y[j + 1]), xytext=(x[j], y[j]),
                        arrowprops=dict(arrowstyle='-|>', color=col, lw=1.0, mutation_scale=9))
    a1.set_xlabel(r'俯仰 $\theta$ [°]（去均值）')
    a1.set_ylabel(r'横滚 $\phi$ [°]（去均值）')
    a1.set_aspect('equal', adjustable='datalim')
    a1.legend(loc='upper left', fontsize=7, bbox_to_anchor=(-0.02, 1.02), handlelength=1.2)
    panel(a1, '(a)', x=-0.3)
    xs = np.arange(len(VX))
    m = lambda k: np.array([np.mean([r[k] for r in rows[v]]) for v in VX]) * 1e3
    a2.bar(xs - 0.18, m('samp'), 0.36, color=COL['C'], label='采样项')
    a2.bar(xs + 0.18, m('euler'), 0.36, color=COL['B'], label='欧拉（倾斜环）项')
    a2.plot(xs, m('e'), 'D', ms=4.5, mfc='white', mec=COL['ink'], mew=1.1, label=r'合计 $e$')
    a2.axhline(0, color=COL['ink'], lw=0.6)
    a2.set_xticks(xs, ['%g' % v for v in VX])
    a2.set_xlabel('步速 vx [m/s]')
    a2.set_ylabel(r'$\overline{\omega_z-\dot\psi}$ [$10^{-3}$ rad/s]')
    a2.legend(loc='upper left', fontsize=7)
    panel(a2, '(b)', x=-0.2)
    for key, col, mk, lab in (('end_pt', COL['gray'], 'o', r'点采样 $\omega_z$'),
                              ('end_z', COL['B'], 's', r'区间平均 $\omega_z$'),
                              ('end_eu', COL['S'], '^', '区间三轴 + 欧拉率公式')):
        a3.plot(VX, [np.mean([r[key] for r in rows[v]]) for v in VX], marker=mk, ms=4, color=col, label=lab)
    a3.set_yscale('log')
    a3.axhline(0.5, color=COL['ink2'], lw=0.7, ls=':')
    a3.set_xlabel('步速 vx [m/s]')
    a3.set_ylabel('60 s 末端航向误差 [°]')
    a3.legend(loc='lower right', fontsize=6.8)
    panel(a3, '(c)', x=-0.2)
    fig.tight_layout(w_pad=0.8)
    save(fig, 'f04_gait')


# ---------------------------------------------------------------------------
# F05 误差预算（第 3、4 章）
# ---------------------------------------------------------------------------
def f05_budget():
    """正式批次 N（10 种子）：确定性部分（运动学+零偏）在闭环路径上重放得到；陀螺、参考噪声用理论期望。
    (a) [30,60) s 均方误差按来源堆叠（点采样陀螺），柱顶标 RMS；(b) C* 各来源单独的 RMS，点采样 vs 区间平均。"""
    import design_family as df
    from analyze_design_family import BMI088
    arms = [('Bs', 'B*', 1e-4, math.inf), ('Cs', 'C*', 1e-16, 0.0), ('S1', 'S1', 1e-16, 1.0)]
    sl = slice(1500, 3000)
    names = ('确定性（运动学+零偏）', '陀螺白噪声', '参考噪声')
    cols = (COL['gray'], COL['ref'], COL['C'])

    def budget(v, arm, q, tau):
        tr = [df.load_trace(formal('N_%s_%s_s%d' % (v, arm, s))[3]) for s in range(70000, 70010)]
        parts = df.decompose(tr, BMI088, q, 0.05)
        var = df.noise_variance(tr[0].upd, BMI088, q, 0.05, [tau])
        det = sum(df.smooth(parts[k][0], parts[k][1], tau) for k in ('kin', 'bias'))[:, sl]
        return [DEG ** 2 * (det ** 2).mean(), DEG ** 2 * var['gyro'][0, sl].mean(), DEG ** 2 * var['ref'][0, sl].mean()]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw=dict(width_ratios=[1, 1.05]))
    for j, (arm, lab, q, tau) in enumerate(arms):
        vals = budget('p', arm, q, tau)
        bottom = 0
        for val, col, name in zip(vals, cols, names):
            a1.bar(j, val, 0.6, bottom=bottom, color=col, label=name if j == 0 else None)
            bottom += val
        a1.text(j, bottom + 0.08, 'RMS %.2f°' % math.sqrt(bottom), ha='center', va='bottom', fontsize=7.5)
    a1.set_xticks(range(3), [a[1] for a in arms])
    a1.set_ylabel('均方误差 [deg²]')
    a1.set_ylim(0, 6.6)
    a1.legend(loc='upper right', fontsize=7)
    a1.text(0, 1.33 / 2, '零偏收敛\n暂态', ha='center', va='center', fontsize=7, color='white')
    panel(a1, '(a) 点采样陀螺，三个方案', x=-0.16)
    bp, bi = budget('p', 'Cs', 1e-16, 0.0), budget('i', 'Cs', 1e-16, 0.0)
    xs = np.arange(3)
    for off, vals, hatch, lab in ((-0.19, bp, None, '点采样陀螺'), (0.19, bi, '////', '区间平均陀螺')):
        r = np.sqrt(vals)
        a2.bar(xs + off, r, 0.36, color=cols, hatch=hatch, edgecolor='white', lw=0, label=lab)
        for x, y in zip(xs + off, r):
            a2.text(x, y + 0.015, '%.3f' % y, ha='center', va='bottom', fontsize=6.8)
    a2.set_xticks(xs, ['确定性', '陀螺噪声', '参考噪声'])
    a2.set_ylabel('单一来源的 RMS [°]')
    a2.set_ylim(0, 0.92)
    from matplotlib.patches import Patch
    a2.legend(handles=[Patch(fc=COL['gray'], label='点采样陀螺'), Patch(fc=COL['gray'], hatch='////', ec='white',
                                                                    label='区间平均陀螺')], loc='upper left', fontsize=7)
    panel(a2, '(b) C* 的各来源', x=-0.15)
    fig.tight_layout(w_pad=1.0)
    save(fig, 'f05_budget')


# ---------------------------------------------------------------------------
# F06 B 为什么比 C 差：B 记住误差，C 不断被参考"刷新"（第 3 章）
# ---------------------------------------------------------------------------
def f06_bvsc():
    """(a) 理论：仅参考噪声造成的误差标准差随时间（1 Hz、σa=0.05、BMI088 σg，精确协方差递推）。
    (b) 离线网格（bias_002 路径 × BMI088 σg，[30,60) s 期望 RMS）：τ=0（C 族）与 τ=∞（B 族）随 q_b 的变化。"""
    import design_family as df
    from analyze_design_family import BMI088
    n = 15000
    k = np.arange(n)
    upd = (k % 50 == 0) & (k > 0)
    t = k * DT
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.35))
    for q, tau, col, ls, lab in ((1e-16, 0.0, COL['C'], '-', r'C：$q_b=10^{-16}$，$\tau=0$'),
                                 (1e-16, math.inf, COL['B'], '-', r'B：$q_b=10^{-16}$，$\tau=\infty$'),
                                 (1e-4, math.inf, COL['B'], '--', r'B*：$q_b=10^{-4}$，$\tau=\infty$')):
        sd = DEG * np.sqrt(df.noise_variance(upd, BMI088, q, 0.05, [tau])['ref'][0])
        a1.plot(t[50:], sd[50:], color=col, ls=ls, lw=1.2, label=lab)
    N = np.maximum(np.floor(t / 1.0), 1)
    a1.plot(t[500:], DEG * 2 * 0.05 / np.sqrt(N[500:]), color=COL['ink'], lw=0.8, ls=':', label=r'近似 $2\sigma_a/\sqrt{N}$')
    a1.set_xscale('log'); a1.set_yscale('log')
    a1.set_xlim(1, 300)
    a1.set_ylim(0.03, 12)
    a1.set_xlabel('时间 [s]')
    a1.set_ylabel('参考噪声误差 SD [°]')
    a1.legend(loc='lower left', fontsize=6.8)
    a1.text(25, 5.2, 'B：早期零偏估计误差被积分后留下', fontsize=7, color=COL['B'], ha='center')
    a1.text(80, 0.33, 'C：每次参考\n都修正航向', fontsize=7, color=COL['C'], ha='center', va='top')
    panel(a1, '(a)', x=-0.2)
    g = json.loads((OFFLINE / 'grid.json').read_text(encoding='utf-8'))
    rows = g['scenarios']['BMI088 · 零偏 0.02']['rows']
    best = {}
    for tau, col, lab in ((0.0, COL['C'], r'C 族（$\tau=0$）'), (math.inf, COL['B'], r'B 族（$\tau=\infty$）')):
        rs = sorted((r for r in rows if float(r['tau']) == tau), key=lambda r: r['q_b'])
        q = [r['q_b'] for r in rs]
        y = [r['windows']['30-60']['expected_rms'] for r in rs]
        a2.plot(q, y, 'o-', ms=3, color=col, label=lab)
        j = int(np.argmin(y))
        a2.plot(q[j], y[j], '*', ms=10, color=col, mec='white', mew=0.5, zorder=5)
        best[tau] = y[j]
    a2.set_xscale('log')
    a2.set_xlabel(r'零偏过程噪声 $q_b$ [(rad/s)²/步]')
    a2.set_ylabel('[30, 60) s 期望 RMS [°]')
    a2.set_ylim(0, 5.5)
    a2.legend(loc='upper left', fontsize=7)
    a2.text(1e-13, 1.45, '各取最优（★）：C %.2f° < B %.2f°' % (best[0.0], best[math.inf]), fontsize=7.3,
            color=COL['ink'])
    a2.annotate('强制共用同一 $q_b$ 时\n排序在此"翻转"', (1.3e-5, 2.52), xytext=(1e-11, 2.55), fontsize=7,
                color=COL['ink2'], arrowprops=dict(arrowstyle='-|>', color=COL['ink2'], lw=0.7, mutation_scale=8))
    panel(a2, '(b)', x=-0.16)
    fig.tight_layout(w_pad=1.0)
    save(fig, 'f06_bvsc')


# ---------------------------------------------------------------------------
# F07 中断后的"失忆"与恢复跳变（第 5 章）
# ---------------------------------------------------------------------------
def f07_outage_mech():
    """正式批次 O_p_Cs_s70000 的闭环路径，离线重放 q_b=1e-16：同一路径上 τ=0（C）与 τ=1 s（S1）的估计误差，
    以及每次参考更新时的卡尔曼增益 K0。S1 曲线是同一路径上的反事实重放。"""
    import design_family as df
    from analyze_design_family import BMI088
    tr = df.load_trace(formal('O_p_Cs_s70000')[3])
    parts = df.decompose([tr], BMI088, 1e-16, 0.05)
    e = {tau: DEG * sum(df.smooth(parts[kk][0], parts[kk][1], tau) for kk in df.SOURCES)[0] for tau in (0.0, 1.0)}
    K0, _ = df.gains(tr.upd, BMI088, 1e-16, 0.05)
    t = np.arange(len(tr.psi)) * DT
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 3.3), sharex=True, gridspec_kw=dict(height_ratios=[2.1, 1]))
    for ax in (a1, a2):
        ax.axvspan(20, 50, color=COL['light'], zorder=0)
    a1.plot(t, e[0.0], color=COL['C'], lw=1.0, label=r'C（$\tau=0$）')
    a1.plot(t, e[1.0], color=COL['S'], lw=1.2, label=r'S1（$\tau=1$ s）')
    a1.axhline(0, color=COL['ink'], lw=0.5)
    a1.set_ylabel('估计误差 [°]')
    a1.set_xlim(10, 70)
    a1.set_ylim(-6.2, 12)
    a1.set_yticks([-5, 0, 5])
    a1.text(21, -5.6, '参考中断 30 s', ha='left', fontsize=7.5, color=COL['ink2'])
    a1.legend(loc='upper right', fontsize=7, ncol=2)
    # 放大窗放在中断区间上方的空白处（那里误差接近 0），刻度标签不压到曲线
    ins = a1.inset_axes([0.2, 0.52, 0.36, 0.42])
    sel = (t >= 49.4) & (t <= 52.2)
    ins.plot(t[sel], e[0.0][sel], color=COL['C'], lw=1.0)
    ins.plot(t[sel], e[1.0][sel], color=COL['S'], lw=1.2)
    ins.axvline(50.0, color=COL['ink2'], lw=0.6, ls=':')
    ins.set_title('50 s 恢复瞬间（放大）', fontsize=6.8, pad=2)
    ins.tick_params(labelsize=6)
    a1.indicate_inset_zoom(ins, edgecolor=COL['gray'], lw=0.6)
    panel(a1, '(a)', x=-0.09)
    kk = np.nonzero(tr.upd)[0]
    a2.vlines(t[kk], 1e-3, K0[kk], color=COL['C'], lw=0.9)
    a2.plot(t[kk], K0[kk], 'o', ms=2.2, color=COL['C'])
    a2.set_yscale('log')
    a2.set_ylim(5e-3, 1.5)
    a2.set_ylabel(r'增益 $K_0$')
    a2.set_xlabel('时间 [s]')
    k1 = kk[kk >= 2500][0]
    a2.annotate('恢复后首次更新 $K_0$=%.2f：\n几乎直接采信一个带噪参考' % K0[k1], (t[k1], K0[k1]), xytext=(55, 0.35),
                fontsize=7, color=COL['ink2'], arrowprops=dict(arrowstyle='-|>', color=COL['ink2'], lw=0.7,
                                                                mutation_scale=8))
    panel(a2, '(b)', x=-0.09)
    fig.tight_layout(h_pad=0.4)
    save(fig, 'f07_outage_mech')


# ---------------------------------------------------------------------------
# 实验结果图
# ---------------------------------------------------------------------------
TAU_POS = lambda t: 0.03 if t == 0 else (1000.0 if math.isinf(t) else t)   # 把 0 与 ∞ 放到对数轴两端


def _tau_axis(ax, taus):
    ticks = [0.03, 0.1, 1, 10, 100, 1000]
    ax.set_xscale('log')
    ax.set_xticks(ticks, ['0', '0.1', '1', '10', '100', r'$\infty$'])
    ax.set_xlim(0.02, 1500)


def f08_plane():
    """离线网格（bias_002 路径 × BMI088 σg，σa=0.05，1 Hz，[30,60) s）。
    (a) (q_b, τ) 平面上的期望 RMS；(b) q_b=1e-16 上的 τ 剖面：RMS 与估计误差最大单步跳变。"""
    import matplotlib.patheffects as pe
    g = json.loads((OFFLINE / 'grid.json').read_text(encoding='utf-8'))
    rows = g['scenarios']['BMI088 · 零偏 0.02']['rows']
    qs = sorted({r['q_b'] for r in rows})
    taus = sorted({float(r['tau']) for r in rows})
    Z = np.array([[next(r for r in rows if r['q_b'] == q and float(r['tau']) == t)['windows']['30-60']['expected_rms']
                   for t in taus] for q in qs])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
    im = a1.imshow(Z, origin='lower', aspect='auto', cmap='viridis_r', vmin=0.75, vmax=4.5)
    shown = {0.0: '0', 0.1: '0.1', 1.0: '1', 10.0: '10', 100.0: '100', math.inf: r'$\infty$'}
    a1.set_xticks(range(len(taus)), [shown.get(t, '') for t in taus], fontsize=6.5)

    def qlab(q):
        e = math.floor(math.log10(q) + 1e-9)
        m = q / 10 ** e
        return r'$10^{%d}$' % e if abs(m - 1) < 1e-6 else r'$%g\times10^{%d}$' % (m, e)
    a1.set_yticks(range(len(qs)), [qlab(q) for q in qs], fontsize=6.5)
    cs = a1.contour(Z, levels=[1.0, 1.5, 2.0, 3.0], colors='white', linewidths=0.6)
    a1.clabel(cs, fmt='%.1f°', fontsize=6)
    halo = [pe.withStroke(linewidth=2.2, foreground='white')]
    for q, t, lab, col, off, ha in ((1e-4, math.inf, 'B*', COL['B'], (-7, 5), 'right'),
                                    (1e-16, 0.0, 'C*', COL['C'], (6, 4), 'left'),
                                    (1e-16, 1.0, 'S1', COL['S'], (6, 4), 'left')):
        x, y = taus.index(t), qs.index(q)
        a1.plot(x, y, 'o', ms=8, mfc='none', mec=COL['ink'], mew=2.4)
        a1.plot(x, y, 'o', ms=8, mfc='none', mec='white', mew=1.1)
        a1.annotate(lab, (x, y), xytext=off, textcoords='offset points', ha=ha, color=col, fontsize=7.5,
                    fontweight='bold', path_effects=halo)
    a1.set_xlabel(r'平滑时间常数 $\tau$ [s]')
    a1.set_ylabel(r'$q_b$ [(rad/s)²/步]')
    cb = fig.colorbar(im, ax=a1, fraction=0.05, pad=0.02)
    cb.set_label('期望 RMS [°]', fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    panel(a1, '(a)', x=-0.2)
    prof = sorted((r for r in rows if r['q_b'] == 1e-16), key=lambda r: float(r['tau']))
    x = [TAU_POS(float(r['tau'])) for r in prof]
    a2.axvline(1.0, color=COL['S'], lw=0.9, ls=':')
    a2.text(1.12, 1.3, r'S1（$\tau$ = 1 s）', ha='left', va='bottom', fontsize=6.8, color=COL['S'])
    a2.plot(x, [r['windows']['30-60']['expected_rms'] for r in prof], 'o-', ms=3, color=COL['C'], label='期望 RMS')
    a2.set_ylabel('期望 RMS [°]', color=COL['C'])
    a2.set_ylim(0, 4.2)
    a2b = a2.twinx()
    a2b.spines['right'].set_visible(True)
    a2b.plot(x, [r['jump_deg'] for r in prof], 's--', ms=3, color=COL['B'], label='最大单步跳变')
    a2b.set_ylabel('估计误差最大单步跳变 [°]', color=COL['B'])
    a2b.set_ylim(0, 1.8)
    _tau_axis(a2, None)
    a2.set_xlabel(r'$\tau$ [s]（$q_b=10^{-16}$）')
    h1, l1 = a2.get_legend_handles_labels()
    h2, l2 = a2b.get_legend_handles_labels()
    a2.legend(h1 + h2, l1 + l2, loc='upper center', fontsize=7)
    panel(a2, '(b)', x=-0.16)
    fig.tight_layout(w_pad=1.2)
    save(fig, 'f08_plane')


def f09_window():
    """τ 对精度与平滑的作用（q_b=1e-16，离线）。数据来自 analyze_thesis_supplement.py 第 6 节（supplement.json）。
    (a) 正式 N 点采样路径：期望 RMS / C* 随 τ，σa=0.05，各参考率；橙线为压力档陀螺 1 Hz；虚线 1.05。
    (b) 1 Hz、σa=0.05：估计误差最大单步变化随 τ，开发集路径、正式点采样、正式区间平均。"""
    tw = json.loads((SUPP / 'supplement.json').read_text(encoding='utf-8'))['tau_window']['data']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw=dict(width_ratios=[1.1, 1]))
    xs = lambda p: [TAU_POS(t) for t in p['taus']]
    blues = {0.25: '#9ecae1', 0.5: '#6baed6', 1.0: '#3182bd', 2.0: '#08519c', 5.0: '#08306b'}
    for hz, col in blues.items():
        p = tw['formal_p_bmi']['%g_0.05' % hz]
        a1.plot(xs(p), p['rms_ratio'], '-', color=col, lw=1.1, label='%g Hz' % hz)
    p = tw['formal_p_stress']['1_0.05']
    a1.plot(xs(p), p['rms_ratio'], '-', color=COL['B'], lw=1.3, label='压力档，1 Hz')
    a1.axhline(1.05, color=COL['ink'], lw=0.8, ls='--')
    a1.text(0.035, 1.065, '1.05 × C*', fontsize=6.8, color=COL['ink'])
    _tau_axis(a1, None)
    a1.set_ylim(0.97, 1.5)
    a1.set_xlabel(r'$\tau$ [s]')
    a1.set_ylabel('期望 RMS / C*')
    a1.legend(loc='upper left', fontsize=6.5, ncol=2, frameon=False, bbox_to_anchor=(0.0, 0.93))
    panel(a1, '(a) 精度上界', x=-0.15)
    for key, col, ls, lab in (('dev_bmi', COL['gray'], '--', '开发集路径'), ('formal_p_bmi', COL['C'], '-', '正式，点采样'),
                              ('formal_i_bmi', COL['S'], '-', '正式，区间平均')):
        p = tw[key]['1_0.05']
        a2.plot(xs(p), p['jump'], ls, marker='o', ms=2.5, color=col, lw=1.1, label=lab)
    a2.axvline(1.0, color=COL['ink2'], lw=0.7, ls=':')
    a2.text(1.1, 2.2, r'S1（$\tau$ = 1 s）', fontsize=6.8, color=COL['ink2'])
    _tau_axis(a2, None)
    a2.set_yscale('log')
    a2.set_yticks([0.01, 0.03, 0.1, 0.3, 1], ['0.01', '0.03', '0.1', '0.3', '1'])
    a2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a2.set_ylim(0.007, 3.5)
    a2.set_xlabel(r'$\tau$ [s]')
    a2.set_ylabel('估计误差最大单步变化 [°]')
    a2.legend(loc='lower left', fontsize=6.5, frameon=False)
    panel(a2, '(b) 平滑（1 Hz）', x=-0.18)
    fig.tight_layout(w_pad=1.2)
    save(fig, 'f09_window')


def _paired(ax, rows, scene, groups, key, ylabel):
    """groups: [(variant, arm, 标签)]；同 variant 内同种子连线，黑点为均值 ± 95% t 区间。"""
    from scipy import stats
    xs = []
    for j, (v, arm, lab) in enumerate(groups):
        vals = {r['seed']: r[key] for r in rows if r['scene'] == scene and r['variant'] == v and r['arm'] == arm}
        xs.append((j, v, arm, vals))
        col = {'Bs': COL['B'], 'Cs': COL['C'], 'S1': COL['S']}[arm]
        seeds = sorted(vals)
        ax.plot(np.full(len(seeds), j) + np.linspace(-0.08, 0.08, len(seeds)), [vals[s] for s in seeds], 'o',
                ms=3, color=col, alpha=0.75)
        a = np.array([vals[s] for s in seeds])
        h = stats.t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / math.sqrt(len(a))
        ax.errorbar(j + 0.22, a.mean(), yerr=h, fmt='D', ms=4, color=COL['ink'], capsize=2, lw=0.9)
    for (j0, v0, _, A), (j1, v1, _, B) in zip(xs[:-1], xs[1:]):
        if v0 == v1:
            for s in set(A) & set(B):
                ax.plot([j0, j1], [A[s], B[s]], color=COL['gray'], lw=0.5, alpha=0.6, zorder=0)
    ax.set_xticks(range(len(groups)), [g[2] for g in groups], fontsize=7.5)
    ax.set_ylabel(ylabel)


def f10_nominal():
    """正式批次 N（种子 70000–70009，两种陀螺信号）：[30,60) s 估计误差 RMS 与反馈量最大单步跳变。"""
    rows = formal_rows()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw=dict(width_ratios=[1.35, 1]))
    _paired(a1, rows, 'N', [('point', 'Bs', 'B*\n点采样'), ('point', 'Cs', 'C*\n点采样'), ('point', 'S1', 'S1\n点采样'),
                            ('interval', 'Bs', 'B*\n区间'), ('interval', 'Cs', 'C*\n区间'), ('interval', 'S1', 'S1\n区间')],
            'err', '[30, 60) s 估计误差 RMS [°]')
    a1.axvline(2.5, color=COL['light'], lw=6, zorder=0)
    a1.set_ylim(0, 3.1)
    panel(a1, '(a) 精度', x=-0.12)
    _paired(a2, rows, 'N', [('point', 'Cs', 'C*\n点采样'), ('point', 'S1', 'S1\n点采样'),
                            ('interval', 'Cs', 'C*\n区间'), ('interval', 'S1', 'S1\n区间')],
            'fb_jump', '反馈量最大单步跳变 [°]')
    a2.axvline(1.5, color=COL['light'], lw=6, zorder=0)
    a2.set_ylim(0, 2.8)
    panel(a2, '(b) 平滑', x=-0.16)
    fig.tight_layout(w_pad=1.0)
    save(fig, 'f10_nominal')


def f11_outage():
    """正式批次 O（point 陀螺，种子 70000–70009，[20,50) s 无参考）。
    (a) C* 与 S1 的估计误差：先对每个种子做一个步态周期（30 步 = 0.6 s）的滑动平均，再取 10 种子均值 ± SD；
    (b) B* 的估计误差（每个种子一条，线性纵轴）；(c) 恢复瞬间的反馈量，3 个种子，各自以 48.5 s 的值为零点。"""
    t = np.arange(4500) * DT
    E = {arm: np.array([DEG * heading_err(*formal('O_p_%s_s%d' % (arm, s))[:2]) for s in range(70000, 70010)])
         for arm in ('Bs', 'Cs', 'S1')}
    fig = plt.figure(figsize=(W, 3.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], height_ratios=[1, 1], hspace=0.45, wspace=0.28)
    a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    for ax in (a1, a2):
        ax.axvspan(20, 50, color=COL['light'], zorder=0)
        ax.set_xlim(0, 90)
    n = 30                                   # 一个步态周期；滑动平均正好抵消步态频率及其谐波
    tv = t[n - 1:] - (n - 1) * DT / 2        # 'valid' 卷积后的中心时刻
    for arm, col, lab in (('Cs', COL['C'], 'C*'), ('S1', COL['S'], 'S1')):
        sm = np.array([np.convolve(row, np.ones(n) / n, mode='valid') for row in E[arm]])
        m, sd = sm.mean(0), sm.std(0, ddof=1)
        a1.fill_between(tv, m - sd, m + sd, color=col, alpha=0.18, lw=0)
        a1.plot(tv, m, color=col, lw=1.0, label=lab + '（均值 ± SD）')
    a1.axhline(0, color=COL['ink'], lw=0.5)
    a1.set_ylabel('估计误差 [°]')
    a1.set_ylim(-3, 3)
    a1.legend(loc='lower right', fontsize=6.8, ncol=2, frameon=False)
    a1.text(35, 2.3, '参考中断', ha='center', fontsize=7, color=COL['ink2'])
    panel(a1, '(a)', x=-0.12)
    for row in E['Bs']:
        a2.plot(t, row, color=COL['B'], lw=0.7, alpha=0.8)
    a2.axhline(0, color=COL['ink'], lw=0.5)
    a2.set_ylim(-140, 140)
    a2.set_yticks([-100, 0, 100])
    a2.set_ylabel('B* 估计误差 [°]')
    a2.set_xlabel('时间 [s]')
    a2.text(70, 80, '恢复后不回到 0：B 的航向\n从不被参考直接校正', ha='center', va='center', fontsize=6.8,
            color=COL['B'])
    panel(a2, '(b)', x=-0.12)
    sel = (t >= 48.5) & (t <= 54)
    gap = 7.0
    for k, s in enumerate((70000, 70003, 70006)):
        for arm, col in (('Cs', COL['C']), ('S1', COL['S'])):
            a, c, _, _ = formal('O_p_%s_s%d' % (arm, s))
            fb = DEG * a[:, c['feedback_heading']]
            a3.plot(t[sel], fb[sel] - fb[int(48.5 / DT)] + gap * k, color=col, lw=1.0,
                    label=({'Cs': 'C*', 'S1': 'S1'}[arm]) if k == 0 else None)
        a3.text(53.95, gap * k + 1.8, '种子 %d' % s, ha='right', fontsize=6.5, color=COL['ink2'])
    a3.axvline(50.0, color=COL['ink2'], lw=0.6, ls=':')
    a3.text(50.1, gap * 2 + 3.3, '参考恢复', va='top', fontsize=6.5, color=COL['ink2'])
    # 纵轴刻度没有意义（各条已平移），改用比例尺
    a3.plot([53.8, 53.8], [-5.5, -3.5], color=COL['ink'], lw=1.2)
    a3.text(53.7, -4.5, '2°', ha='right', va='center', fontsize=6.8)
    a3.set_ylim(-7, gap * 2 + 3.5)
    a3.set_xlabel('时间 [s]')
    a3.set_ylabel('反馈量（各自平移）')
    a3.set_yticks([])
    a3.legend(loc='lower left', ncol=1, fontsize=7, frameon=False)
    panel(a3, '(c) 恢复瞬间', x=-0.1, y=1.02)
    save(fig, 'f11_outage')


def f12_speed():
    """正式批次 V（种子 70000–70004）：vx 在 30 s 由 0.5 变为 0.8。C*（q_b=1e-16）与 C8（q_b=1e-8）的
    估计误差绝对值（5 种子平均，1 s 滑动平均）。灰底为预注册评价窗口 [60,90) s。"""
    t = np.arange(4500) * DT
    k = np.ones(50) / 50
    rows = formal_rows()
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.3), sharey=True)
    for ax, (v, vv, name) in zip(axes, (('p', 'point', '点采样陀螺'), ('i', 'interval', '区间平均陀螺'))):
        ax.axvspan(60, 90, color=COL['light'], zorder=0)
        ax.axvline(30, color=COL['ink2'], lw=0.7, ls=':')
        for arm, col, lab in (('Cs', COL['C'], r'C*（$q_b=10^{-16}$）'), ('C8', COL['C8'], r'C8（$q_b=10^{-8}$）')):
            e = np.array([np.abs(DEG * heading_err(*formal('V_%s_%s_s%d' % (v, arm, s))[:2])) for s in range(70000, 70005)])
            ax.plot(t[25:-25], np.convolve(e.mean(0), k, mode='same')[25:-25], color=col, lw=1.1, label=lab)
            m = np.mean([r['err_post'] for r in rows if r['scene'] == 'V' and r['variant'] == vv and r['arm'] == arm])
            ax.text(75, {'Cs': 2.05, 'C8': 1.85}[arm], '%s %.2f°' % ({'Cs': 'C*', 'C8': 'C8'}[arm], m),
                    ha='center', fontsize=6.8, color=col)
        ax.text(75, 2.27, '[60, 90) s RMS', ha='center', fontsize=6.8, color=COL['ink2'])
        ax.set_title(name, fontsize=8.5)
        ax.set_xlabel('时间 [s]')
        ax.set_xlim(6, 90)
        ax.text(31, 2.35, 'vx 0.5→0.8', fontsize=7, color=COL['ink2'])
    axes[0].set_ylabel('|估计误差| [°]')
    axes[0].set_ylim(0, 2.6)
    axes[0].legend(loc='upper left', bbox_to_anchor=(0.0, 0.84), fontsize=6.8, frameon=False)
    panel(axes[0], '(a)', x=-0.14)
    panel(axes[1], '(b)', x=-0.05)
    fig.tight_layout(w_pad=0.6)
    save(fig, 'f12_speed')


def f13_pulse():
    """正式批次 pulse_formal_20260929（C*，点采样陀螺，脉冲持续 1 s，种子 72000–72004，起点错开 0.12 s，0/35 摔倒）。
    超额 = 同种子 Δtilt(A) − Δtilt(0)。竖线：正式批次 N 中实际的指令最大单步变化（点采样）：C* 在 [5 s, 末尾) 的均值与最大值（出现在收敛期）、C* 在稳态 [30, 60) s 的均值（supplement.json）、S1 的均值。
    (a) 实际跳变量级的幅值：种子点、均值（菱形）与单侧 95% 上界；点划线为预注册非劣界 0.10°。"""
    from matplotlib.lines import Line2D
    res = json.loads((PULSE / 'results.json').read_text(encoding='utf-8'))
    by = {(r['amp'], r['seed']): r for r in res['rows']}
    seeds = sorted({r['seed'] for r in res['rows']})
    amps = sorted({r['amp'] for r in res['rows'] if r['amp'] > 0})
    exc = {a: np.array([by[(a, s)]['d_tilt'] - by[(0.0, s)]['d_tilt'] for s in seeds]) for a in amps}
    dev = {a: np.array([by[(a, s)]['yaw_dev'] for s in seeds]) for a in amps}
    base = np.mean([by[(0.0, s)]['yaw_dev'] for s in seeds])
    tests = {t['amp']: t for t in res['tests']}
    rows = formal_rows()
    nC = [r['cmd_jump'] for r in rows if r['scene'] == 'N' and r['arm'] == 'Cs' and r['variant'] == 'point']
    nS = [r['cmd_jump'] for r in rows if r['scene'] == 'N' and r['arm'] == 'S1' and r['variant'] == 'point']
    steady = json.loads((SUPP / 'supplement.json').read_text(encoding='utf-8'))['nominal']['data']['p_Cs']['cmd30']
    marks = ((np.mean(nC), COL['C'], '--', 'C* 收敛期跳变：均值'), (np.max(nC), COL['C'], ':', 'C* 收敛期：最大'),
             (np.mean(steady), COL['C8'], '-.', 'C* 稳态'), (np.mean(nS), COL['S'], '--', 'S1'))
    fig = plt.figure(figsize=(W, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.1, 1.0], wspace=0.5, left=0.08, right=0.99, top=0.88, bottom=0.27)
    a1, a2, a3 = (fig.add_subplot(gs[0, j]) for j in range(3))
    jit = np.linspace(0.9, 1.1, len(seeds))

    def xaxis(ax, hi, ticks):
        ax.set_xscale('log')
        ax.set_xlim(0.006, hi)
        ax.set_xticks(ticks, ['%g' % t for t in ticks])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(r'脉冲幅值 $A$ [rad/s]')
        for x, c, ls, _ in marks:
            ax.axvline(x, color=c, lw=0.9, ls=ls)
    for a in (a for a in amps if a <= 0.25):
        a1.plot(a * jit, exc[a], 'o', ms=3, color=COL['B'], alpha=0.75)
        m, t = exc[a].mean(), tests.get(a)
        if t is not None and t['kind'] == 'noninf':
            a1.plot([a * 1.2] * 2, [m, t['bound']], color=COL['ink'], lw=1.0)
            a1.plot(a * 1.2, t['bound'], '_', ms=6, mew=1.2, color=COL['ink'])
        a1.plot(a * 1.2, m, 'D', ms=3.5, color=COL['ink'])
    a1.axhline(0.10, color=COL['ink2'], lw=0.8, ls='-.')
    a1.text(0.0065, 0.112, '非劣界 0.10°', fontsize=6.5, color=COL['ink2'])
    a1.axhline(0, color=COL['ink'], lw=0.5)
    xaxis(a1, 0.4, [0.01, 0.03, 0.1, 0.3])
    a1.set_ylim(-0.08, 0.3)
    a1.set_ylabel('倾角最大值的超额 [°]')
    panel(a1, '(a) 实际跳变量级', x=-0.32)
    for a in amps:
        a2.plot(a * jit, exc[a], 'o', ms=3, color=COL['B'], alpha=0.75)
    a2.plot(amps, [exc[a].mean() for a in amps], '-', color=COL['B'], lw=1.0)
    a2.axvspan(1.0, 3.5, color=COL['light'], zorder=0)
    a2.text(1.9, 0.3, '超出训练\n指令范围', ha='center', va='center', fontsize=6.5, color=COL['ink2'])
    a2.axhline(0, color=COL['ink'], lw=0.5)
    xaxis(a2, 3.5, [0.01, 0.1, 1])
    a2.set_ylim(-0.1, 2.8)
    a2.set_ylabel('倾角最大值的超额 [°]')
    panel(a2, '(b) 全部幅值（0/35 摔倒）', x=-0.25)
    for a in amps:
        a3.plot(a * jit, dev[a], 'o', ms=3, color=COL['ink2'], alpha=0.75)
    a3.plot(amps, [dev[a].mean() for a in amps], '-', color=COL['ink2'], lw=1.0)
    aa = np.geomspace(0.2, 2.5, 20)
    a3.plot(aa, 26 * aa, ':', color=COL['ink'], lw=0.8)
    a3.text(0.6, 32, r'$26^\circ\times A$', ha='right', fontsize=7, color=COL['ink'])
    a3.axhline(base, color=COL['gray'], lw=0.8, ls='-.')
    a3.text(0.3, base * 0.9, '无脉冲 %.1f°' % base, va='top', fontsize=6.5, color=COL['ink2'])
    a3.set_yscale('log')
    a3.set_yticks([2, 5, 10, 20, 50], ['2', '5', '10', '20', '50'])
    a3.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a3.set_ylim(1.5, 80)
    xaxis(a3, 3.5, [0.01, 0.1, 1])
    a3.set_ylabel('航向最大偏离 [°]')
    panel(a3, '(c) 航向', x=-0.25)
    fig.legend(handles=[Line2D([], [], color=c, ls=ls, lw=0.9, label=lab) for _, c, ls, lab in marks],
               loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=6.8, frameon=False)
    save(fig, 'f13_pulse')


def f14_rate():
    """(a) C* 的 [30,60) s 误差随参考率 f。灰线：离线扫描（bias_002 路径，σa=0.05）的理论期望；
    灰菱形：同一公式在本批闭环路径上的理论期望（5 条路径合并）；蓝点：本批闭环实测（0.5、2 Hz 来自 R，1 Hz 来自 N；
    种子 70000–70004；均值 ± SD）。实测低于期望是这批种子的参考噪声实现偏小（DEVIATIONS D4）。
    (b) 反馈量最大单步跳变随 f：C* 与 S1，两种陀螺信号。"""
    import design_family as df
    from analyze_design_family import BMI088, evaluate
    sw = json.loads((OFFLINE / 'sweeps.json').read_text(encoding='utf-8'))['rate']
    rows = formal_rows()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.3))
    off = sorted((r['hz'], r['c_star']) for r in sw if r['sigma_a'] == 0.05)
    a1.plot([h for h, _ in off], [c for _, c in off], 'o-', ms=3, color=COL['gray'], label='理论期望（离线扫描路径）')

    def sel(scene, v, arm, hz):
        return [r for r in rows if r['scene'] == scene and r['variant'] == v and r['arm'] == arm
                and (scene == 'N' and r['seed'] < 70005 or scene == 'R' and r['abs_hz'] == hz)]
    for v, mk, dx in (('point', 'o', 0.93), ('interval', 's', 1.07)):
        exp = []
        for hz in (0.5, 1.0, 2.0):
            ids = [('N_%s_Cs_s%d' % (v[0], s)) if hz == 1.0 else ('R_f%g_%s_Cs_s%d' % (hz, v[0], s))
                   for s in range(70000, 70005)]
            tr = [df.load_trace(formal(c)[3]) for c in ids]
            exp.append(evaluate(tr, BMI088, 0.05, [1e-16], [0.0], [(30, 60)])[0]['windows']['30-60']['expected_rms'])
        a1.plot([h * dx for h in (0.5, 1.0, 2.0)], exp, 'D', ms=4, mfc='white', mec=COL['ink2'], mew=0.9,
                label='理论期望（本批闭环路径）' if v == 'point' else None)
        pts = [(hz, sel('R' if hz != 1.0 else 'N', v, 'Cs', hz)) for hz in (0.5, 1.0, 2.0)]
        a1.errorbar([h * dx for h, _ in pts], [np.mean([r['err'] for r in rs]) for _, rs in pts],
                    yerr=[np.std([r['err'] for r in rs], ddof=1) for _, rs in pts], fmt=mk, ms=4, capsize=2,
                    color=COL['C'], mfc='white' if v == 'interval' else COL['C'], lw=0.9,
                    label='闭环实测 C*（%s）' % ('点采样' if v == 'point' else '区间'))
    a1.set_xscale('log')
    a1.set_xticks([0.25, 0.5, 1, 2, 5], ['0.25', '0.5', '1', '2', '5'])
    a1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a1.set_xlabel(r'参考更新率 $f_{\mathrm{ref}}$ [Hz]')
    a1.set_ylabel('[30, 60) s 误差 RMS [°]')
    a1.set_ylim(0, 1.8)
    a1.legend(loc='upper right', fontsize=6.5, frameon=False)
    panel(a1, '(a) 精度', x=-0.16)
    for arm, col in (('Cs', COL['C']), ('S1', COL['S'])):
        for v, mk, ls in (('point', 'o', '-'), ('interval', 's', '--')):
            ys = [np.mean([r['fb_jump'] for r in sel('R' if hz != 1.0 else 'N', v, arm, hz)]) for hz in (0.5, 1.0, 2.0)]
            a2.plot([0.5, 1.0, 2.0], ys, marker=mk, ls=ls, ms=4, color=col, mfc='white' if v == 'interval' else col,
                    label='%s（%s）' % ({'Cs': 'C*', 'S1': 'S1'}[arm], '点采样' if v == 'point' else '区间'))
    a2.set_xscale('log')
    a2.set_xticks([0.5, 1, 2], ['0.5', '1', '2'])
    a2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a2.set_xlabel(r'参考更新率 $f_{\mathrm{ref}}$ [Hz]')
    a2.set_ylabel('反馈量最大单步跳变 [°]')
    a2.legend(loc='upper right', fontsize=6.5)
    panel(a2, '(b) 平滑', x=-0.16)
    fig.tight_layout(w_pad=1.0)
    save(fig, 'f14_rate')


FIGS.update({f.__name__.split('_')[0]: f for f in (
    f01_route, f02_system, f03_aliasing, f04_gait, f05_budget, f06_bvsc, f07_outage_mech,
    f08_plane, f09_window, f10_nominal, f11_outage, f12_speed, f13_pulse, f14_rate)})


if __name__ == '__main__':
    for name in (sys.argv[1:] or sorted(FIGS)):
        FIGS[name]()

