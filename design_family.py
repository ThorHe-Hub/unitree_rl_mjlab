"""航向反馈设计族的离线分析核心：按误差来源叠加重放 + 协方差理论预测。

设计族
------
S(τ) = i + LPF_τ(f − i)，其中 f 是融合航向、i 是扣零偏后的陀螺积分。
τ → 0 时 S = f（方案 C），τ → ∞ 时 S = i（方案 B）。
估计器本身由 q_b（零偏过程噪声）决定，所以整个设计空间是 (q_b, τ) 平面。

为什么能按来源拆开
------------------
卡尔曼增益序列 K_k 只取决于 (σg, q_b, σa, 更新时刻)，与数据无关。
固定增益后，估计器、积分器、低通都是线性递推，初值全为 0，
所以输出对输入是**严格线性**的。把陀螺输入拆成

    z = r（机体 ω_z 真值） + ε（注入零偏） + w（白噪声）

把参考观测拆成

    y = ψ_ref（前一步真值航向） + v（参考噪声）

每个来源单独重放一次，四份结果相加必须逐位（到浮点舍入）等于整体重放。
test_design_family.py 把这一点当作硬断言。

前提：参考从未被拒收、丢包率为 0、新息从未跨 ±π（否则 arctan2 绕回非线性）。
load_trace() 会逐条检查这三点。

与闭环的关系
------------
这里冻结的是已记录的物理路径。换 (q_b, τ) 会改变 PID 指令、进而改变路径，
离线结果只回答"在同一路径上，换成这个反馈量会怎样"。结论需闭环抽点确认。
"""
from dataclasses import dataclass
import glob
import math
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

DT = 0.02
SOURCES = ('kin', 'bias', 'gyro', 'ref')
"""误差来源：kin=理想传感器下的残差（步态运动学项 + 参考的一步延迟），
bias=注入零偏 ε_b(+ε_r)，gyro=陀螺白噪声，ref=参考观测噪声。"""


@dataclass
class Trace:
    name: str
    meta: dict
    r: np.ndarray          # 估计器用的陀螺真值部分：raw_yaw（point）或区间平均角速度（interval）
    psi: np.ndarray        # 真值航向，展开并以首步为 0
    eps: np.ndarray        # 注入零偏序列 ε_b + ε_r
    w_unit: np.ndarray     # 陀螺白噪声 / σg（单位方差，按原种子重建）
    v_unit: np.ndarray     # 参考噪声 / σa，只在测量时刻非零
    meas: np.ndarray       # bool，参考采样时刻（不管是否被中断屏蔽）
    upd: np.ndarray        # bool，实际同化的时刻
    recorded: dict         # 原记录的 fusion_heading / fusion_bias / integrated_yaw


def _R(p, t, y):
    """ZYX 欧拉角 -> 旋转矩阵（机体到世界），批量。"""
    cp, sp, ct, st, cy, sy = np.cos(p), np.sin(p), np.cos(t), np.sin(t), np.cos(y), np.sin(y)
    return np.stack([np.stack([cy*ct, cy*st*sp - sy*cp, cy*st*cp + sy*sp], -1),
                     np.stack([sy*ct, sy*st*sp + cy*cp, sy*st*cp - cy*sp], -1),
                     np.stack([-st, ct*sp, ct*cp], -1)], -2)


def interval_rate(phi, th, psi):
    """相邻姿态反推的机体角速度，区间 k→k+1 的平均（理想 delta-angle 陀螺），形状 (n−1, 3)。"""
    Rk = _R(phi, th, psi)
    M = np.einsum('kji,kjl->kil', Rk[:-1], Rk[1:])              # R_kᵀ R_{k+1}
    cos = np.clip((np.trace(M, axis1=1, axis2=2) - 1) / 2, -1, 1)
    ang = np.arccos(cos)
    k = np.stack([M[:, 2, 1] - M[:, 1, 2], M[:, 0, 2] - M[:, 2, 0], M[:, 1, 0] - M[:, 0, 1]], -1)
    scale = np.where(ang < 1e-9, 0.5, ang / (2 * np.sin(np.maximum(ang, 1e-12))))
    return k * scale[:, None] / DT


def load_trace(path, sigma_g_expected=None, abs_hz=None, outage=None, rate_source=None):
    """读一条 fusion 路径，并按 SensorRig 的抽样顺序重建各随机来源。

    abs_hz / outage=(start_s, duration_s) 用于**反事实**分析：在同一物理路径上
    换参考更新率或中断时段。参考噪声仍用原 seed_reference 按采样顺序抽取，
    与 SensorRig 在该设置下会抽到的序列逐位相同（丢包率为 0 时 dropout 流不影响结果）。

    rate_source 选估计器的陀螺真值部分：None = 按记录的 gyro_signal（旧轨迹为 point）；
    'point' = raw_yaw；'interval' = raw_yaw_interval 列，没有该列的 v5 轨迹由记录的
    欧拉角重建（与 play.py 的四元数算法同一区间，数值差在浮点量级）。
    与记录不同即为反事实。给定任一反事实参数时，不再与记录的 reference_updated 对照，recorded 也置空。"""
    with np.load(path, allow_pickle=False) as z:
        a = z['data']
        c = {str(n): i for i, n in enumerate(z['columns'])}
        meta = {k: z[k].item() for k in z.files if k not in ('data', 'columns')}
    if meta.get('filt') != 'fusion':
        raise ValueError('%s is not a fusion trace' % path)
    if float(meta.get('abs_dropout', 0.0)) != 0.0:
        raise ValueError('dropout > 0 breaks the fixed-gain assumption: %s' % path)
    if np.any(a[:, c['reference_rejected']] > 0.5):
        raise ValueError('reference rejection present: %s' % path)
    n = len(a)
    recorded_signal = meta.get('gyro_signal', 'point')
    source = recorded_signal if rate_source is None else rate_source
    if source == 'point':
        raw = a[:, c['raw_yaw']]
    elif source == 'interval':
        if 'raw_yaw_interval' in c:
            raw = a[:, c['raw_yaw_interval']]
        else:                                       # v5：由欧拉角重建，段首用点采样
            raw = a[:, c['raw_yaw']].copy()
            raw[1:] = interval_rate(a[:, c['gt_roll']], a[:, c['gt_pitch']],
                                    np.unwrap(a[:, c['gt_yaw']]))[:, 2]
    else:
        raise ValueError('rate_source must be point or interval')
    psi = np.unwrap(a[:, c['gt_yaw']])
    psi = psi - psi[0]
    sg = float(meta['noise_std'])
    if sigma_g_expected is not None and sg != sigma_g_expected:
        raise ValueError('unexpected sigma_g %r in %s' % (sg, path))

    # 陀螺流：每步一次 normal(0, σg)；按 NumPy 实现等于 σg·standard_normal
    w_unit = np.random.default_rng(meta['seed_gyro']).standard_normal(n)
    # 零偏随机游走（旧轨迹无此键 -> 0）
    rrw = float(meta.get('bias_rrw', 0.0) or 0.0)
    eps = np.full(n, float(meta['gyro_bias']))
    if rrw > 0:
        eps = eps + np.cumsum(np.random.default_rng(meta['seed_bias']).normal(0.0, math.sqrt(rrw), n))

    counterfactual = abs_hz is not None or outage is not None or source != recorded_signal
    hz = float(meta['abs_hz']) if abs_hz is None else float(abs_hz)
    step = int(round(1.0 / hz / DT))
    if not math.isclose(step * DT * hz, 1.0, rel_tol=1e-9):
        raise ValueError('reference period must be a whole number of control steps: %g Hz' % hz)
    k = np.arange(n)
    meas = (k % step == 0) & (k > 0)            # k=0 时 ref_psi 为 None，不采样
    v_unit = np.zeros(n)
    v_unit[meas] = np.random.default_rng(meta['seed_reference']).standard_normal(int(meas.sum()))
    if outage is None:
        outage = (float(meta.get('abs_outage_start_s', 0.0) or 0.0),
                  float(meta.get('abs_outage_duration_s', 0.0) or 0.0))
    s0 = int(round(outage[0] / DT))
    s1 = s0 + int(round(outage[1] / DT))
    upd = meas & ~((k >= s0) & (k < s1))
    if counterfactual:
        recorded = {}
    else:
        if not np.array_equal(upd, a[:, c['reference_updated']] > 0.5):
            raise ValueError('reconstructed update mask disagrees with record: %s' % path)
        recorded = {name: a[:, c[name]].copy()
                    for name in ('fusion_heading', 'fusion_bias', 'integrated_yaw')}
    return Trace(Path(path).parent.name, meta, raw, psi, eps, w_unit, v_unit, meas, upd, recorded)


def load_glob(pattern, **kw):
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise FileNotFoundError(pattern)
    return [load_trace(p, **kw) for p in paths]


# ---------------------------------------------------------------------------
# 固定增益序列
# ---------------------------------------------------------------------------
def gains(upd, sigma_g, q_b, sigma_a, prior=0.05):
    """与 HeadingFusion 完全同一套递推（含 Joseph 形式），返回每步的 (K0, K1)。
    非更新步增益为 0。只依赖更新时刻，不依赖数据。"""
    F = np.array([[1.0, -DT], [0.0, 1.0]])
    Q = np.diag([(sigma_g * DT) ** 2, q_b])
    R = sigma_a ** 2
    P = np.diag([1e-4, prior ** 2])
    n = len(upd)
    K0 = np.zeros(n)
    K1 = np.zeros(n)
    for k in range(n):
        P = F @ P @ F.T + Q
        if upd[k]:
            S = P[0, 0] + R
            K = P[:, 0] / S
            K0[k], K1[k] = K
            IKH = np.array([[1.0 - K[0], 0.0], [-K[1], 1.0]])
            P = IKH @ P @ IKH.T + np.outer(K, K) * R
    return K0, K1


# ---------------------------------------------------------------------------
# 线性重放
# ---------------------------------------------------------------------------
def replay(u, y, upd, K0, K1):
    """u, y: (M, n)。返回 f（融合航向）、b（零偏估计）、i（扣零偏积分）。

    与 SensorRig._fusion + HeadingFeedback.advance 的执行顺序一致：
    预测 -> （若更新）校正 -> 用校正后的零偏得到 used -> 积分。"""
    M, n = u.shape
    f = np.empty((M, n))
    b = np.empty((M, n))
    i = np.empty((M, n))
    x0 = np.zeros(M)
    x1 = np.zeros(M)
    acc = np.zeros(M)
    for k in range(n):
        x0 = x0 + (u[:, k] - x1) * DT
        if upd[k]:
            e = y[:, k] - x0
            x0 = x0 + K0[k] * e
            x1 = x1 + K1[k] * e
        acc = acc + (u[:, k] - x1) * DT
        f[:, k] = x0
        b[:, k] = x1
        i[:, k] = acc
    return f, b, i


def source_inputs(traces, sigma_g, sigma_a):
    """四个来源各自的 (u, y)，形状 (M, n)。真值航向用前一步（参考的一步延迟）。"""
    r = np.stack([t.r for t in traces])
    psi = np.stack([t.psi for t in traces])
    y_true = np.zeros_like(psi)
    y_true[:, 1:] = psi[:, :-1]
    zero = np.zeros_like(r)
    return {
        'kin':  (r, y_true),
        'bias': (np.stack([t.eps for t in traces]), zero),
        'gyro': (sigma_g * np.stack([t.w_unit for t in traces]), zero),
        'ref':  (zero, sigma_a * np.stack([t.v_unit for t in traces])),
    }, psi


def decompose(traces, sigma_g, q_b, sigma_a, prior=0.05):
    """按来源重放。返回 {来源: (f, i)} 与真值 psi。

    kin 来源的 f/i 减去 psi 才是误差；其余来源的 f/i 本身就是误差贡献。"""
    upd = traces[0].upd
    if any(not np.array_equal(t.upd, upd) for t in traces):
        raise ValueError('traces in one call must share the same update mask')
    K0, K1 = gains(upd, sigma_g, q_b, sigma_a, prior)
    inputs, psi = source_inputs(traces, sigma_g, sigma_a)
    parts = {}
    for name in SOURCES:
        u, y = inputs[name]
        f, _, i = replay(u, y, upd, K0, K1)
        if name == 'kin':
            f, i = f - psi, i - psi
        parts[name] = (f, i)
    return parts


def smooth(f, i, tau):
    """S(τ) = i + LPF_τ(f − i)，与 HeadingFeedback 的离散形式一致（初值 0）。"""
    if tau == 0:
        return f.copy()
    if math.isinf(tau):
        return i.copy()
    a = math.exp(-DT / tau)
    return i + lfilter([1.0 - a], [1.0, -a], f - i, axis=-1)


# ---------------------------------------------------------------------------
# 理论：噪声来源的误差方差（精确协方差递推，不需要蒙特卡洛）
# ---------------------------------------------------------------------------
def noise_variance(upd, sigma_g, q_b, sigma_a, taus, prior=0.05):
    """陀螺白噪声与参考噪声各自造成的 S(τ) 误差方差，逐步给出。

    误差状态 x = [e_f, e_b, e_i, e_o]（融合航向、零偏估计、积分、低通偏移），
    输入 n = [w, v]（陀螺噪声、参考噪声，各自单位方差再乘幅值）。一步内：
        预测  e_f += (w − e_b)·dt
        校正  e_f += K0·(v − e_f)，e_b += K1·(v − e_f)       （仅更新步）
        积分  e_i += (w − e_b)·dt                           （用校正后的 e_b）
        低通  e_o  = a·e_o + (1−a)·(e_f − e_i)
    输出  e_S = e_i + e_o。整个过程线性，所以 x_{k} = A_k x_{k-1} + B_k n_k，
    协方差按 P ← A P Aᵀ + B diag(σ²) Bᵀ 精确传播；两个来源独立，分开传播。

    返回 {'gyro': (len(taus), n), 'ref': (len(taus), n)} 的方差数组。
    """
    K0, K1 = gains(upd, sigma_g, q_b, sigma_a, prior)
    n = len(upd)
    out = {'gyro': np.zeros((len(taus), n)), 'ref': np.zeros((len(taus), n))}
    for j, tau in enumerate(taus):
        a = 0.0 if tau == 0 else (1.0 if math.isinf(tau) else math.exp(-DT / tau))
        for src, sigma in (('gyro', sigma_g), ('ref', sigma_a)):
            P = np.zeros((4, 4))
            for k in range(n):
                A, B = _step_matrices(K0[k], K1[k], a, src)
                P = A @ P @ A.T + (sigma ** 2) * np.outer(B, B)
                c = np.array([0.0, 0.0, 1.0, 1.0])       # e_S = e_i + e_o
                if tau == 0:
                    c = np.array([1.0, 0.0, 0.0, 0.0])   # τ=0 时 S 就是 f
                out[src][j, k] = c @ P @ c
    return out


def _step_matrices(k0, k1, a, src):
    """一步的线性映射。把上面 docstring 的四行按顺序写成矩阵乘积。"""
    w = 1.0 if src == 'gyro' else 0.0
    v = 1.0 if src == 'ref' else 0.0
    # 状态 [e_f, e_b, e_i, e_o, n]，最后一维是本步噪声（单位幅值）
    pred = np.eye(5)
    pred[0, 1] = -DT
    pred[0, 4] = w * DT
    corr = np.eye(5)
    corr[0, 0] = 1.0 - k0
    corr[0, 4] = k0 * v
    corr[1, 0] = -k1
    corr[1, 4] = k1 * v
    integ = np.eye(5)
    integ[2, 1] = -DT
    integ[2, 4] = w * DT
    lp = np.eye(5)
    lp[3, :] = 0.0
    lp[3, 3] = a
    lp[3, 0] = 1.0 - a
    lp[3, 2] = -(1.0 - a)
    T = lp @ integ @ corr @ pred
    return T[:4, :4], T[:4, 4]

