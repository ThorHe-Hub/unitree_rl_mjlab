"""陀螺 + 绝对航向参考的融合估计，以及"参考要多好才够"的定量回答。

背景（为什么必须引入绝对参考）：
    本项目的航向是靠角速度积分得到的，整条链路没有航向的绝对观测。
    单陀螺下零偏不可观测（z = ω_真 + b，一个方程两个未知数），
    而零偏积分成航向是线性漂移，随时长无界增长。
    任何无偏因果滤波器的直流增益都是 1，所以换滤波器改变不了这一点。
    出路只有一条：给航向一个绝对参考（磁力计 / 视觉里程计 / 已知地标）。

本模块干两件事：
    1. HeadingFusion —— 两状态卡尔曼 [航向 ψ, 陀螺零偏 b]。
       预测靠陀螺（高速率），更新靠绝对观测（低速率、带噪、可能丢）。
       绝对观测一进来，零偏就变得可观测了 —— 这才是它的真正价值，
       不只是"多一个传感器"，而是把一个不可观测的量变成可观测的。
    2. 扫描：参考噪声 × 更新率 × 丢失率 -> 航向漂移，
       回答"磁力计要多准、多快，才能把漂移压到 X"。

这部分是纯数值的，不需要仿真器。闭环效果（PID 拿到好航向后实际走得如何）
另需在 mjlab 里验证。
"""

import numpy as np


class HeadingFusion:
    """状态 X = [ψ, b]：航向角、陀螺零偏。

    预测：ψ ← ψ + (ω_meas − b)·dt，b 视为随机游走（真实零偏是慢变的）
    更新：拿到绝对航向观测时做标准卡尔曼更新，H = [1, 0]

    零偏可观测性来自绝对观测：没有它时 P[1,1] 只会随 q_bias 单调增长，
    有了它，ψ 的残差会通过 P 的非对角元反推回 b。
    """

    def __init__(self, dt=0.02, sigma_gyro=0.15, q_bias=1e-8, sigma_abs=0.05,
                 sigma_bias_prior=0.05):
        """
        sigma_gyro: 陀螺白噪声标准差 [rad/s]
        q_bias:     零偏随机游走的过程噪声 [rad^2/s^2 per step]。真实零偏很慢变，
                    取小值；取 0 则假设零偏恒定，收敛后就不再跟随了。
                    ponytail: 这是校准旋钮，换 IMU 要重扫。
        sigma_abs:  绝对航向观测噪声标准差 [rad]
        sigma_bias_prior: 零偏先验标准差 [rad/s]，按 IMU 规格给。
                    这个不能当"完全未知"设成 1.0：那等于允许 57 °/s 的零偏，
                    滤波器会用单个带噪采样推出一个巨大的零偏（增益 P01/S 可达 -33），
                    在稀疏更新下直接发散。0.05 rad/s ≈ 2.9 °/s，宽松覆盖 MEMS 典型范围。
        """
        self.dt = dt
        self.X = np.zeros(2)                  # [psi, bias]
        # 初始航向按定义为 0（PID 启动时把当时的航向锁为目标），所以基本已知；
        # 零偏完全未知。不取 0 是为了数值：P00 太小会让协方差在一步预测后
        # 相关系数逼近 1、矩阵近奇异。1e-4 对应 1σ≈0.01 rad，物理上也说得通。
        self.P = np.diag([1e-4, sigma_bias_prior ** 2])
        self.F = np.array([[1.0, -dt], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.diag([(sigma_gyro * dt) ** 2, q_bias])
        self.R = sigma_abs ** 2

    def predict(self, omega_meas):
        """喂一步陀螺测量（含零偏和噪声）。"""
        self.X[0] = self.X[0] + (omega_meas - self.X[1]) * self.dt
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.X[0]

    def update(self, psi_meas):
        """拿到一次绝对航向观测。角度差要绕回 (-pi, pi]。

        协方差用 Joseph 形式：(I-KH)P(I-KH)ᵀ + KRKᵀ。
        简化式 P-KHP 在 P 病态时会丢正定性并发散——本模块的 P00 初值小、
        P11 初值大，一步预测后相关系数就逼近 1，正是那种病态场景。
        """
        y = np.arctan2(np.sin(psi_meas - self.X[0]), np.cos(psi_meas - self.X[0]))
        S = float((self.H @ self.P @ self.H.T)[0, 0]) + self.R
        K = (self.P @ self.H.T) / S
        self.X = self.X + K.flatten() * y
        IKH = np.eye(2) - K @ self.H
        self.P = IKH @ self.P @ IKH.T + K @ K.T * self.R
        return self.X[0]

    @property
    def heading(self):
        return float(self.X[0])

    @property
    def bias(self):
        return float(self.X[1])


def simulate(bias, T=16.0, dt=0.02, sigma_gyro=0.15, sigma_abs=None,
             rate_hz=0.0, dropout=0.0, q_bias=1e-8, seed=0, true_yaw_rate=None,
             sigma_bias_prior=0.05):
    """跑一遍，返回 (航向误差RMS, 末端误差, 零偏估计误差)。

    主指标用全程 RMS 而不是末端瞬时值：末端值取决于最后一次绝对观测离终点多远，
    低更新率下会出现"更新越少反而越准"的假象。

    sigma_abs=None 或 rate_hz=0 表示没有绝对参考 —— 退化成纯积分。
    true_yaw_rate: 真实角速度序列；默认给一个带步态振荡的直行信号。
    """
    rng = np.random.default_rng(seed)
    N = int(T / dt)
    t = np.arange(N) * dt
    if true_yaw_rate is None:
        # 直行：步态基频 1.688 Hz 的振荡 + 一点低频物理扰动。零均值。
        true_yaw_rate = (0.30 * np.sin(2 * np.pi * 1.688 * t)
                         + 0.10 * np.cos(2 * np.pi * 0.3 * t))
    true_psi = np.cumsum(true_yaw_rate) * dt
    meas = true_yaw_rate + bias + rng.normal(0, sigma_gyro, N)

    f = HeadingFusion(dt, sigma_gyro, q_bias, sigma_abs if sigma_abs else 1.0,
                      sigma_bias_prior)
    step = int(round(1.0 / rate_hz / dt)) if rate_hz > 0 else 0
    err = np.empty(N)
    for i in range(N):
        f.predict(meas[i])
        if step and i % step == 0 and sigma_abs is not None:
            if rng.random() >= dropout:                     # 丢包
                f.update(true_psi[i] + rng.normal(0, sigma_abs))
        err[i] = f.heading - true_psi[i]
    return float(np.sqrt((err ** 2).mean())), abs(err[-1]), abs(f.bias - bias)


def _demo():
    dt, T, bias = 0.02, 16.0, 0.02          # 0.02 rad/s ≈ 1.15 °/s，MEMS 典型

    # 1. 没有绝对参考时，误差就是零偏积分 bias·T，跟滤波器无关
    r_open, e_open, b_open = simulate(bias, T=T, rate_hz=0.0)
    assert abs(e_open - bias * T) < 4 * 0.15 * np.sqrt(dt * T), \
        f"纯积分末端误差应 ≈ bias·T = {bias*T:.3f}，实际 {e_open:.3f}"
    assert abs(r_open - bias * T / np.sqrt(3)) < 0.05, \
        f"线性漂移的 RMS 应 ≈ bias·T/√3 = {bias*T/np.sqrt(3):.3f}，实际 {r_open:.3f}"
    assert b_open > 0.5 * bias, "没有绝对观测时零偏不可观测，不该被估出来"

    # 2. 加上 1 Hz / σ=0.05 rad 的绝对参考，航向误差应降一个量级，零偏也被估出来
    # RMS 的改善幅度小于末端误差：开头几秒滤波器还没把零偏估出来，
    # 那段的误差和纯积分一样，会被平均进 RMS。这是真实代价，不该藏。
    r_fus, e_fus, b_fus = simulate(bias, T=T, sigma_abs=0.05, rate_hz=1.0)
    assert r_fus < r_open / 3, f"融合后没有明显改善: {r_fus:.4f} vs {r_open:.4f}"
    assert b_fus < 0.3 * bias, f"零偏没被估出来: 误差 {b_fus:.4f} / 真值 {bias}"

    # 3. 参考越准越好，且单调
    errs = [np.mean([simulate(bias, T=T, sigma_abs=s, rate_hz=1.0, seed=k)[0]
                     for k in range(20)]) for s in (0.01, 0.05, 0.2)]  # [0] = RMS
    assert errs[0] < errs[1] < errs[2], f"不单调: {errs}"

    # 4. 全丢包 = 没有参考
    r_drop, _, _ = simulate(bias, T=T, sigma_abs=0.05, rate_hz=1.0, dropout=1.0)
    assert abs(r_drop - r_open) < 1e-9, "丢包率 100% 应等价于没有参考"

    print("自检通过 4 项：纯积分=bias·T / 融合有效 / 参考质量单调 / 全丢包退化")
    print(f"  零偏 {bias} rad/s，{T:.0f} 秒：")
    print(f"    无参考          RMS {np.degrees(r_open):5.2f}°  末端 {np.degrees(e_open):5.2f}°"
          f"   零偏估计误差 {b_open:.5f}")
    print(f"    1Hz/σ=0.05 参考 RMS {np.degrees(r_fus):5.2f}°  末端 {np.degrees(e_fus):5.2f}°"
          f"   零偏估计误差 {b_fus:.5f}")


def sweep(T=16.0, bias=0.02, n=40):
    """参考质量 -> 航向漂移。报告要的那张表。"""
    print(f"\n绝对航向参考需要多好？（零偏 {bias} rad/s = {np.degrees(bias):.2f} °/s，"
          f"{T:.0f} 秒，每格 {n} 次平均）")
    e0 = np.mean([simulate(bias, T=T, rate_hz=0.0, seed=k)[0] for k in range(n)])
    print(f"无绝对参考（纯积分）: RMS {np.degrees(e0):.2f}°\n")

    rates = [0.2, 0.5, 1.0, 5.0, 25.0]
    sigmas = [0.01, 0.02, 0.05, 0.10, 0.20]
    print(f"{'参考噪声σ':>10s} " + "".join(f"{r:>9.1f}Hz" for r in rates))
    print("-" * (11 + 11 * len(rates)))
    for s in sigmas:
        row = f"{s:8.2f}rad "
        for r in rates:
            e = np.mean([simulate(bias, T=T, sigma_abs=s, rate_hz=r, seed=k)[0]
                         for k in range(n)])
            row += f"{np.degrees(e):9.2f}° "
        print(row)
    print(f"（单元格 = 全程航向误差 RMS。对照：无参考 {np.degrees(e0):.2f}°）")

    print(f"\n丢包率的影响（σ=0.05 rad，1 Hz）")
    print(f"{'丢包率':>8s} {'航向误差':>12s}")
    for d in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        e = np.mean([simulate(bias, T=T, sigma_abs=0.05, rate_hz=1.0, dropout=d, seed=k)[0]
                     for k in range(n)])
        print(f"{d*100:6.0f}%  {np.degrees(e):10.2f}°")


if __name__ == "__main__":
    _demo()
    sweep()
