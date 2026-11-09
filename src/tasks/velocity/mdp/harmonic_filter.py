"""步态相位锁定的谐波卡尔曼滤波器。

思路：行走时的 yaw rate 不是随机游走，是锁在步态频率上的准周期信号——
双腿交替踏地，基频 = 步态频率，触地冲击在 2 倍频。CA（恒加速度）模型
是个通用平滑先验，跟周期信号根本不匹配，所以要么滞后要么压平幅值。

这里换个状态量：不估"当前值和它的导数"，而是估"各次谐波的幅值 + 直流偏置"。
频率不用估——步态相位本来就在观测里（velocity_env_cfg 的 phase 项）。
幅值是慢变量，所以能重度平滑降噪；而重构出的当前值用的是**当前相位**，
所以结构上零滞后。本质是卡尔曼实现的锁相放大器。

    z = bias + a1·sin(θ) + b1·cos(θ) + a2·sin(2θ) + b2·cos(2θ) + noise

bias 这一项额外送一个礼物：它就是 PID 一直在对抗的那个物理偏航偏置，
可以直接前馈补偿，不必靠积分项去磨。
"""

import numpy as np


class HarmonicKalmanFilter:
    def __init__(self, n_harmonics=2, sigma=0.15, q_bias=5e-4, q_harm=5e-6):
        """
        n_harmonics: 用几次谐波。2 够用（基频=迈步，2倍频=触地冲击）。
        sigma:       测量噪声标准差。这个是知道的，不用在线估。
        q_bias:      偏置的过程噪声。偏置快变（转向指令每 3-8 秒重采），所以大。
        q_harm:      谐波幅值的过程噪声。幅值慢变（地形/步速），所以小两个量级。
                     这俩的时间尺度差得远，别绑在一起调——绑住会白丢一半降噪。
                     ponytail: q_* 是校准旋钮，地形/步速/真机换了都要重扫，
                               二维扫 q_bias x q_harm 看降噪倍数，别只调一个。

        已知转向指令时还能再好 ~12%，不用改这个类，调用侧减掉再加回来即可：
            est = f.update(z - cmd_vyaw, s, c) + cmd_vyaw
        """
        self.n = n_harmonics
        self.dim = 1 + 2 * n_harmonics
        self.X = np.zeros(self.dim)
        self.P = np.eye(self.dim) * 1.0
        self.R = sigma ** 2
        self.Q = np.full(self.dim, q_harm)
        self.Q[0] = q_bias

    def _basis(self, sin_t, cos_t):
        """从步态相位的 sin/cos 直接构造谐波基，不需要绝对时间。"""
        h = np.empty(self.dim)
        h[0] = 1.0
        s, c = sin_t, cos_t
        for k in range(self.n):
            h[1 + 2 * k] = s
            h[2 + 2 * k] = c
            if k + 1 < self.n:  # 倍角公式推下一次谐波
                s, c = 2 * s * c, c * c - s * s
        return h

    def update(self, z, phase_sin, phase_cos):
        """喂一个带噪观测 + 当前步态相位，返回该时刻的干净估计。"""
        # 站立时 phase 被置零（见 mdp.phase 的 stand_mask），基退化成只剩偏置
        moving = abs(phase_sin) + abs(phase_cos) > 1e-6
        H = self._basis(phase_sin, phase_cos) if moving else self._basis(0.0, 0.0)
        if not moving:
            H[1:] = 0.0

        self.P += np.diag(self.Q)          # A = I：幅值是慢变随机游走

        S = H @ self.P @ H + self.R
        K = (self.P @ H) / S
        self.X = self.X + K * (z - H @ self.X)
        self.P = self.P - np.outer(K, H @ self.P)

        return float(H @ self.X)

    @property
    def bias(self):
        """当前估到的直流偏置 [rad/s]——PID 在对抗的那个东西。"""
        return float(self.X[0])


def _demo():
    """自检：幅值随地形起伏、偏置随转向指令跳变——恒幅值信号太容易，不算数。"""
    rng = np.random.default_rng(0)
    N, dt, period, sigma = 3000, 0.02, 0.6, 0.15
    t = np.arange(N) * dt
    th = 2 * np.pi * t / period

    amp1 = 0.30 * (1 + 0.35 * np.sin(2 * np.pi * t / 4.0))   # 幅值慢变
    amp2 = 0.12 * (1 + 0.30 * np.cos(2 * np.pi * t / 2.7))
    bias = np.zeros(N)                                        # 偏置每 3-8 秒跳一次
    k = 0
    while k < N:
        seg = int(rng.uniform(3, 8) / dt)
        bias[k:k + seg] = rng.uniform(-0.4, 0.4)
        k += seg
    truth = bias + amp1 * np.sin(th) + amp2 * np.cos(2 * th)
    noisy = truth + rng.normal(0, sigma, N)

    hf = HarmonicKalmanFilter(sigma=sigma)
    est = np.array([hf.update(z, np.sin(a), np.cos(a)) for z, a in zip(noisy, th)])

    warm = N // 5                                    # 跳过收敛段
    rmse_in = np.sqrt(((noisy - truth)[warm:] ** 2).mean())
    rmse_out = np.sqrt(((est - truth)[warm:] ** 2).mean())
    ratio = rmse_in / rmse_out
    assert ratio > 2.0, f"降噪不足: {ratio:.2f}x（AKF 实测才 1.10x，这里要明显更好）"

    # 零滞后：与真值的互相关峰必须落在 0
    a, b = est[warm:] - est[warm:].mean(), truth[warm:] - truth[warm:].mean()
    lags = [np.dot(a[k:], b[:len(b) - k]) if k else np.dot(a, b) for k in range(11)]
    assert np.argmax(lags) == 0, f"出现滞后 {np.argmax(lags)} 步"

    print(f"自检通过：降噪 {ratio:.2f}x，滞后 0 步（AKF 实测 1.10x / 0 步）")


if __name__ == "__main__":
    _demo()