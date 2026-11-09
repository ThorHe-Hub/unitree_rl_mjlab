"""传感器劣化与信号调理台。play.py 只调用它，实验旋钮全在这里。

放项目根目录而不是 src/ 下：src/tasks/__init__.py 会拉整个 mjlab 进来，
放在这里离线分析脚本才能直接 import，也才能在没有 GPU 的机器上自检。

为什么要有常值零偏这一档（这是整套实验重新设计的原因）：
    白噪声积分成航向 = σ·√(dt·T)，零均值，任何无偏因果滤波器都去不掉直流分量。
    常值零偏积分成航向 = bias·T，线性增长。
    16 秒时两者相等的门槛只要 0.0053 rad/s = 0.30 °/s，
    而消费级 MEMS 陀螺开机零偏重复性典型 0.5~2 °/s。
    也就是说：真机上零偏比我们之前注的 8.6 °/s 白噪声还重要几倍。

为什么 KF/AKF 去不掉零偏：
    测量 z = ω_真 + b，一个方程两个未知数，单陀螺下零偏不可观测。
    KF/AKF 是测量的无偏估计器，常值零偏原样穿过。要去掉它必须引入外部约束：
    绝对航向参考（磁力计/视觉），或者对 ω_真 的先验。BiasEstimator 用的是后者。
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from heading_fusion import HeadingFusion


@dataclass(frozen=True)
class SensorRigCfg:
    """命令行展开成 --rig.noise-std / --rig.gyro-bias / --rig.filt / ..."""

    noise_std: float = 0.15
    """注入偏航角速度的高斯白噪声标准差 [rad/s]。零均值，积分成航向是 √t 随机游走。"""

    gyro_bias: float = 0.0
    # ponytail: 训练侧需要的是分布（每回合重采），那个走 event 不走这里
    """固定注入的陀螺零偏 [rad/s]；同一配置的各回合使用相同值。

    即开机零偏 ε_b（逐次启动常值）。器件量级见 sensor_profiles.py，
    例如 BMI088 的 ±1 °/s。"""

    bias_rrw: float = 0.0
    """注入的零偏随机游走强度 [(rad/s)^2 / 控制步]，即回合内零偏慢变 ε_r。

    与 fusion_q_bias 分开：那个是滤波器假设的过程噪声，这个是真值。
    取 0（默认）时 ε_r 恒为 0，且完全不抽随机数，行为逐位退化为旧模型。
    器件量级见 sensor_profiles.py（手册未直接给出，属粗略假设）。"""

    filt: Literal["akf", "kf", "dirty", "none", "lpf", "biasest", "oracle", "fusion"] = "akf"
    """提供给策略及独立积分器的角速度信号；PID 航向输入由 --feedback-mode 选择。
    none    = 干净真值，不加噪不滤（基线）
    dirty   = 加噪不滤
    kf/akf  = 现有两个卡尔曼滤波器（注意：它们去不掉常值零偏）
    lpf     = 一阶低通，带宽扫描用
    biasest = 以转向指令为先验的零偏估计
    oracle  = 直接减掉注入的真零偏，作为理想零偏补偿对照
    fusion  = 陀螺 + 绝对航向参考的两状态融合，用估到的零偏修正陀螺。
              同时提供航向后验估计，供 fusion_heading 反馈模式使用。"""

    lpf_hz: float = 0.0
    """一阶低通截止频率 [Hz]，仅 filt="lpf" 时生效。0 = 直通。
    带宽扫描用：真实 ωz 有 59% 功率在 3 Hz 以上，滤掉的是策略要用的信号。"""

    abs_sigma: float = 0.05
    """绝对航向观测的噪声标准差 [rad]，仅 filt="fusion" 时生效。
    仿真参考为相对航向真值叠加独立高斯噪声，0.05 rad ≈ 2.9°。"""

    abs_hz: float = 1.0
    """绝对航向观测的更新率 [Hz]，仅 filt="fusion" 时生效。"""

    abs_dropout: float = 0.0
    """绝对航向观测的丢包率，仅 filt="fusion" 时生效。"""

    abs_outage_start_s: float = 0.0
    """Start of a scheduled absolute-reference outage, seconds from episode start."""

    abs_outage_duration_s: float = 0.0
    """Outage duration; zero disables it. Samples are still drawn to preserve RNG alignment."""

    fusion_q_bias: float = 1e-8
    """Fusion bias process-noise variance per control step, (rad/s)^2."""

    fusion_sigma_bias_prior: float = 0.05
    """Initial fusion bias standard deviation, rad/s."""

    gyro_signal: Literal["point", "interval"] = "point"
    """航向估计器（fusion 预测与独立积分）用的陀螺信号；策略观测一律是 point。

    point：50 Hz 控制步末端的点采样 ω_z（历史行为）。
    interval：相邻两步根链接姿态反推的区间平均角速度 z 分量，相当于在控制周期内
    积分的 delta-angle 陀螺。两者加同一份零偏与白噪声，只换真值部分。
    point 采样与步态锁相会产生直流偏置（见 unitree_heading_results/gait_dev_20260928）。
    interval 只对 filt="fusion" 有定义。"""

    bias_tau: float = 10.0
    """零偏估计的时间常数 [s]，仅 filt="biasest" 时生效。
    核心取舍旋钮：小 = 收敛快但会把机体真实漂移一起当零偏吃掉，PID 就瞎了；大 = 反之。
    估计器自身噪声 σ_ema = σ·√(a/(2−a))，a=dt/τ：τ=5s 时 0.0067，τ=10s 时 0.0048 rad/s。"""


class BiasEstimator:
    """用指令作参考估计陀螺零偏。

    单陀螺下零偏不可观测，必须外部约束。这里的约束是"指令偏航为 0 时机体不该在转"，
    于是测量的长期均值就是零偏。代价是它分不清零偏和真实漂移。
    ponytail: 假设指令偏航恒为 0，只在 --pin-cmd 下成立；
              要支持转向指令，把 cmd 传进 update 即可。
    """

    WARM = 50   # 冷启动等效样本数（1 秒）。不设它的话 k=1 时 a=1，
                # 第一个样本会被全额当成零偏吸收，回合开头真实转动直接归零。
                #
                # ⚠ 与 tau_s 的交互（扫 τ 之前必读）：滑动平均段一直持续到
                #   1/(k+WARM) 降到 dt/τ 以下，即 k = τ/dt − WARM。
                #   dt=0.02 时 τ=60s 的交叉点是第 2950 步——若回合只有 3000 步，
                #   则几乎全程都是滑动平均，τ 这个旋钮是空的。
                #   实测 τ=10/30/60/120 四档结果统计上不可区分，正是这个原因。
                #   要真正扫 τ，回合长度必须远大于 τ/dt，或临时把 WARM 关掉。
                #   （注：滑动平均本身是常值零偏在白噪声下的最大似然估计，
                #     所以 τ≫回合长度这一档实际测的是"最优零偏估计器"。）

    def __init__(self, tau_s=10.0, dt=0.02):
        self.a = dt / max(tau_s, dt)
        self.b = 0.0
        self.k = 0

    def update(self, z, cmd=0.0):
        self.k += 1
        a = max(self.a, 1.0 / (self.k + self.WARM))  # 冷启动走滑动平均，收敛后退化成 EMA
        self.b += a * ((z - cmd) - self.b)
        return z - self.b

    def reset(self):
        self.b = 0.0
        self.k = 0


class SensorRig:
    MAX_REF_RATE = 5.0   # rad/s，绝对参考的合理转速上限（PID 饱和在 1.6，步态再加 0.6）

    def __init__(self, cfg: SensorRigCfg, kf_update=None, akf_update=None,
                 dt: float = 0.02, rng=None, reference_rng=None, dropout_rng=None,
                 bias_rng=None):
        """kf_update / akf_update 是 (z)->估计值 的可调用对象。
        传 None 时该路直通——离线自检不需要真滤波器也能跑。"""
        self.cfg = cfg
        self.dt = dt
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        if not all(np.isfinite(v) for v in (cfg.noise_std, cfg.gyro_bias, cfg.bias_rrw,
                                            cfg.abs_sigma, cfg.abs_hz, cfg.abs_dropout,
                                            cfg.bias_tau, cfg.lpf_hz, cfg.fusion_q_bias,
                                            cfg.fusion_sigma_bias_prior,
                                            cfg.abs_outage_start_s, cfg.abs_outage_duration_s)):
            raise ValueError("Sensor parameters must be finite")
        if cfg.bias_rrw < 0:
            raise ValueError("Bias random-walk variance must be nonnegative")
        if min(cfg.noise_std, cfg.abs_sigma, cfg.abs_hz, cfg.lpf_hz) < 0 or not 0 <= cfg.abs_dropout <= 1:
            raise ValueError("Noise/rate must be nonnegative and dropout must be in [0, 1]")
        if cfg.fusion_q_bias < 0 or cfg.fusion_sigma_bias_prior <= 0:
            raise ValueError("Fusion process variance must be nonnegative and prior SD positive")
        if cfg.abs_hz > 1.0 / dt:
            raise ValueError("Reference rate cannot exceed the control sample rate")
        if cfg.gyro_signal not in ("point", "interval"):
            raise ValueError("gyro_signal must be 'point' or 'interval'")
        if cfg.gyro_signal == "interval" and cfg.filt != "fusion":
            raise ValueError("gyro_signal='interval' is only defined for filt='fusion'")
        if min(cfg.abs_outage_start_s, cfg.abs_outage_duration_s) < 0:
            raise ValueError("Outage start and duration must be nonnegative")
        for value in (cfg.abs_outage_start_s, cfg.abs_outage_duration_s):
            if not np.isclose(round(value / dt) * dt, value, rtol=0, atol=1e-9):
                raise ValueError("Outage times must align with the control sample period")
        self._outage_start = int(round(cfg.abs_outage_start_s / dt))
        self._outage_end = self._outage_start + int(round(cfg.abs_outage_duration_s / dt))
        self._kf = kf_update
        self._akf = akf_update
        self.rng = rng if rng is not None else np.random.default_rng()
        # None retains the historical shared-generator call sequence for replay.
        # New experiments supply distinct seeded streams so reference settings do
        # not change the gyro noise realization.
        self.reference_rng = reference_rng if reference_rng is not None else self.rng
        self.dropout_rng = dropout_rng if dropout_rng is not None else self.rng
        # 零偏随机游走必须走独立流：若与 self.rng 共用，一旦 bias_rrw > 0
        # 就会多抽随机数、扰动陀螺噪声序列，破坏"换参数不换噪声实现"的可比性。
        #
        # 只在**同时传入了 rng**（说明调用方关心流本身）时才报错。像
        # play.py 里读 effective_reference_hz 的那种一次性构造不传 rng，
        # 拿到的是全新生成器，共用不会污染任何被记录的东西。
        if rng is not None and bias_rng is None and cfg.bias_rrw > 0:
            raise ValueError(
                "bias_rrw > 0 requires a distinct bias_rng so the gyro stream "
                "stays unperturbed")
        self.bias_rng = bias_rng if bias_rng is not None else self.rng
        self._eps_r = 0.0
        self.alpha = 1.0 - np.exp(-2.0 * np.pi * cfg.lpf_hz * dt) if cfg.lpf_hz > 0 else 1.0
        self._lp = None
        self._be = BiasEstimator(cfg.bias_tau, dt)
        # Both feedback modes keep the same bias-corrected actor input. The
        # caller can additionally use fusion_heading as PID feedback.
        self._hf = HeadingFusion(dt, cfg.noise_std, cfg.fusion_q_bias, cfg.abs_sigma,
            sigma_bias_prior=cfg.fusion_sigma_bias_prior) \
            if cfg.filt == "fusion" else None
        self._abs_step = int(round(1.0 / cfg.abs_hz / dt)) if cfg.abs_hz > 0 else 0
        self._ref_prev = None
        self._ref_acc = 0.0
        self._k = 0
        self.reference_updated = False
        self.reference_rejected = False
        self.reference_update_count = 0
        self.reference_reject_count = 0

    def reset(self):
        """每回合开头调一次。滤波器自身的状态由 play.py 那边管。"""
        self._lp = None
        self._be.reset()
        if self._hf is not None:
            self._hf = HeadingFusion(self.dt, self.cfg.noise_std, self.cfg.fusion_q_bias, self.cfg.abs_sigma,
                sigma_bias_prior=self.cfg.fusion_sigma_bias_prior)
        self._ref_prev = None
        self._ref_acc = 0.0
        self._k = 0
        self._eps_r = 0.0
        self.reference_updated = False
        self.reference_rejected = False
        self.reference_update_count = 0
        self.reference_reject_count = 0

    def corrupt(self, raw: float) -> float:
        """真值 -> 带噪观测（Allan 三分法）：

            z = raw + ε_b + ε_r + w_g
            ε_b = gyro_bias   每回合常值（开机零偏重复性）
            ε_r = 零偏随机游走，由 bias_rrw 驱动（速率随机游走）
            w_g = 白噪声，由 noise_std 驱动（角度随机游走）

        bias_rrw = 0 时 ε_r 恒为 0，且**不抽任何随机数**，
        保证 self.rng（陀螺流）不受扰动，逐位退化为旧行为。

        注意 ε_r 的抽取用 self.bias_rng 而非 self.rng：两条流互不干扰，
        换 bias_rrw 不会改变陀螺噪声的实现，反之亦然。
        """
        if self.cfg.bias_rrw > 0.0:
            self._eps_r += self.bias_rng.normal(0.0, np.sqrt(self.cfg.bias_rrw))
        # 保存本步白噪声，interval 模式要把同一份噪声加到区间平均角速度上
        self._last_noise = self.rng.normal(0.0, self.cfg.noise_std)
        return (raw + self.cfg.gyro_bias + self._eps_r + self._last_noise)

    @property
    def injected_bias(self) -> float:
        """当前实际注入的总零偏 [rad/s] = ε_b + ε_r。

        只有 bias_rrw = 0 时它才等于 gyro_bias。oracle 档与离线核验
        都应该用它，而不是 cfg.gyro_bias。
        """
        return self.cfg.gyro_bias + self._eps_r

    def lowpass(self, z: float) -> float:
        self._lp = z if self._lp is None else self._lp + self.alpha * (z - self._lp)
        return self._lp

    def _fusion(self, dirty: float, ref_psi, dirty_est=None):
        """陀螺 + 绝对航向参考。返回用估到的零偏修正后的角速度（给策略观测）。

        dirty_est 是给估计器的陀螺读数（interval 模式下与 dirty 不同）；
        扣零偏后的估计器侧角速度存在 self.used_feedback，供独立积分使用。

        ref_psi 是真值航向（弧度，缠绕的）。逐步累加缠绕差得到相对起点的航向，
        避免跨 ±π 时跳变；绝对观测按 abs_hz 采样、加噪、按 abs_dropout 丢包。
        """
        dirty_est = dirty if dirty_est is None else dirty_est
        self._hf.predict(dirty_est)
        if ref_psi is not None:
            if self._ref_prev is None:
                self._ref_prev = ref_psi
            d = float(np.arctan2(np.sin(ref_psi - self._ref_prev),
                                 np.cos(ref_psi - self._ref_prev)))
            # 一步转不过 MAX_REF_RATE：超了就是传送（回合边界 / 摔倒重生），
            # 不是转动。放进去的话融合器会忠实地把机器人转到那个假方向去。
            if abs(d) <= self.MAX_REF_RATE * self.dt:
                self._ref_acc += d
            else:
                self.reference_rejected = True
                self.reference_reject_count += 1
            self._ref_prev = ref_psi
            if self._abs_step and self._k % self._abs_step == 0:
                if self.dropout_rng.random() >= self.cfg.abs_dropout:
                    measurement = self._ref_acc + self.reference_rng.normal(0, self.cfg.abs_sigma)
                    # Blocking assimilation must not shift later reference/noise draws.
                    # Relative truth tracking here only supplies the synthetic sensor;
                    # it is never passed to the estimator during the outage.
                    if not self._outage_start <= self._k < self._outage_end:
                        self._hf.update(measurement)
                        self.reference_updated = True
                        self.reference_update_count += 1
        self._k += 1
        self.used_feedback = dirty_est - self._hf.bias
        return dirty - self._hf.bias

    def step(self, raw: float, cmd: float = 0.0, ref_psi=None, raw_interval=None):
        """一步。返回 (dirty, kf, akf, used)。

        三路滤波器每步都跑、都记，跟 filt 选谁无关——这样 trace 的 18 列语义
        在所有实验条件下保持可比，事后还能离线重放。
        used 是回写到策略观测的那一路；self.used_feedback 是驱动 PID 独立积分的那一路。
        point 模式下两者相同。raw_interval 是区间平均角速度真值，
        interval 模式下必须提供（首步没有上一姿态时由调用方用 raw 代替）。
        """
        if not np.isfinite(raw) or (ref_psi is not None and not np.isfinite(ref_psi)):
            raise ValueError("Non-finite gyro or reference sample")
        interval = self.cfg.gyro_signal == "interval"
        if interval and (raw_interval is None or not np.isfinite(raw_interval)):
            raise ValueError("gyro_signal='interval' requires a finite raw_interval")
        self.reference_updated = False
        self.reference_rejected = False
        dirty = self.corrupt(raw)
        # 同一份零偏与白噪声，只换真值部分；表达式与 corrupt 同序
        dirty_est = (raw_interval + self.cfg.gyro_bias + self._eps_r + self._last_noise) \
            if interval else dirty
        kf = self._kf(dirty) if self._kf is not None else dirty
        akf = self._akf(dirty) if self._akf is not None else dirty
        lp = self.lowpass(dirty)
        be = self._be.update(dirty, cmd)
        fus = self._fusion(dirty, ref_psi, dirty_est) if self._hf is not None else dirty
        used = {"akf": akf, "kf": kf, "dirty": dirty, "none": raw, "lpf": lp,
                "biasest": be, "oracle": dirty - self.injected_bias,
                "fusion": fus}[self.cfg.filt]
        if self.cfg.filt != "fusion":
            self.used_feedback = used
        return dirty, kf, akf, used

    @property
    def bias_est(self) -> float:
        """当前估到的零偏 [rad/s]，落盘用——可以直接跟真值比。"""
        return self._be.b

    @property
    def fusion_heading(self):
        return self._hf.heading if self._hf is not None else None

    @property
    def fusion_bias(self):
        return self._hf.bias if self._hf is not None else None

    @property
    def effective_reference_hz(self):
        return 1.0 / (self._abs_step * self.dt) if self._abs_step else 0.0


def _demo():
    dt, N = 0.02, 4000
    rng = np.random.default_rng(0)

    # 1. 加噪：均值应等于零偏，标准差应等于 noise_std
    rig = SensorRig(SensorRigCfg(noise_std=0.15, gyro_bias=0.02, filt="dirty"), dt=dt, rng=rng)
    d = np.array([rig.step(0.0)[0] for _ in range(N)])
    assert abs(d.mean() - 0.02) < 0.01, f"零偏没对上: {d.mean():.4f}"
    assert abs(d.std() - 0.15) < 0.01, f"白噪声 std 没对上: {d.std():.4f}"

    # 2. 零偏超过白噪声漂移的门槛
    cross = 0.15 * np.sqrt(dt / 16.0)
    assert abs(cross - 0.0053) < 2e-4, f"门槛算错: {cross:.5f}"

    # 3. 任何无偏平滑器都去不掉常值零偏（DC 增益为 1）
    ema = {"v": 0.0}

    def _passthrough_kf(z):
        ema["v"] += 0.2 * (z - ema["v"])       # 重平滑，但 DC 增益仍是 1
        return ema["v"]

    rig = SensorRig(SensorRigCfg(noise_std=0.15, gyro_bias=0.02, filt="kf"),
                    kf_update=_passthrough_kf, dt=dt, rng=np.random.default_rng(3))
    out = np.array([rig.step(0.0)[3] for _ in range(N)])
    assert abs(out[N // 2:].mean() - 0.02) < 0.01, "平滑器居然把零偏去掉了？检查测试"

    # 4. BiasEstimator 能去掉。容差由估计器自身噪声算，不是拍的：σ_ema = σ·√(a/(2−a))
    rig = SensorRig(SensorRigCfg(noise_std=0.15, gyro_bias=0.02, filt="biasest", bias_tau=10.0),
                    dt=dt, rng=np.random.default_rng(4))
    out, bs = [], []
    for _ in range(N):
        out.append(rig.step(0.0)[3])
        bs.append(rig.bias_est)
    out, bs = np.array(out), np.array(bs)
    a_ema = dt / 10.0
    sig_ema = 0.15 * np.sqrt(a_ema / (2 - a_ema))
    warm = N // 4
    assert abs(out[warm:].mean()) < 0.005, f"零偏没去干净: {out[warm:].mean():.4f}"
    assert abs(bs[warm:].mean() - 0.02) < 3 * sig_ema, \
        f"零偏估计均值不对: {bs[warm:].mean():.4f}（σ_ema={sig_ema:.4f}）"

    # 4b. 可估性的界 = 门槛：σ·√(dt/T) 两边是同一个式子，
    #     所以零偏可估当且仅当它主导航向漂移。这条界是理论结果，不是实现细节。
    for T in (16.0, 60.0):
        assert abs(0.15 * np.sqrt(dt / T) - 0.15 * np.sqrt(dt * T) / T) < 1e-12

    # 5. 代价：真实转动也会被逐渐吃掉。但回合开头不该被立刻抹平（冷启动的 WARM 保证）。
    rig = SensorRig(SensorRigCfg(noise_std=0.0, gyro_bias=0.0, filt="biasest", bias_tau=10.0),
                    dt=dt)
    out = np.array([rig.step(0.1)[3] for _ in range(N)])   # 机体真的在以 0.1 rad/s 转
    assert out[0] > 0.09, f"起始时刻不该被抹掉: {out[0]:.4f}"
    assert abs(out[warm:].mean()) < 0.01, "真实转动没被吃掉——那 biasest 的取舍就不存在了"

    # 6. oracle 给出上界
    rig = SensorRig(SensorRigCfg(noise_std=0.0, gyro_bias=0.02, filt="oracle"), dt=dt)
    assert abs(rig.step(0.0)[3]) < 1e-12, "oracle 应该把零偏减干净"

    # 7. 低通：截止频率处 -3dB
    for fc in (1.0, 5.0):
        rig = SensorRig(SensorRigCfg(noise_std=0, gyro_bias=0, filt="lpf", lpf_hz=fc), dt=dt)
        t = np.arange(N) * dt
        x = np.sin(2 * np.pi * fc * t)
        y = np.array([rig.step(v)[3] for v in x])
        w = slice(N // 2, None)
        gain = np.ptp(y[w]) / np.ptp(x[w])
        assert 0.62 < gain < 0.79, f"{fc} Hz 处增益 {gain:.3f}，不在 -3dB 附近"

    # 8. fusion：在不知道真零偏的前提下，用带噪的绝对参考把它估出来并修正陀螺
    bias8 = 0.02
    rig = SensorRig(SensorRigCfg(noise_std=0.15, gyro_bias=bias8, filt="fusion",
                                 abs_sigma=0.05, abs_hz=1.0), dt=dt,
                    rng=np.random.default_rng(7))
    true_rate = 0.30 * np.sin(2 * np.pi * 1.688 * np.arange(N) * dt)   # 零均值步态振荡
    psi = 0.0
    used8 = np.empty(N)
    for i in range(N):
        psi += true_rate[i] * dt
        used8[i] = rig.step(true_rate[i], ref_psi=psi)[3]
    w8 = N // 2
    resid = (used8 - true_rate)[w8:]
    # 残余 = 零偏估计误差 + 白噪声的样本均值。后者的标准误是 σ/√n，容差按它给，
    # 不是拍一个数：σ=0.15、n=2000 时 SE≈0.0034，取 3 倍。
    tol8 = 3 * 0.15 / np.sqrt(len(resid))
    assert abs(resid.mean()) < tol8, f"fusion 没把零偏修掉: 残余 {resid.mean():+.4f} > {tol8:.4f}"
    # 真正要测的是零偏本身估得准不准（这一项不含噪声均值）
    assert abs(rig._hf.bias - bias8) < 0.1 * bias8, \
        f"零偏估计不准: {rig._hf.bias:.5f} vs {bias8}"

    # 9. 参考里的传送跳变（回合边界重抽初始朝向）必须完全不影响结果：这条曾让
    #    fusion 在闭环里把机器人转飞 5 rad（实测 4.973 vs 不滤 1.088）。
    #    判据用差分——同种子跑两遍，有无传送的输出逐位相同，不用猜容差。
    def _run(tp):
        r = SensorRig(SensorRigCfg(noise_std=0.15, gyro_bias=0.02, filt="fusion"),
                      dt=0.02, rng=np.random.default_rng(9))
        return [r.step(0.0, ref_psi=(tp if i >= 100 else 0.0))[3] for i in range(300)], r
    out0, rig0 = _run(0.0)
    out1, rig1 = _run(2.15)
    assert out0 == out1, "传送跳变改变了输出"
    assert rig0._ref_acc == rig1._ref_acc == 0.0, \
        f"传送跳变漏进参考: {rig1._ref_acc:+.4f}"

    print("自检通过 9 项：加噪 / 零偏门槛 / 平滑器去不掉零偏 / biasest去得掉 / "
          "biasest也吃真实漂移 / oracle / 低通 / fusion估零偏 / 挡传送")
    print(f"  16 秒窗口：零偏门槛 = 可分辨下限 = {cross:.5f} rad/s = {np.degrees(cross):.2f} °/s")
    print(f"  60 秒窗口：{0.15*np.sqrt(dt/60):.5f} rad/s = {np.degrees(0.15*np.sqrt(dt/60)):.2f} °/s"
          f"   <- 所以扫描用 --batch-steps 3000")
    print(f"  τ=10s 时零偏估计自身噪声 σ_ema = {sig_ema:.5f} rad/s")


if __name__ == "__main__":
    _demo()
