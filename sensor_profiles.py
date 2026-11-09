"""陀螺器件档案：把手册给出的噪声指标换算成仿真单位。

用途是给 sigma_g、b0 提供器件量级的依据，作为敏感性分析的档位；
主实验的结论不依赖某一款具体器件。

误差模型
--------
    z = ω_真 + ε_b + ε_r + w_g
    ε_b  开机零偏（逐次启动常值）      —— 每回合一个常数，对应 gyro_bias
    ε_r  回合内零偏慢变               —— 用随机游走近似，对应 bias_rrw
    w_g  白噪声（角度随机游走）        —— 每步独立，对应 noise_std

单位换算
--------
    sigma_g [rad/s per step] = ARW[°/√h] · (π/180) / 60 / √dt
    gyro_bias [rad/s]        = turn-on[°/s] · π/180
    ARW[°/√h]                = noise density[°/s/√Hz] · 60   （与带宽无关）

关于 bias_rrw 的限制（引用时必须写明）：
手册通常只给零偏不稳定性 B [°/h]，那是 Allan 曲线的平台值，
不是随机游走系数，两者单位也不同。bias_rrw() 用"1 小时内零偏
漂移 1σ = B"做了一个粗略折算，只能算假设，不能写成由手册换算得到。
按这个折算，BMI088 在 60 s 回合内的零偏变化约 1e-6 rad/s，
远小于 ε_b 与估计残差，对实验结果没有可测影响。
"""
from dataclasses import dataclass

import numpy as np

DEG = np.pi / 180.0


@dataclass(frozen=True)
class GyroProfile:
    """一份器件档案。角度量一律用手册单位，换算在属性里做。"""

    name: str
    arw_deg_sqrth: float
    """角度随机游走 [°/√h]。手册常写作 'rate noise density' 或 ARW。"""
    turn_on_bias_deg_s: float
    """开机零偏重复性 [°/s]（手册 'turn-on bias' / 'initial bias'），取 ±值。"""
    bias_instability_deg_h: float
    """零偏不稳定性 [°/h]（手册 'bias instability' / 'in-run bias stability'）。"""
    source: str
    """数值出处。引用到具体手册/文档，便于核对。"""
    note: str = ""
    exact_noise_std: float | None = None
    """直接给定每步噪声标准差 [rad/s]，跳过 ARW 往返换算（压力档用，保证精确等于旧值）。"""

    def noise_std(self, dt: float) -> float:
        """角度随机游走 -> 每控制步的白噪声标准差 [rad/s]。"""
        if self.exact_noise_std is not None:
            return self.exact_noise_std
        return self.arw_deg_sqrth * DEG / np.sqrt(3600.0) / np.sqrt(dt)

    def gyro_bias(self, sign: float = 1.0) -> float:
        """开机常值零偏 [rad/s]。sign 取 ±1 用来扫零偏符号。"""
        return sign * self.turn_on_bias_deg_s * DEG

    def bias_rrw(self, dt: float) -> float:
        """零偏慢变的随机游走强度 [(rad/s)²/step]，粗略折算，见模块说明。

        B [°/h] 换成 rad/s 要除 3600（2 °/h = 9.70e-6 rad/s），
        再按 σ_b(3600 s)² = Q_c·3600 反解 Q_c。
        """
        sigma_1h = self.bias_instability_deg_h * DEG / 3600.0   # rad/s
        q_continuous = sigma_1h ** 2 / 3600.0                   # (rad/s)²/s
        return q_continuous * dt


# ---------------------------------------------------------------------------
# 器件档案
# ---------------------------------------------------------------------------
# ARW 与零偏稳定性取自各手册的典型值；RRW 手册没有给，见模块说明。
PROFILES = {
    "icm42688": GyroProfile(
        name="ICM-42688-P",
        arw_deg_sqrth=0.168,          # 2.8 mdps/√Hz × 60
        turn_on_bias_deg_s=1.0,
        bias_instability_deg_h=2.0,   # 手册未列，取常见第三方实测区间中值
        source="TDK InvenSense ICM-42688-P datasheet / SmartMotion flyer",
        note="zero-bias 与稳定性手册未直接给出，按同类器件取值，引用时须注明。"),
    "adis16470": GyroProfile(
        name="ADIS16470",
        arw_deg_sqrth=0.48,
        turn_on_bias_deg_s=0.5,
        bias_instability_deg_h=8.0,   # 手册 Table 1
        source="Analog Devices ADIS16470 datasheet, Table 1",
        note="手册明确给出 rate noise density 与 bias stability，provenance 最干净。"),
    "bmi088": GyroProfile(
        name="Bosch BMI088",
        arw_deg_sqrth=0.84,           # 0.014 °/s/√Hz × 60
        turn_on_bias_deg_s=1.0,       # 手册 ±1 °/s
        bias_instability_deg_h=2.0,   # 手册 '< 2 °/h'
        source="Bosch BMI088 datasheet BST-DHW-FL025-01",
        note="开机零偏 ±1 °/s 与 in-run 稳定性 < 2 °/h 均来自手册。"),
}

# 压力档直接给定旧值，避免 ARW 往返换算的浮点误差（0.15 会变成 0.14999999999999997）。
# arw_deg_sqrth 只作说明：dt = 0.02 时的等效 ARW。
STRESS = GyroProfile(
    name="stress-0.15",
    arw_deg_sqrth=0.15 * np.sqrt(0.02) / DEG * 60.0,
    turn_on_bias_deg_s=0.02 / DEG,
    bias_instability_deg_h=0.0,       # 不注入随机游走
    source="论文现状参数（无器件依据），保留作压力档对照",
    note="比典型 MEMS 大 2~3 个数量级，仅作压力档。",
    exact_noise_std=0.15)

ALL_PROFILES = dict(PROFILES)
ALL_PROFILES["stress"] = STRESS


def resolve(name: str) -> GyroProfile:
    if name not in ALL_PROFILES:
        raise KeyError("Unknown gyro profile %r; available: %s"
                       % (name, ", ".join(sorted(ALL_PROFILES))))
    return ALL_PROFILES[name]


def describe(dt: float = 0.02) -> str:
    """打印各档位的仿真单位数值，供论文表格与人工核对使用。"""
    lines = ["%-14s %14s %14s %16s" % ("档位", "sigma_g[rad/s]", "eps_b[rad/s]", "bias_rrw[(rad/s)²/step]"),
             "-" * 62]
    for key in ("icm42688", "adis16470", "bmi088"):
        p = PROFILES[key]
        lines.append("%-14s %14.3e %14.3e %16.3e"
                     % (p.name, p.noise_std(dt), p.gyro_bias(), p.bias_rrw(dt)))
    lines.append("%-14s %14.3e %14.3e %16s"
                 % (STRESS.name, STRESS.noise_std(dt), STRESS.gyro_bias(), "0 (纯常值)"))
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())
