"""Allan 三分法零偏模型（sensor_rig.bias_rrw）的行为测试。

三条最要紧的性质：
  1. bias_rrw = 0 时逐位退化为旧模型（这是所有历史批次可比性的前提）
  2. bias_rrw 的随机游走只走 bias_rng，**不扰动陀螺流**
  3. 未提供独立 bias_rng 却启用 bias_rrw 时必须报错，而不是静默腐蚀噪声流
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest

import numpy as np
from sensor_rig import SensorRig, SensorRigCfg
from experiment_record import episode_seed_metadata


def make(filt="dirty", seed=17, **cfg_kwargs):
    seeds = episode_seed_metadata(seed, 0)
    return SensorRig(SensorRigCfg(filt=filt, **cfg_kwargs),
                     rng=np.random.default_rng(seeds["seed_gyro"]),
                     reference_rng=np.random.default_rng(seeds["seed_reference"]),
                     dropout_rng=np.random.default_rng(seeds["seed_dropout"]),
                     bias_rng=np.random.default_rng(seeds["seed_bias"]))


def feed(rig, n=500, raw=0.3):
    return np.array([rig.step(raw)[0] for _ in range(n)])


class AllanBiasModelTest(unittest.TestCase):

    def test_stream_names_only_appended(self):
        """新增流必须追加在末尾，否则会改变既有流的种子。"""
        base = episode_seed_metadata(24000, 0)
        for name in ("env", "gyro", "reference", "dropout", "vx"):
            self.assertIn("seed_" + name, base)
        # 这五个的取值由 stream_id = enumerate 下标决定，与名字无关；
        # 只要 'bias' 排在最后，它们就不该变。
        self.assertIn("seed_bias", base)
        self.assertEqual(len({base["seed_" + n] for n in
                              ("env", "gyro", "reference", "dropout", "vx", "bias")}), 6)

    def test_zero_rrw_reproduces_legacy_bitwise(self):
        """bias_rrw = 0 时，与不带该字段的等价配置逐位一致。"""
        a = make(noise_std=0.15, gyro_bias=0.02)
        b = make(noise_std=0.15, gyro_bias=0.02, bias_rrw=0.0)
        np.testing.assert_array_equal(feed(a), feed(b))

    def test_zero_rrw_leaves_bias_constant(self):
        rig = make(noise_std=0.15, gyro_bias=0.02, bias_rrw=0.0)
        for _ in range(200):
            rig.step(0.0)
            self.assertEqual(rig.injected_bias, 0.02)

    def test_rrw_does_not_perturb_gyro_stream(self):
        """核心性质：换 bias_rrw 不能改变陀螺白噪声流。

        直接比较两个生成器的内部状态最干净——比数值比较更严格，
        也不受浮点结合序影响（(raw+eps+w)-eps 与 raw+w 在 1e-17 上不等，
        那是舍入不是流扰动）。
        """
        base = make(noise_std=0.15, gyro_bias=0.02)
        drift = make(noise_std=0.15, gyro_bias=0.02, bias_rrw=1e-12)
        raw = 0.3
        for _ in range(500):
            base.step(raw)
            drift.step(raw)
        np.testing.assert_equal(base.rng.bit_generator.state,
                                drift.rng.bit_generator.state)

    def test_rrw_produces_expected_wander(self):
        """随机游走强度应与解析值相符。

        单次实现的轨迹 std 方差极大（有效自由度很少），所以跑 M 次独立
        实现、取末值，用 M 个末值的 std 去比 σ = √(q·N)。
        """
        n, rrw, m = 1000, 1e-12, 200
        finals = []
        for seed in range(m):
            rig = make(seed=1000 + seed, noise_std=0.0, gyro_bias=0.0,
                       bias_rrw=rrw)
            for _ in range(n):
                rig.step(0.0)
            finals.append(rig.injected_bias)
        measured = float(np.std(finals, ddof=1))
        expected = np.sqrt(rrw * n)          # σ = √(q·N)
        self.assertLess(abs(measured - expected) / expected, 0.15,
                        "measured %.3e vs expected %.3e" % (measured, expected))

    def test_rrw_requires_distinct_stream(self):
        """给了陀螺流却要随机游走、又不给独立流，必须报错而不是静默腐蚀。

        但不传 rng 的一次性构造（如 play.py 读 effective_reference_hz）应当放行：
        它拿到的是全新生成器，共用不会污染任何被记录的东西。
        """
        with self.assertRaises(ValueError):
            SensorRig(SensorRigCfg(filt="dirty", bias_rrw=1e-12),
                      rng=np.random.default_rng(1))
        SensorRig(SensorRigCfg(filt="dirty", bias_rrw=1e-12))   # 不抛
        SensorRig(SensorRigCfg(filt="dirty", bias_rrw=1e-12),
                  rng=np.random.default_rng(1),
                  bias_rng=np.random.default_rng(2))            # 不抛

    def test_negative_rrw_rejected(self):
        with self.assertRaises(ValueError):
            SensorRig(SensorRigCfg(filt="dirty", bias_rrw=-1e-12),
                      rng=np.random.default_rng(1),
                      bias_rng=np.random.default_rng(2))

    def test_oracle_subtracts_total_injected_bias(self):
        """oracle 档必须扣掉 eps_b + eps_r，不能只扣 cfg.gyro_bias。"""
        rig = make(filt="oracle", noise_std=0.0, gyro_bias=0.02, bias_rrw=1e-11)
        for _ in range(300):
            raw = 0.4
            _, _, _, used = rig.step(raw)
            self.assertAlmostEqual(used, raw, places=12)

    def test_interval_gyro_shares_noise_and_leaves_policy_input(self):
        """interval：估计器侧与策略侧用同一份零偏和白噪声，只差真值；point 下 used_feedback == used。"""
        kw = dict(noise_std=0.03, gyro_bias=0.02, abs_sigma=0.05, abs_hz=1.0)
        p, i = make("fusion", **kw), make("fusion", gyro_signal="interval", **kw)
        for k in range(300):
            point, inter = 0.2 * np.sin(k * 0.21), 0.2 * np.sin(k * 0.21) + 0.004
            ref = None if k == 0 else 0.01 * k
            dp = p.step(point, ref_psi=ref, raw_interval=inter)
            di = i.step(point, ref_psi=ref, raw_interval=inter)
            self.assertEqual(dp[0], di[0])                      # 点采样的带噪读数一致
            self.assertEqual(p.used_feedback, dp[3])            # point：两路相同
            # interval：估计器侧读数 = 区间真值 + 同一份零偏与噪声
            self.assertAlmostEqual(i.used_feedback - di[3], inter - point, places=12)
        self.assertNotEqual(p.fusion_bias, i.fusion_bias)
        with self.assertRaises(ValueError):
            make("fusion", gyro_signal="interval", **kw).step(0.1)          # 缺 raw_interval
        with self.assertRaises(ValueError):
            make("dirty", gyro_signal="interval")                           # 只对 fusion 有定义

    def test_reset_clears_random_walk(self):
        rig = make(noise_std=0.0, gyro_bias=0.02, bias_rrw=1e-12)
        for _ in range(200):
            rig.step(0.0)
        self.assertNotAlmostEqual(rig.injected_bias, 0.02)
        rig.reset()
        self.assertEqual(rig.injected_bias, 0.02)


if __name__ == "__main__":
    unittest.main()
