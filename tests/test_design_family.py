"""design_family 的正确性：与 SensorRig 生产路径逐位一致、叠加恒等、理论方差与蒙特卡洛一致。

不依赖任何已记录轨迹：用 SensorRig + HeadingFeedback 现场生成一条合成路径并写成
与 play.py 相同键名的 npz，再交给 design_family.load_trace。
"""
import math
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import design_family as df                                  # noqa: E402
from experiment_record import episode_seed_metadata          # noqa: E402
from heading_feedback import HeadingFeedback                 # noqa: E402
from sensor_rig import SensorRig, SensorRigCfg               # noqa: E402

COLS = ['raw_yaw', 'gt_yaw', 'reference_updated', 'reference_rejected',
        'fusion_heading', 'fusion_bias', 'integrated_yaw', 'feedback_heading', 'raw_yaw_interval']


def synth_trace(path, n=1200, sg=2e-3, q=1e-10, bias=0.02, outage=(8.0, 6.0), tau=0.5, seed=5, abs_hz=1.0,
                gyro_signal='point'):
    """用生产代码生成一条轨迹。真值航向取正弦转动，r 为其解析导数（无运动学项）。

    interval 模式下点采样另加一个 1.7 Hz 的振荡，模拟与区间平均不同的点采样。"""
    s = episode_seed_metadata(seed, 0)
    cfg = SensorRigCfg(filt='fusion', noise_std=sg, gyro_bias=bias, abs_sigma=0.05, abs_hz=abs_hz,
                       fusion_q_bias=q, abs_outage_start_s=outage[0], abs_outage_duration_s=outage[1],
                       gyro_signal=gyro_signal)
    rig = SensorRig(cfg, dt=df.DT, rng=np.random.default_rng(s['seed_gyro']),
                    reference_rng=np.random.default_rng(s['seed_reference']),
                    dropout_rng=np.random.default_rng(s['seed_dropout']),
                    bias_rng=np.random.default_rng(s['seed_bias']))
    fb = HeadingFeedback('fusion_offset_smoothed', 'fusion', tau)
    t = np.arange(n) * df.DT
    psi = 0.4 * np.sin(0.3 * t)
    r = np.gradient(psi, df.DT)
    point = r + (0.05 * np.sin(2 * np.pi * 1.7 * t) + 0.003 if gyro_signal == 'interval' else 0.0)
    rows = []
    for k in range(n):
        _, _, _, used = rig.step(float(point[k]), ref_psi=None if k == 0 else float(psi[k - 1]),
                                 raw_interval=float(r[k]))
        fb.advance(rig.used_feedback, df.DT, rig.fusion_heading)
        rows.append([point[k], psi[k], rig.reference_updated, rig.reference_rejected,
                     rig._hf.heading, rig._hf.bias, fb.integrated_yaw, fb.value, r[k]])
    meta = dict(filt='fusion', abs_dropout=0.0, noise_std=sg, gyro_bias=bias, abs_hz=abs_hz,
                abs_sigma=0.05, fusion_q_bias=q, abs_outage_start_s=outage[0], gyro_signal=gyro_signal,
                abs_outage_duration_s=outage[1], **{k: v for k, v in s.items() if k.startswith('seed_')})
    np.savez(path, data=np.array(rows, dtype=float), columns=np.array(COLS), **meta)
    return path, np.array(rows)


class DesignFamilyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path, cls.rows = synth_trace(Path(cls.tmp.name) / 't.npz')
        cls.tr = df.load_trace(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_replay_matches_production_and_superposition_is_exact(self):
        m = self.tr.meta
        parts = df.decompose([self.tr], m['noise_std'], m['fusion_q_bias'], m['abs_sigma'])
        f = sum(parts[s][0] for s in df.SOURCES)[0] + self.tr.psi
        i = sum(parts[s][1] for s in df.SOURCES)[0] + self.tr.psi
        self.assertLess(np.abs(f - self.tr.recorded['fusion_heading']).max(), 1e-12)
        self.assertLess(np.abs(i - self.tr.recorded['integrated_yaw']).max(), 1e-12)
        # 设计族：S(τ) 复现 HeadingFeedback 的记录值；两端退化为 f、i
        s = df.smooth(f[None], i[None], 0.5)[0]
        self.assertLess(np.abs(s - self.rows[:, 7]).max(), 1e-12)
        self.assertTrue(np.array_equal(df.smooth(f[None], i[None], 0.0)[0], f))
        self.assertTrue(np.array_equal(df.smooth(f[None], i[None], math.inf)[0], i))

    def test_outage_mask_is_reconstructed(self):
        k = np.arange(len(self.tr.upd))
        inside = (k >= 400) & (k < 700)
        self.assertFalse(self.tr.upd[inside].any())
        self.assertTrue(self.tr.meas[inside].any())          # 采样照常，只是不同化

    def test_noise_variance_matches_monte_carlo(self):
        upd, sg, q, sa = self.tr.upd, 2e-3, 1e-6, 0.05
        taus = [0.0, 2.0, math.inf]
        th = df.noise_variance(upd, sg, q, sa, taus)
        K0, K1 = df.gains(upd, sg, q, sa)
        rng = np.random.default_rng(1)
        M, n = 3000, len(upd)
        zero = np.zeros((M, n))
        y = zero.copy()
        y[:, upd] = sa * rng.standard_normal((M, int(upd.sum())))
        for src, (u, yy) in (('gyro', (sg * rng.standard_normal((M, n)), zero)), ('ref', (zero, y))):
            f, _, i = df.replay(u, yy, upd, K0, K1)
            for j, tau in enumerate(taus):
                mc = (df.smooth(f, i, tau)[:, 300:] ** 2).mean()
                self.assertAlmostEqual(math.sqrt(mc / th[src][j, 300:].mean()), 1.0, delta=0.05,
                                       msg='%s tau=%s' % (src, tau))

    def test_counterfactual_rate_and_outage_match_production(self):
        """1 Hz 轨迹按 abs_hz=2 / 另一中断时段反事实重放，必须逐位等于生产代码在该设置下的记录。"""
        for j, (kw, override) in enumerate(((dict(abs_hz=2.0), dict(abs_hz=2.0)),
                             (dict(outage=(4.0, 10.0)), dict(outage=(4.0, 10.0))),
                             (dict(abs_hz=0.5, outage=(0.0, 0.0)), dict(abs_hz=0.5, outage=(0.0, 0.0))))):
            prod_path, _ = synth_trace(Path(self.tmp.name) / ('p%d.npz' % j), **kw)
            prod = df.load_trace(prod_path)
            cf = df.load_trace(self.path, **override)
            self.assertEqual(cf.recorded, {})
            self.assertTrue(np.array_equal(cf.upd, prod.upd), msg=str(override))
            m = prod.meta
            parts = df.decompose([cf], m['noise_std'], m['fusion_q_bias'], m['abs_sigma'])
            f = sum(parts[s][0] for s in df.SOURCES)[0] + cf.psi
            self.assertLess(np.abs(f - prod.recorded['fusion_heading']).max(), 1e-12, msg=str(override))
        with self.assertRaises(ValueError):
            df.load_trace(self.path, abs_hz=3.0)       # 1/3 s 不是 0.02 s 的整数倍

    def test_interval_gyro_replay_and_counterfactual_switch(self):
        """interval 模式：估计器吃区间平均、策略吃点采样；重放用 raw_yaw_interval，
        另一种信号作反事实时逐位等于生产代码在该模式下的记录。"""
        p_int, rows_int = synth_trace(Path(self.tmp.name) / 'gi.npz', gyro_signal='interval')
        p_pt, _ = synth_trace(Path(self.tmp.name) / 'gp.npz', gyro_signal='point')
        tr = df.load_trace(p_int)                         # 按记录：interval
        self.assertTrue(np.array_equal(tr.r, rows_int[:, 8]))
        m = tr.meta
        parts = df.decompose([tr], m['noise_std'], m['fusion_q_bias'], m['abs_sigma'])
        f = sum(parts[s][0] for s in df.SOURCES)[0] + tr.psi
        i = sum(parts[s][1] for s in df.SOURCES)[0] + tr.psi
        self.assertLess(np.abs(f - tr.recorded['fusion_heading']).max(), 1e-12)
        self.assertLess(np.abs(i - tr.recorded['integrated_yaw']).max(), 1e-12)
        # 同一点采样路径，按 interval 反事实读取 point 记录：等于 interval 模式的生产结果
        with np.load(p_pt) as z:
            d = {k: z[k] for k in z.files}
        d['data'] = d['data'].copy()
        d['data'][:, COLS.index('raw_yaw')] = rows_int[:, 0]     # 换成带振荡的点采样
        d['data'][:, COLS.index('raw_yaw_interval')] = rows_int[:, 8]
        cf_path = Path(self.tmp.name) / 'cf.npz'
        np.savez(cf_path, **d)
        cf = df.load_trace(cf_path, rate_source='interval')
        self.assertEqual(cf.recorded, {})
        parts = df.decompose([cf], m['noise_std'], m['fusion_q_bias'], m['abs_sigma'])
        f_cf = sum(parts[s][0] for s in df.SOURCES)[0] + cf.psi
        self.assertLess(np.abs(f_cf - tr.recorded['fusion_heading']).max(), 1e-12)

    def test_interval_rate_matches_constant_body_rate(self):
        """匀角速度转动：相邻姿态反推的区间平均角速度应精确回到该角速度。"""
        w = np.array([0.3, -0.2, 0.5])
        n = 50
        R = np.eye(3)
        eul = []
        th = np.linalg.norm(w) * df.DT
        K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]) / np.linalg.norm(w)
        step = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K        # 机体系下的 exp
        for _ in range(n):
            eul.append((np.arctan2(R[2, 1], R[2, 2]), np.arcsin(-R[2, 0]), np.arctan2(R[1, 0], R[0, 0])))
            R = R @ step
        phi, thy, psi = map(np.array, zip(*eul))
        est = df.interval_rate(phi, thy, np.unwrap(psi))
        np.testing.assert_allclose(est, np.tile(w, (n - 1, 1)), atol=1e-10)

    def test_rejects_traces_that_break_linearity(self):
        with np.load(self.path) as z:
            d = {k: z[k] for k in z.files}
        d['data'] = d['data'].copy()
        d['data'][50, COLS.index('reference_rejected')] = 1.0
        bad = Path(self.tmp.name) / 'bad.npz'
        np.savez(bad, **d)
        with self.assertRaises(ValueError):
            df.load_trace(bad)


if __name__ == '__main__':
    unittest.main()
