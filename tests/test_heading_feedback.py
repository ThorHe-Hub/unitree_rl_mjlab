import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import unittest

import numpy as np
from heading_feedback import HeadingFeedback
from sensor_rig import SensorRig, SensorRigCfg
from experiment_record import episode_seed_metadata


def rig(seed=17, **kwargs):
    seeds = episode_seed_metadata(seed, 0)
    return SensorRig(SensorRigCfg(filt="fusion", **kwargs),
        rng=np.random.default_rng(seeds["seed_gyro"]),
        reference_rng=np.random.default_rng(seeds["seed_reference"]),
        dropout_rng=np.random.default_rng(seeds["seed_dropout"]))


def controller():
    return SimpleNamespace(Kp=.91, Ki=.1, Kd=.01, yaw_target=0.,
                           is_turning=False, error_integral=0., last_error=0.)


class FeedbackTests(unittest.TestCase):
    def test_mode_changes_feedback_only_and_preserves_timing(self):
        rs = [rig(gyro_bias=.02), rig(gyro_bias=.02)]
        fs = [HeadingFeedback("integrated", "fusion"), HeadingFeedback("fusion_heading", "fusion")]
        cs = [controller(), controller()]
        outputs, values = [[], []], [[], []]
        previous, integral = [0., 0.], 0.
        for k in range(800):
            rate = .3 * np.sin(k * .1)
            # Same physical input and reference for paired offline comparison.
            ref = None if k == 0 else .06 * np.sin((k-1) * .1)
            for j in range(2):
                self.assertEqual(fs[j].value, previous[j])
                fs[j].command(cs[j], False, .02)
                outputs[j].append(rs[j].step(rate, ref_psi=ref)[3])
                fs[j].advance(outputs[j][-1], .02, rs[j].fusion_heading)
                previous[j] = fs[j].value
                values[j].append(fs[j].value)
            integral += outputs[0][-1] * .02
            self.assertEqual(fs[0].value, integral)
            self.assertEqual(fs[1].integrated_yaw, integral)
            self.assertEqual(fs[1].value, rs[1].fusion_heading)
        np.testing.assert_array_equal(outputs[0], outputs[1])
        self.assertGreater(np.max(np.abs(np.array(values[0])-values[1])), .001)

    def test_legacy_pid_arithmetic_and_turning_target(self):
        f, c = HeadingFeedback(), controller()
        f.advance(.5, .02)
        # error=-.01, derivative=-.5, I=-.0002 at the next sample.
        self.assertAlmostEqual(f.command(c, False, .02), -.0091-.005-.00002)
        self.assertEqual(f.command(c, True, .02), 0.)
        self.assertEqual(c.yaw_target, .01)
        f.advance(.2, .02)
        self.assertEqual(f.command(c, False, .02), 0.)
        self.assertEqual(c.yaw_target, .014)

    def test_continuous_heading_crosses_pi(self):
        r = rig(noise_std=0., gyro_bias=0., abs_sigma=.01, abs_hz=10.)
        hs=[]
        for k in range(1000):
            absolute = 3.0 + .3 * k * .02
            wrapped = np.arctan2(np.sin(absolute), np.cos(absolute))
            r.step(.3, ref_psi=wrapped)
            hs.append(r.fusion_heading)
        self.assertGreater(hs[-1], np.pi)
        self.assertLess(np.max(np.abs(np.diff(hs))), .08)
        self.assertEqual(r.reference_reject_count, 0)

    def test_reference_cadence_dropout_and_gyro_stream_isolation(self):
        rs=[rig(abs_hz=2., abs_dropout=p) for p in [0., .5, 1.]]
        rs.append(rig(abs_hz=0.))
        dirty=[[] for _ in rs]
        for _ in range(100):
            for j,r in enumerate(rs): dirty[j].append(r.step(0., ref_psi=0.)[0])
        self.assertEqual(rs[0].reference_update_count,4)
        self.assertTrue(0 < rs[1].reference_update_count < 4)
        self.assertEqual(rs[2].reference_update_count,0)
        self.assertEqual(rs[3].reference_update_count,0)
        for d in dirty[1:]: np.testing.assert_array_equal(dirty[0],d)
        with self.assertRaises(ValueError): rig(abs_hz=51.)

    def test_reset_matches_fresh_and_rejected_reference_is_logged(self):
        r=rig(noise_std=0., abs_sigma=0.)
        for _ in range(200): r.step(.2, ref_psi=.2)
        r.reset()
        fresh=rig(noise_std=0., abs_sigma=0.)
        for _ in range(100):
            np.testing.assert_allclose(r.step(0., ref_psi=2.),fresh.step(0.,ref_psi=2.))
            self.assertEqual(r.fusion_heading,fresh.fusion_heading)
        r.step(0.,ref_psi=-1.)
        self.assertTrue(r.reference_rejected)
        self.assertEqual(r.reference_reject_count,1)
        f=HeadingFeedback("fusion_heading","fusion")
        f.advance(.5,.02,1.)
        f.reset()
        self.assertEqual((f.value,f.integrated_yaw),(0.,0.))

    def test_known_bias_converges_on_turning_signal(self):
        r=rig(noise_std=.03, gyro_bias=.02, abs_sigma=.01,abs_hz=5.)
        for k in range(3000):
            r.step(.25, ref_psi=np.arctan2(np.sin(.25*k*.02),np.cos(.25*k*.02)))
        self.assertLess(abs(r.fusion_bias-.02),.004)
        self.assertTrue(np.isfinite(r.fusion_heading))

    def test_invalid_feedback_combination(self):
        with self.assertRaises(ValueError): HeadingFeedback("fusion_heading","dirty")
        with self.assertRaises(ValueError): HeadingFeedback("bad")
        with self.assertRaises(ValueError):
            HeadingFeedback("fusion_heading","fusion").advance(0.,.02,None)
        for tau in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                HeadingFeedback("fusion_offset_smoothed", "fusion", tau)

    def test_offset_tau_default_is_legacy_and_limits_reach_b_and_c(self):
        rng = np.random.default_rng(3)
        rates, heads = rng.normal(0, .3, 400), np.cumsum(rng.normal(0, .01, 400))
        legacy = HeadingFeedback("fusion_offset_smoothed", "fusion")
        explicit = HeadingFeedback("fusion_offset_smoothed", "fusion", 0.1)
        fast = HeadingFeedback("fusion_offset_smoothed", "fusion", 1e-9)
        slow = HeadingFeedback("fusion_offset_smoothed", "fusion", 1e12)
        # Legacy formula written out independently of the class.
        a, off, integ = math.exp(-.02 / .1), 0., 0.
        for w, f in zip(rates, heads):
            for fb in (legacy, explicit, fast, slow):
                fb.advance(w, .02, f)
            integ += w * .02
            off = a * off + (1 - a) * (f - integ)
            self.assertEqual(legacy.value, integ + off)
            self.assertEqual(explicit.value, legacy.value)
            self.assertAlmostEqual(fast.value, f, places=12)             # tau -> 0: C
            self.assertAlmostEqual(slow.value, slow.integrated_yaw, places=9)  # tau -> inf: B


if __name__ == "__main__":
    unittest.main()
