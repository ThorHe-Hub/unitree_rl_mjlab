"""Run with python -m unittest discover -s tests -p test_agg.py after installation."""
import contextlib
import io
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agg


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.before = Path.cwd()
        self.temp = tempfile.TemporaryDirectory(prefix="thesis_agg_test_")
        os.chdir(self.temp.name)

    def tearDown(self):
        os.chdir(self.before)
        self.temp.cleanup()

    def trace(self, name, steps=3000, **meta):
        d = np.zeros((steps, 5))
        d[:, 1] = np.arange(steps) * 0.01
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = dict(dt=.02, friction=.8, noise_std=.15, gyro_bias=0., filt="dirty",
                      checkpoint="logs/run/model_10.pt")
        fields.update(meta)
        np.savez(path, data=d, columns=np.array(agg.REQUIRED), **fields)
        return path

    def test_both_directories_and_overlapping_patterns(self):
        self.trace("trace_a.npz")
        self.trace("traces/trace_b.npz", gyro_bias=.01)
        found = agg.discover(["trace*.npz", "traces/trace*.npz"])
        self.assertEqual(len(found), 2)
        self.assertEqual(len(agg.discover(["traces/trace*.npz"])), 1)

    def test_duplicate_copies_are_tracked_not_counted_twice(self):
        p = self.trace("trace_a.npz")
        Path("traces").mkdir()
        shutil.copyfile(p, "traces/trace_copy.npz")
        records = agg.collect([], "main")
        self.assertEqual(len(records), 2)
        self.assertEqual(sum(r["selected"] for r in records), 1)
        self.assertEqual(sum(bool(r["duplicate_of"]) for r in records), 1)

    def test_cohorts_and_duration_are_not_mixed(self):
        self.trace("trace_main.npz")
        self.trace("trace_sweep.npz", sweep_vyaw=.5, sweep_seed=0)
        self.trace("trace_seed1.npz", sweep_vyaw=.5, sweep_seed=1)
        self.trace("traces/trace_old.npz", steps=800)
        self.trace("traces/trace_other.npz", steps=812)
        records = agg.collect([], "main")
        self.assertEqual(sum(r["selected"] for r in records), 1)
        _, stats = agg.summarize(agg.collect([], "all"))
        self.assertEqual(len(stats), 5)

    def test_checkpoint_filename_prevents_accidental_merge(self):
        self.trace("trace_a.npz", checkpoint=r"D:\logs\run\model_10.pt")
        self.trace("trace_b.npz", checkpoint=r"D:\logs\run\model_20.pt")
        _, stats = agg.summarize(agg.collect([], "main"))
        self.assertEqual(len(stats), 2)

    def test_filter_pivot_uses_filter_dimension(self):
        for bias in [0., .01]:
            for filt in ["dirty", "fusion"]:
                self.trace(f"trace_{bias}_{filt}.npz", gyro_bias=bias, filt=filt)
        records = agg.collect([], "main")
        _, stats = agg.summarize(records)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            agg.print_stats(records, stats)
        self.assertIn("列 = 闭环方案", out.getvalue())
        self.assertIn("fusion vs dirty", out.getvalue())
        self.assertIn("样本不足", out.getvalue())

    def test_missing_flags_are_not_invented_and_n1_has_no_sd(self):
        self.trace("trace_a.npz")
        records = agg.collect([], "main")
        self.assertIn("fixed_body", records[0]["missing_metadata"])
        self.assertNotIn("fixed_body", records[0]["assumed_defaults"])
        self.assertEqual(records[0]["key"][agg.DIMS.index("固定机体")], "unrecorded")
        _, stats = agg.summarize(records)
        self.assertTrue(np.isnan(next(iter(stats.values()))[1]).all())
        groups, stats = agg.summarize(records)
        agg.export_audit("audit", records, groups, stats)
        self.assertIn('"sd": null', Path("audit/summary.json").read_text(encoding="utf-8"))

    def test_metrics_known_geometry_and_invalid_data(self):
        self.trace("trace_a.npz", steps=800)
        r = agg.load_record("trace_a.npz")
        np.testing.assert_allclose(r["metrics"][:7], [0, 0, 7.99, 7.99, 0, 0, 0])
        self.assertTrue(np.isnan(r["metrics"][7:]).all())
        d = np.zeros((2, 5)); d[0, 0] = np.nan
        with self.assertRaises(ValueError):
            agg.metrics(d, agg.REQUIRED)

    def test_new_feedback_metrics_use_actual_pre_update_feedback(self):
        columns=agg.REQUIRED+["pid_feedback_yaw","segment_id"]
        d=np.zeros((3,len(columns)))
        d[:,0]=[0.,.1,.2]
        d[:,3]=[.02,.15,.28]  # post-update integral, intentionally different
        d[:,5]=[0.,.09,.18]   # actual pre-update feedback
        values,_=agg.metrics(d,columns)
        self.assertAlmostEqual(values[5],.06)
        self.assertAlmostEqual(values[7],.02)
        self.assertAlmostEqual(values[8],np.sqrt((.01**2+.02**2)/3))
        d[-1,6]=1
        with self.assertRaisesRegex(ValueError,"多个区段"):
            agg.metrics(d,columns)

    def test_recorded_runs_keep_feedback_and_code_identity_separate(self):
        for i,(mode,code) in enumerate([("integrated","a"),("fusion_heading","a"),("fusion_heading","b")]):
            self.trace(f"trace_{i}.npz",filt="fusion",trace_schema_version=2,
                       feedback_mode=mode,source_tree_sha256=code)
        self.assertEqual(sum(r["selected"] for r in agg.collect([],"main")),0)
        groups,stats=agg.summarize(agg.collect([],"recorded"))
        self.assertEqual(len(stats),3)


if __name__ == "__main__":
    unittest.main()
