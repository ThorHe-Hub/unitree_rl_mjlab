"""Run with python -m unittest discover -s tests -p test_experiment_record.py."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment_record import build_run_metadata, episode_seed_metadata


class ExperimentMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="experiment_metadata_")
        self.project = Path(self.temp.name)
        self.checkpoint = self.project / "model_12000.pt"
        self.checkpoint.write_bytes(b"tiny checkpoint for metadata test")
        self.source = self.project / "play.py"
        self.source.write_text("x = 1\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_npz_safe_round_trip_preserves_full_config_and_actual_values(self):
        cfg = dict(fixed_body=False, pin_cmd=True, task_id="Test-Task", kp=.91,
                   ki=0., kd=.01, feedback_mode="fusion_heading",
                   cfg={"checkpoint_file": self.checkpoint, "noise": np.float64(.15),
                        "limits": np.array([1., 2.]), "optional": None})
        metadata = build_run_metadata(self.project, self.checkpoint, cfg)
        metadata.update(episode_seed_metadata(42, 3))
        np.savez(self.project / "trace.npz", **metadata)
        with np.load(self.project / "trace.npz", allow_pickle=False) as trace:
            for key in trace.files:
                self.assertEqual(trace[key].shape, (), key)
                self.assertNotEqual(trace[key].dtype.kind, "O", key)
            self.assertFalse(trace["fixed_body"].item())
            self.assertEqual(trace["seed"].item(), 42)
            restored = json.loads(trace["config_json"].item())
            self.assertEqual(restored["cfg"]["limits"], [1., 2.])
            self.assertIsNone(restored["cfg"]["optional"])
            self.assertEqual(trace["checkpoint"].item(), str(self.checkpoint.resolve()))
            self.assertEqual(trace["checkpoint_sha256"].item(),
                             hashlib.sha256(self.checkpoint.read_bytes()).hexdigest())

    def test_working_tree_fingerprint_tracks_edits_and_ignores_excluded_files(self):
        (self.project / "src").mkdir()
        source = self.project / "src" / "filter.py"
        source.write_text("value = 2\n")
        (self.project / "tests").mkdir()
        test_file = self.project / "tests" / "test_x.py"
        test_file.write_text("pass\n")
        selected = ["play.py", "src"]
        first = build_run_metadata(self.project, self.checkpoint, {}, selected)
        same = build_run_metadata(self.project, self.checkpoint, {}, reversed(selected))
        self.assertEqual(first["source_tree_sha256"], same["source_tree_sha256"])
        self.assertNotEqual(first["run_id"], same["run_id"])
        self.assertEqual(first["source_file_count"], 2)
        test_file.write_text("raise AssertionError\n")
        self.assertEqual(first["source_tree_sha256"], build_run_metadata(
            self.project, self.checkpoint, {}, selected)["source_tree_sha256"])
        source.write_text("value = 3\n")
        changed = build_run_metadata(self.project, self.checkpoint, {}, selected)
        self.assertNotEqual(first["source_tree_sha256"], changed["source_tree_sha256"])
        self.assertEqual(first["checkpoint_sha256"], changed["checkpoint_sha256"])
        self.checkpoint.write_bytes(b"different checkpoint")
        self.assertNotEqual(first["checkpoint_sha256"], build_run_metadata(
            self.project, self.checkpoint, {}, selected)["checkpoint_sha256"])

    def test_missing_files_and_unknown_fields_are_not_fabricated(self):
        metadata = build_run_metadata(self.project, "missing.pt", {}, ["missing.py"])
        self.assertEqual(metadata["checkpoint_status"], "missing")
        self.assertNotIn("checkpoint_sha256", metadata)
        self.assertNotIn("fixed_body", metadata)
        self.assertNotIn("git_revision", metadata)
        self.assertNotIn("git_dirty", metadata)
        self.assertEqual(json.loads(metadata["source_manifest_json"]),
                         {"missing.py": {"status": "missing"}})
        self.assertEqual(json.loads(metadata["source_sha256_json"]), {})
        self.assertFalse(metadata["source_snapshot_complete"])
        absent = build_run_metadata(self.project, None, {}, [])
        self.assertEqual(absent["checkpoint_status"], "not_provided")
        self.assertNotIn("checkpoint", absent)

    @unittest.skipUnless(shutil.which("git"), "git is unavailable")
    def test_git_revision_and_dirty_state_capture_local_edits(self):
        def git(*args):
            return subprocess.run(["git", "-C", str(self.project), *args],
                                  check=True, capture_output=True, text=True)
        git("init", "--quiet")
        git("add", ".")
        git("-c", "user.name=Metadata Test", "-c", "user.email=test@example.invalid",
            "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "fixture")
        clean = build_run_metadata(self.project, self.checkpoint, {}, [self.source])
        self.assertEqual(clean["git_status"], "ok")
        self.assertFalse(clean["git_dirty"])
        self.source.write_text("x = 2\n", encoding="utf-8")
        dirty = build_run_metadata(self.project, self.checkpoint, {}, [self.source])
        self.assertEqual(clean["git_revision"], dirty["git_revision"])
        self.assertTrue(dirty["git_dirty"])
        self.assertNotEqual(clean["source_tree_sha256"], dirty["source_tree_sha256"])

    def test_episode_seed_streams_are_order_independent_and_isolated(self):
        seeds = episode_seed_metadata(7, 2)
        episode_seed_metadata(7, 99)
        self.assertEqual(seeds, episode_seed_metadata(7, 2))
        names = ("env", "gyro", "reference", "dropout", "vx")
        self.assertEqual(len({seeds["seed_" + name] for name in names}), 5)
        self.assertNotEqual(seeds, episode_seed_metadata(7, 3))
        self.assertNotEqual(seeds, episode_seed_metadata(8, 2))
        baseline = np.random.default_rng(seeds["seed_gyro"]).normal(size=10)
        np.random.default_rng(seeds["seed_reference"]).normal(size=100)
        np.random.default_rng(seeds["seed_dropout"]).random(size=13)
        np.testing.assert_array_equal(baseline, np.random.default_rng(
            episode_seed_metadata(7, 2)["seed_gyro"]).normal(size=10))
        for name in names:
            self.assertNotEqual(seeds["seed_" + name],
                                episode_seed_metadata(7, 3)["seed_" + name])

    def test_seed_inputs_reject_ambiguous_or_unsafe_values(self):
        for value in (True, 1.5, "42"):
            with self.assertRaises(TypeError):
                episode_seed_metadata(value, 0)
        for value in (-1, 2 ** 63):
            with self.assertRaises(ValueError):
                episode_seed_metadata(value, 0)
            with self.assertRaises(ValueError):
                episode_seed_metadata(42, value)
        self.assertEqual(episode_seed_metadata(np.int64(7), np.int64(2)),
                         episode_seed_metadata(7, 2))


if __name__ == "__main__":
    unittest.main()
