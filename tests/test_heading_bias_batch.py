"""Resume and provenance tests using CPU fake-environment records, never a simulator."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heading_bias_batch import inspect_case, inspect_attempt, summarize, write_json, BASELINE_SHA256
from run_heading_fixed_input import digest
from run_heading_bias import run_case, mark_complete, batch_lock
from analyze_heading_bias import check_comparable, collect_batch
from experiment_record import episode_seed_metadata
import test_play_feedback


class BiasBatchTests(unittest.TestCase):
    def fixture(self, fall=False):
        helper=test_play_feedback.PlayFeedbackIntegrationTests()
        self.addCleanup(helper.doCleanups)
        run=helper.run_case(mode='fusion_heading', real_metadata=True, steps=6,
                           reset_every=3 if fall else None, stop_on_done=True)
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root=Path(temp.name); case_dir=root/'C_seed73'; attempt=case_dir/'attempt_001'
        attempt.mkdir(parents=True)
        path=attempt/'trace_test.npz'; shutil.copyfile(run['trace_paths'][0],path)
        trace=run['traces'][0]
        common=json.loads(trace['config_json'].item()); common.update(steps=6,duration_s=.12)
        case=dict(case_id='C_seed73',arm='C',seed=73,filt='fusion',feedback_mode='fusion_heading',
                  derived_seeds=episode_seed_metadata(73,0))
        plan=dict(common=common,cases=[case],identity={k:trace[k].item() for k in ('checkpoint_sha256','source_tree_sha256')})
        write_json(attempt/'run_record.json',dict(status='process_ended',returncode=0,trace_sha256=digest(path)))
        return root,case_dir,attempt,path,case,plan,trace

    def test_resume_skips_a_verified_completed_case_without_launching(self):
        root,d,attempt,path,case,plan,_=self.fixture()
        with patch('run_heading_bias.subprocess.Popen',side_effect=AssertionError('must not launch')):
            row=run_case(root,root,case,plan,True)
        self.assertTrue(row['full_duration'])
        self.assertTrue((d/'completed.json').exists())

    def test_fall_is_final_even_if_launcher_cleanup_failed(self):
        root,d,attempt,path,case,plan,_=self.fixture(fall=True)
        write_json(attempt/'run_record.json',dict(status='infrastructure_error',returncode=1,trace_sha256=digest(path)))
        with patch('run_heading_bias.subprocess.Popen',side_effect=AssertionError('must not retry a fall')):
            row=run_case(root,root,case,plan,True)
        self.assertTrue(row['terminated']); self.assertFalse(row['full_duration'])

    def test_incomplete_attempt_is_not_a_final_outcome(self):
        root,d,attempt,path,case,plan,trace=self.fixture()
        np.savez(path,**dict(trace,stop_reason=np.array('interrupted_or_error')))
        write_json(attempt/'run_record.json',dict(status='interrupted',returncode=-1,trace_sha256=digest(path)))
        self.assertIsNone(inspect_case(d,case,plan))
        self.assertTrue(path.exists())

    def test_trace_checksum_mismatch_is_not_treated_as_retryable(self):
        root,d,attempt,path,case,plan,_=self.fixture()
        with path.open('ab') as stream: stream.write(b'changed')
        with self.assertRaisesRegex(ValueError,'checksum'):
            inspect_case(d,case,plan)

    def test_resume_allocates_new_attempt_and_retains_interrupted_bytes(self):
        root,d,attempt,path,case,plan,trace=self.fixture()
        np.savez(path,**dict(trace,stop_reason=np.array('interrupted_or_error')))
        write_json(attempt/'run_record.json',dict(status='interrupted',returncode=-1,trace_sha256=digest(path)))
        original=digest(path)
        def launch(command,**kwargs):
            output=Path(command[command.index('--output-dir')+1])
            np.savez(output/'trace_retry.npz',**trace)
            return SimpleNamespace(pid=123456,stdout=[],returncode=0,wait=lambda:0)
        with patch('run_heading_bias.subprocess.Popen',side_effect=launch) as mocked, \
             patch('run_heading_bias.psutil.Process',return_value=SimpleNamespace(create_time=lambda:0.)):
            row=run_case(root,root,case,plan,True)
        self.assertEqual(mocked.call_count,1)
        self.assertTrue(row['full_duration'])
        self.assertEqual(digest(path),original)
        self.assertEqual(json.loads((d/'completed.json').read_text())['attempt'],'attempt_002')

    def test_missing_final_trace_does_not_trigger_replacement(self):
        root,d,attempt,path,case,plan,_=self.fixture()
        mark_complete(d,inspect_attempt(attempt,case,plan))
        path.rename(path.with_suffix('.removed_for_test'))
        with self.assertRaisesRegex(ValueError,'missing'):
            inspect_case(d,case,plan)

    def test_duplicate_final_outcomes_are_rejected(self):
        root,d,attempt,path,case,plan,_=self.fixture()
        other=d/'attempt_002'; other.mkdir()
        shutil.copyfile(path,other/path.name)
        with self.assertRaisesRegex(ValueError,'Multiple final'):
            inspect_case(d,case,plan)

    def test_completed_marker_cannot_hide_another_final_outcome(self):
        _,d,attempt,path,case,plan,_=self.fixture()
        mark_complete(d,inspect_case(d,case,plan))
        other=d/'attempt_002'; other.mkdir(); shutil.copyfile(path,other/path.name)
        with self.assertRaisesRegex(ValueError,'Multiple final'):
            inspect_case(d,case,plan)

    def test_a_partial_read_waits_for_completion_marker(self):
        root,d,attempt,path,case,plan,_=self.fixture()
        write_json(root/'protocol.json',plan)
        summary,_=collect_batch(root,plan,True)
        self.assertEqual(len(summary['rows']),0)
        mark_complete(d,inspect_case(d,case,plan))
        summary,_=collect_batch(root,plan,True)
        self.assertEqual(len(summary['rows']),1)

    def test_bias_is_the_only_permitted_experiment_parameter_change(self):
        _,_,_,_,_,plan,_=self.fixture()
        plan.update(input_contract='held',trace_schema_version=4,stability_definition={},dependency_versions={},analysis_rule='paired')
        changed=deepcopy(plan); changed['common']['gyro_bias']=.04
        changed['baseline_protocol_sha256']=BASELINE_SHA256
        check_comparable(plan,changed)
        changed['common']['kp']+=.1
        with self.assertRaisesRegex(ValueError,'parameters'):
            check_comparable(plan,changed)
        changed=deepcopy(plan); changed['baseline_protocol_sha256']=BASELINE_SHA256
        changed['identity']['source_tree_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'source_tree'):
            check_comparable(plan,changed)

    def test_failure_counts_are_retained_and_not_in_complete_metrics(self):
        _,d,_,_,case,plan,_=self.fixture(fall=True)
        row,_=inspect_case(d,case,plan)
        r=summarize([row],plan)
        self.assertEqual(r['groups']['C']['terminated_n'],1)
        self.assertIsNone(r['groups']['C']['completed_only_metrics']['heading_rms_deg']['mean'])
        self.assertEqual(r['paired_statistics']['heading_rms_deg']['n'],0)

    def test_same_batch_cannot_be_locked_twice(self):
        with tempfile.TemporaryDirectory() as temp:
            with batch_lock(Path(temp)):
                with self.assertRaises(OSError):
                    with batch_lock(Path(temp)): pass

    def test_prepare_only_cannot_call_the_simulation_runner(self):
        import run_heading_bias
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(sys,'argv',['run_heading_bias.py','--bias','.02','--output-dir',temp,'--prepare-only']), \
                 patch.object(run_heading_bias,'protocol',return_value={}), \
                 patch.object(run_heading_bias,'freeze',return_value={}), \
                 patch.object(run_heading_bias,'run_case',side_effect=AssertionError('simulation forbidden')), \
                 redirect_stdout(io.StringIO()):
                run_heading_bias.main()


if __name__=='__main__':
    unittest.main()
