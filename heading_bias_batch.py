"""Protocol and record validation shared by the user-run bias batches."""
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import zipfile

import numpy as np
from analyze_heading_fixed_input import read_trial, METRICS, LABELS, paired_statistics
from experiment_record import build_run_metadata
from run_heading_fixed_input import SOURCES, digest, verify_identity

BASELINE = Path('build/heading_fixed_input_20260906')
BASELINE_SHA256 = '36441b5b6d6fc6b3a3110569d331ec24850c3d14f38a7fd121635e4ff74b1c69'
EXTRA_SOURCES = ['heading_bias_batch.py', 'run_heading_bias.py', 'analyze_heading_bias.py',
    'run_heading_fixed_input.py', 'run_heading_pilot.py', 'analyze_heading_fixed_input.py',
    'analyze_heading_pilot.py', 'pinned_input_audit.py']


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def load_plan(root):
    path = root/'protocol.json'
    if (root/'protocol.sha256').read_text().strip() != digest(path):
        raise ValueError('Protocol checksum mismatch')
    plan = json.loads(path.read_text(encoding='utf-8'))
    sources = json.loads(plan['identity']['source_sha256_json'])
    sources.update(plan['extra_source_sha256'])
    with zipfile.ZipFile(root/'experiment_source.zip') as archive:
        if set(archive.namelist()) != set(sources):
            raise ValueError('Source archive membership mismatch')
        import hashlib
        for name, expected in sources.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f'Source archive checksum mismatch: {name}')
    return plan


def protocol(project, bias):
    if bias not in (.02, .04):
        raise ValueError('Only the approved 0.02 and 0.04 rad/s batches are supported')
    baseline = project/BASELINE
    if digest(baseline/'protocol.json') != BASELINE_SHA256:
        raise ValueError('The reviewed zero-bias baseline protocol changed')
    base = load_plan(baseline)
    plan = deepcopy(base)
    plan.update(kind='paired_heading_bias_batch', created_utc=datetime.now(timezone.utc).isoformat(),
                baseline_protocol_sha256=BASELINE_SHA256, baseline_relative_path=BASELINE.as_posix())
    plan['common']['gyro_bias'] = bias
    current = build_run_metadata(project, base['common']['checkpoint'], plan['common'], source_paths=SOURCES)
    for key in ('source_tree_sha256', 'checkpoint_sha256'):
        if current[key] != base['identity'][key]:
            raise ValueError(f'Core experiment differs from the zero-bias baseline: {key}')
    plan['identity'] = current
    for name, expected in base['dependency_versions'].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = 'not_registered'
        if actual != expected:
            raise ValueError(f'Dependency differs from baseline: {name}: {actual} != {expected}')
    plan['runner_dependency_versions'] = {'psutil': importlib.metadata.version('psutil')}
    plan['extra_source_sha256'] = {name: digest(project/name) for name in EXTRA_SOURCES}
    plan['scope'] = f'Only 30 trials at gyro_bias={bias}; reuse the existing zero-bias baseline.'
    plan['resume_rule'] = ('A valid completed, fallen or timed-out trace is final and never replaced. '
        'Incomplete infrastructure attempts are retained in numbered attempt directories; '
        '--resume retries the same case and seed only after the prior process has stopped. '
        'Corrupt final traces or checksum mismatches require inspection, never automatic replacement.')
    plan['cross_bias_analysis_rule'] = ('Report C-minus-B full-duration paired differences separately at each bias. '
        'The three bias levels reuse seeds and are not 30 independent seed replicates. '
        'Intervals are marginal, not a multiplicity-adjusted family-wide claim. '
        'Report failures and all attempts; secondary and between-bias comparisons are descriptive.')
    return plan


def freeze(project, root, candidate):
    if (root/'protocol.json').exists():
        saved = load_plan(root)
        for key in ('kind', 'common', 'cases', 'input_contract', 'trace_schema_version',
                    'analysis_rule', 'primary_outcome', 'stability_definition', 'extra_source_sha256',
                    'dependency_versions', 'runner_dependency_versions', 'baseline_protocol_sha256',
                    'resume_rule', 'cross_bias_analysis_rule'):
            if saved[key] != candidate[key]:
                raise ValueError(f'Frozen protocol differs: {key}')
        verify_identity(project, saved)
        return saved
    if any(root.glob('*/attempt_*')) or any(root.glob('*/trace*.npz')):
        raise ValueError('Existing traces without a protocol; choose a new output directory')
    verify_identity(project, candidate)
    sources = json.loads(candidate['identity']['source_sha256_json'])
    sources.update(candidate['extra_source_sha256'])
    with zipfile.ZipFile(root/'experiment_source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, expected in sorted(sources.items()):
            if digest(project/name) != expected:
                raise ValueError(f'Source changed: {name}')
            archive.write(project/name, name)
    write_json(root/'protocol.json', candidate)
    (root/'protocol.sha256').write_text(digest(root/'protocol.json')+'\n', encoding='ascii')
    bias = candidate['common']['gyro_bias']
    (root/'protocol.md').write_text(f'# 常值零偏 {bias:.2f} rad/s 对照\n\n'
        'A/B/C × 种子 24000–24009，共 30 次；每次独立进程、3000 步、60 s。'
        '使用 model_12000，核心代码及全部非零偏参数与已完成的零偏复验一致。\n\n'
        '主要比较为同种子 C−B 航向 RMS。先报告完成率与失败，再报告完整时长配对的均值、SD、95% 配对 t 区间。'
        '三档复用种子，各档区间不是联合多重检验结论。\n\n'
        '跌倒和超时是最终实验结果，不补抽；基础设施中断按 attempt 编号完整保留，续跑使用原种子。'
        '详细参数、源码校验值、运行顺序和统计规则见 protocol.json。\n', encoding='utf-8')
    return candidate


def inspect_attempt(directory, case, plan):
    """Return a final valid row, or None for an incomplete infrastructure attempt."""
    record_path = directory/'run_record.json'
    record = json.loads(record_path.read_text(encoding='utf-8')) if record_path.exists() else {}
    traces = list(directory.glob('trace*.npz'))
    if len(traces) > 1:
        raise ValueError(f'Multiple traces in {directory}; inspect manually')
    if not traces:
        if record.get('status') == 'complete' or record.get('trace_sha256'):
            raise ValueError(f'A finalized trace is missing: {directory}')
        return None
    path = traces[0]
    if record.get('trace_sha256') and record['trace_sha256'] != digest(path):
        raise ValueError(f'Trace checksum mismatch: {path}')
    with np.load(path, allow_pickle=False) as z:
        reason = z['stop_reason'].item()
        if reason == 'interrupted_or_error':
            c = {str(v): i for i, v in enumerate(z['columns'])}
            if np.any(z['data'][:, [c['terminated_after_step'], c['truncated_after_step']]] == 1):
                raise ValueError('Interrupted trace contains a terminal transition; inspect rather than retry')
            if record.get('status') == 'complete':
                raise ValueError('Final record contradicts interrupted trace')
            return None
    row, series = read_trial(path, case, plan)
    row['trace_file'] = path.relative_to(directory.parent.parent).as_posix()
    return row, series


def inspect_case(directory, case, plan):
    if not directory.exists():
        return None
    marker = directory/'completed.json'
    if marker.exists():
        selected = json.loads(marker.read_text(encoding='utf-8'))
        name = selected['attempt']
        if Path(name).name != name or not name.startswith('attempt_'):
            raise ValueError('Invalid completed attempt path')
        result = inspect_attempt(directory/name, case, plan)
        if result is None or result[0]['sha256'] != selected['trace_sha256']:
            raise ValueError('Completed attempt changed or disappeared')
        for previous in directory.glob('attempt_*'):
            if previous.is_dir() and previous.name != name and inspect_attempt(previous, case, plan) is not None:
                raise ValueError('Multiple final outcomes for one case; no outcome-based selection')
        return result
    finals = [result for path in sorted(directory.glob('attempt_*'))
              if path.is_dir() and (result := inspect_attempt(path, case, plan)) is not None]
    if len(finals) > 1:
        raise ValueError('Multiple final outcomes for one case; no outcome-based selection')
    return finals[0] if finals else None


def summarize(rows, plan):
    groups, pairs = {}, []
    for arm in LABELS:
        subset = [r for r in rows if r['arm'] == arm]
        full = [r for r in subset if r['full_duration']]
        groups[arm] = dict(recorded_n=len(subset), planned_n=10, full_duration_n=len(full),
            terminated_n=sum(r['terminated'] for r in subset), truncated_n=sum(r['truncated'] for r in subset),
            completed_only_metrics={m: dict(mean=float(np.mean(v)) if v else None,
                sd=float(np.std(v, ddof=1)) if len(v)>1 else None)
                for m in METRICS for v in [[r[m] for r in full]]})
    for seed in sorted({r['seed'] for r in rows}):
        arms = {r['arm']: r for r in rows if r['seed']==seed}
        poses = [r['initial_pose']+[r['initial_height_m']] for r in arms.values()]
        np.testing.assert_allclose(poses, np.broadcast_to(poses[0], (len(poses), 4)), atol=1e-9, rtol=0)
        if all(a in arms and arms[a]['full_duration'] for a in 'BC'):
            pairs.append(dict(seed=seed, **{m: arms['C'][m]-arms['B'][m] for m in METRICS}))
    return dict(complete=len(rows)==30, gyro_bias=plan['common']['gyro_bias'], rows=rows, groups=groups,
        paired_C_minus_B=pairs, paired_statistics={m: paired_statistics([p[m] for p in pairs]) for m in METRICS})
