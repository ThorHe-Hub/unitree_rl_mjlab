"""Run only the fixed 30-trial zero-injected-bias heading experiment.

Fresh process per trial; no replacement of failures or outcome-based stopping.
--resume skips only previously completed and verified trial records.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from experiment_record import build_run_metadata, episode_seed_metadata
from run_heading_pilot import CHECKPOINT, CHECKPOINT_SHA256, SOURCES, MODES, protocol as pilot_protocol
from analyze_heading_fixed_input import read_trial
SOURCES = [*SOURCES, "pinned_velocity_command.py"]

EXTRA_SOURCES = ['run_heading_fixed_input.py', 'analyze_heading_fixed_input.py',
                 'run_heading_pilot.py', 'analyze_heading_pilot.py', 'pinned_input_audit.py']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol(project):
    plan = pilot_protocol(project)
    cases = []
    orders = ('ABC', 'BCA', 'CAB', 'ACB', 'CBA', 'BAC')
    for index, seed in enumerate(range(24000, 24010)):
        for arm in orders[index % len(orders)]:
            filt, feedback = MODES[arm]
            cases.append({'case_id': f'{arm}_seed{seed}', 'arm': arm, 'seed': seed,
                          'filt': filt, 'feedback_mode': feedback,
                          'derived_seeds': episode_seed_metadata(seed, 0)})
    plan['kind'] = 'ten_seed_zero_bias_source_held_command_rerun'
    plan['cases'] = cases
    plan['common']['gyro_bias'] = 0.0
    plan['identity'] = build_run_metadata(project, CHECKPOINT, plan['common'], source_paths=SOURCES)
    plan['trace_schema_version'] = 4
    plan['input_contract'] = 'Commands held at source before phase observation; actor/environment receive current PID, observation receives previous held PID; vx=0.5, vy=0, standing=False, phase period 0.6 s.'
    plan['prior_batch'] = 'build/heading_zero_bias_20260906'
    plan['prior_protocol_sha256'] = digest(project / plan['prior_batch'] / 'protocol.json')
    plan['prior_protocol_expected_sha256'] = '5ff404d7a08bb0ecafe28c8660021397bd41993f91ad5f83c53400801e726b16'
    if plan['prior_protocol_sha256'] != plan['prior_protocol_expected_sha256']:
        raise RuntimeError('Prior protocol changed')
    plan['extra_source_sha256'] = {name: digest(project / name) for name in EXTRA_SOURCES}
    for name in ('scipy', 'matplotlib'):
        plan['dependency_versions'][name] = importlib.metadata.version(name)
    plan['primary_outcome'] = 'Physical heading RMS over the full 60 s; paired C-minus-B difference by seed'
    plan['secondary_outcomes'] += ['Root-link tilt RMS/peak/P95', 'Root-link world z minimum/mean/SD/range']
    plan['analysis_rule'] = (
        'Keep all 30 planned trials, including falls and timeouts; no resampling or outcome-based stopping. '
        'Report completion and failure counts before completed-trial metrics. Primary comparison uses '
        'only B/C pairs both completing 3000 steps, each seed equally weighted: mean(C RMS - B RMS), '
        'sample SD and two-sided 95% paired t interval (n-1 degrees of freedom). This interval assumes '
        'independent seed-level differences and uses a small-sample t approximation. If failures occur, '
        'completed-pair analysis is conditional and does not establish overall superiority. '
        'All secondary metrics/comparisons are descriptive; no multiplicity-adjusted significance '
        'or gait/energy equivalence claims. Report individual seed results and all trajectories.')
    plan['stability_definition'] = {
        'base_tilt': 'acos of normalized root-link rotation matrix zz entry, radians; angle to world vertical',
        'base_height_w': 'root-link world z, metres; not total robot height',
        'timing': 'pre_physics_step_same_as_gt_yaw'}
    plan['scope'] = 'This command runs only gyro_bias=0.0, not the later 0.02 or 0.04 batches.'
    return plan


def verify_identity(project, plan):
    current = build_run_metadata(project, CHECKPOINT, {}, source_paths=SOURCES)
    for name in ('source_tree_sha256', 'checkpoint_sha256'):
        if current[name] != plan['identity'][name]:
            raise RuntimeError(f'Experiment identity changed: {name}')
    if current['checkpoint_sha256'] != CHECKPOINT_SHA256 or not current['source_snapshot_complete']:
        raise RuntimeError('Missing source or incorrect checkpoint')
    for name, expected in plan['extra_source_sha256'].items():
        if digest(project / name) != expected:
            raise RuntimeError(f'Runner/analysis source changed: {name}')


def freeze(project, output, candidate):
    path = output / 'protocol.json'
    if path.exists():
        plan = json.loads(path.read_text(encoding='utf-8'))
        for key in ('kind', 'common', 'cases', 'primary_outcome', 'analysis_rule',
                    'trace_schema_version', 'stability_definition', 'extra_source_sha256', 'dependency_versions',
                    'input_contract', 'prior_batch', 'prior_protocol_sha256'):
            if plan[key] != candidate[key]:
                raise RuntimeError(f'Frozen protocol differs: {key}')
        if (output / 'protocol.sha256').read_text().strip() != digest(path):
            raise RuntimeError('Frozen protocol bytes changed')
        verify_identity(project, plan)
        if not (output / 'experiment_source.zip').exists():
            raise RuntimeError('Frozen source archive is missing')
        return plan
    verify_identity(project, candidate)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(candidate, stream, ensure_ascii=False, indent=2, allow_nan=False)
    (output / 'protocol.sha256').write_text(digest(path) + '\n', encoding='ascii')
    sources = json.loads(candidate['identity']['source_sha256_json'])
    sources.update(candidate['extra_source_sha256'])
    with zipfile.ZipFile(output / 'experiment_source.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, expected in sorted(sources.items()):
            source = project / name
            if digest(source) != expected:
                raise RuntimeError(f'Source changed during snapshot: {name}')
            archive.write(source, name)
    text = [
        '# 修复指令与相位一致性后的零偏复验协议（运行前固定）', '',
        '只执行本批 30 次：A/B/C 三方案 × 与旧批次相同的种子 24000–24009。每次独立进程，计划 3000 步、60 s。', '',
        'A：加噪陀螺直接积分；B：融合补偿角速度后独立积分；C：融合航向直接反馈。', '',
        '三方案保持 model_12000.pt、PID（0.91/0/0.01）、平地、固定物理参数、摩擦 0.8、前进指令 0.5 m/s。'
        '陀螺白噪声标准差 0.15 rad/s、注入零偏 0；合成航向参考标准差 0.05 rad、1 Hz、无丢包。', '',
        '主要指标：60 s 的真实航向 RMS。主要比较：同种子 C 减 B 的差值均值、样本 SD 与双侧 95% 配对 t 区间。'
        '不同种子差值视为独立，采用小样本 t 近似；其余指标作描述分析。', '',
        '保留全部失败，首次终止就结束该次，不补抽种子。若 B 或 C 失败，该对不进入完整时长比较；'
        '先报告全部完成率和失败记录，条件于成功的误差不能单独代表总体表现。基础设施错误中止批次、保留日志。', '',
        '新增倾角为根链接 z 轴对竖直方向夹角，高度为根链接世界 z，与航向真值同在物理步之前采样。'
        '末个样本为 59.98 s；发生终止时，不把重置后的状态混入。倾角、高度和未跌倒不能证明步态等效，控制指令不是能耗。', '',
        '指令在观测计算之前固定；保留原重采样随机数消耗以维持配对条件，但随机提案不生效。记录观测入口、实际策略输入、环境指令及站立标志，逐步验证相位。旧批次独立保留。', '',
        '## 固定运行顺序', '', '| 序号 | 运行 |', '|---:|---|']
    text += [f'| {i} | {case["case_id"]} |' for i, case in enumerate(candidate['cases'], 1)]
    text += ['', f'权重 SHA256：`{CHECKPOINT_SHA256}`', '',
             f'源码树 SHA256：`{candidate["identity"]["source_tree_sha256"]}`', '',
             f'协议 SHA256：`{digest(path)}`', '',
             '详细参数、衍生随机种子、依赖版本和源码校验值见 protocol.json；源码存档见 experiment_source.zip。', '']
    (output / 'protocol.md').write_text('\n'.join(text), encoding='utf-8')
    return candidate


def command_for(project, case, case_dir, common):
    optional = [value for name in ('fusion_q_bias', 'fusion_sigma_bias_prior', 'abs_outage_start_s', 'abs_outage_duration_s')
                if name in common for value in ('--rig.' + name.replace('_', '-'), str(common[name]))]
    if common.get('contact_friction') is not None:
        optional.extend(['--contact-friction', str(common['contact_friction'])])
    return [sys.executable, '-X', 'utf8', 'scripts/play.py', common['task_id'],
        '--checkpoint-file', CHECKPOINT, '--device', 'cuda:0', '--pin-cmd', 'True',
        '--pin-vx', str(common['pin_vx']), '--fixed-body', 'True', '--friction', str(common['friction']),
        '--kp', str(common['kp']), '--ki', str(common['ki']), '--kd', str(common['kd']),
        '--rig.noise-std', str(common['noise_std']), '--rig.gyro-bias', str(common['gyro_bias']),
        '--rig.abs-sigma', str(common['abs_sigma']), '--rig.abs-hz', str(common['abs_hz']),
        '--rig.abs-dropout', str(common['abs_dropout']), '--rig.filt', case['filt'],
        '--feedback-mode', case['feedback_mode'], '--seed', str(case['seed']),
        '--batch-steps', str(common['steps']), '--repeats', '1', '--no-terminations', 'False',
        '--batch-stop-on-done', 'True', '--output-dir', str(case_dir), *optional]


def verify_finished(case_dir, case, plan):
    record = json.loads((case_dir / 'run_record.json').read_text(encoding='utf-8'))
    traces = list(case_dir.glob('trace*.npz'))
    if record['returncode'] != 0 or len(traces) != 1:
        raise RuntimeError(f'Incomplete/failed infrastructure record: {case["case_id"]}; no automatic replacement')
    row, _ = read_trial(traces[0], case, plan)
    if record.get('trace_sha256') != row['sha256']:
        raise RuntimeError(f'Trace changed: {case["case_id"]}')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = freeze(project, output, protocol(project))
    print(f'[PROTOCOL] {output / "protocol.json"}', flush=True)
    if args.prepare_only:
        return
    env = {**os.environ, 'GIT_PYTHON_REFRESH': 'quiet', 'PYTHONIOENCODING': 'utf-8',
           'PYTHONUNBUFFERED': '1', 'PYTHONPATH': str(project) + os.pathsep + os.environ.get('PYTHONPATH', '')}
    for number, case in enumerate(plan['cases'], 1):
        verify_identity(project, plan)
        case_dir = output / case['case_id']
        if case_dir.exists():
            if not args.resume:
                raise RuntimeError(f'Existing case: {case["case_id"]}; use --resume only to verify and skip finished trials')
            verify_finished(case_dir, case, plan)
            print(f'[SKIP VERIFIED] {number}/30 {case["case_id"]}', flush=True)
            continue
        case_dir.mkdir()
        command = command_for(project, case, case_dir, plan['common'])
        record = {**case, 'command': command, 'started_utc': datetime.now(timezone.utc).isoformat()}
        (case_dir / 'launch_record.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        print(f'[RUN] {number}/30 {case["case_id"]}', flush=True)
        started = time.monotonic()
        with (case_dir / 'console.txt').open('x', encoding='utf-8') as log:
            child = subprocess.Popen(command, cwd=project, env=env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace',
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            for line in child.stdout:
                log.write(line)
                log.flush()
                if line.startswith(('[PROGRESS]', '[TRACE]')):
                    print(line.rstrip(), flush=True)
            returncode = child.wait()
        record.update(returncode=returncode, wall_seconds=time.monotonic() - started,
                      ended_utc=datetime.now(timezone.utc).isoformat())
        traces = list(case_dir.glob('trace*.npz'))
        record['traces'] = [str(p.relative_to(output)) for p in traces]
        if returncode == 0 and len(traces) == 1:
            record['trace_sha256'] = digest(traces[0])
        (case_dir / 'run_record.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        row = verify_finished(case_dir, case, plan)
        print(f'[END] {number}/30 {case["case_id"]} reason={row["stop_reason"]} '
              f'steps={row["samples"]} wall={record["wall_seconds"]:.1f}s', flush=True)
    print('[COMPLETE] All 30 planned zero-injected-bias trials recorded and audited.', flush=True)


if __name__ == '__main__':
    main()
