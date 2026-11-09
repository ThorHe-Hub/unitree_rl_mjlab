"""Audit and describe the fixed nine-case heading pilot; no significance tests."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

LABELS = {'A': 'Noisy gyro integral', 'B': 'Bias-corrected integral', 'C': 'Fused heading feedback'}
METRICS = ['heading_rms_deg', 'heading_final_abs_deg', 'feedback_rms_deg',
           'control_rms_rad_s', 'control_peak_rad_s', 'control_saturation_pct',
           'max_control_step_rad_s', 'forward_m', 'lateral_abs_m', 'path_length_m']


def rrw_walk(meta, n):
    """重建注入的零偏随机游走 ε_r（Allan 三分法的第二项）。

    SensorRig.corrupt() 每步抽一次 N(0, sqrt(bias_rrw)) 并累加，
    所以这里是该增量的前缀和。旧轨迹没有 bias_rrw 键，返回全零，
    于是 `dirty - raw - gyro_bias` 仍等于纯白噪声，保持向后兼容。
    """
    sigma = float(meta.get('bias_rrw', 0.0) or 0.0)
    if sigma <= 0.0:
        return np.zeros(n)
    seed_bias = meta.get('seed_bias')
    if seed_bias is None:
        raise ValueError('Trace has bias_rrw > 0 but no seed_bias; cannot rebuild eps_r')
    draws = np.random.default_rng(seed_bias).normal(0.0, np.sqrt(sigma), n)
    return np.cumsum(draws)


def describe_trace(path, case, plan):
    with np.load(path, allow_pickle=False) as z:
        data, columns = z['data'].copy(), z['columns'].tolist()
        meta = {key: z[key].item() for key in z.files if key not in ('data', 'columns')}
    c = {name: i for i, name in enumerate(columns)}
    col = lambda name: data[:, c[name]]
    expected = {**plan['common'], **case}
    # bias_rrw 只在启用 Allan 三分法的批次里出现；旧批次没有这个键，
    # 所以按协议里是否存在来决定是否校验，保持向后兼容。
    checked = ['seed', 'filt', 'feedback_mode', 'pin_cmd', 'pin_vx', 'fixed_body',
               'friction', 'kp', 'ki', 'kd', 'noise_std', 'gyro_bias', 'abs_sigma',
               'abs_hz', 'abs_dropout', 'batch_stop_on_done', 'no_terminations', 'dt']
    if 'bias_rrw' in expected:
        checked.append('bias_rrw')
    for name in checked:
        if meta[name] != expected[name]:
            raise ValueError(f'{case["case_id"]}: unexpected {name}: {meta[name]}')
    for name in ('checkpoint_sha256', 'source_tree_sha256'):
        if meta[name] != plan['identity'][name]:
            raise ValueError(f'{case["case_id"]}: unexpected {name}')
    for name, seed in case['derived_seeds'].items():
        if meta[name] != seed:
            raise ValueError(f'{case["case_id"]}: seed mismatch {name}')
    config = json.loads(meta['config_json'])
    if set(config['termination_terms']) != {'fell_over', 'time_out'}:
        raise ValueError('Expected fall and timeout terms')
    if config['termination_terms']['fell_over']['time_out']:
        raise ValueError('Fall incorrectly classified as timeout')
    if config['episode_length_s'] < plan['common']['duration_s']:
        raise ValueError('Environment timeout is shorter than the planned run')
    if len(data) < 1 or meta['segments'] != 1 or np.any(col('segment_id') != 0):
        raise ValueError('Missing or spliced trajectory')
    for name in ('gt_yaw', 'gt_x', 'gt_y', 'raw_yaw', 'dirty_yaw', 'used_yaw',
                 'pid_feedback_yaw', 'pid_target_yaw', 'pid_correction'):
        if not np.isfinite(col(name)).all():
            raise ValueError(f'Invalid samples: {name}')
    if not np.all(col('pid_active') == 1) or not np.all(col('pid_target_yaw') == 0):
        raise ValueError('Pilot requires constant heading lock from the initial origin')
    if not np.allclose(col('cmd_vyaw'), col('pid_correction'), rtol=0, atol=1e-12):
        raise ValueError('Actor turning command differs from heading controller output')
    np.testing.assert_array_equal(col('pid_feedback_yaw')[1:], col('feedback_heading')[:-1])
    if case['feedback_mode'] == 'fusion_offset_smoothed':
        # Independently evaluate the fixed recurrence using scipy's IIR filter.
        from scipy.signal import lfilter
        a = np.exp(-meta['dt'] / .1)
        expected_feedback = col('integrated_yaw') + lfilter([1-a], [1., -a], col('fusion_heading')-col('integrated_yaw'))
        np.testing.assert_allclose(col('feedback_heading'), expected_feedback, rtol=0, atol=1e-12)
    else:
        source_name = 'fusion_heading' if case['feedback_mode'] == 'fusion_heading' else 'integrated_yaw'
        np.testing.assert_array_equal(col('feedback_heading'), col(source_name))
    noise = (col('dirty_yaw') - col('raw_yaw') - meta['gyro_bias']
             - rrw_walk(meta, len(data)))
    expected_noise = np.random.default_rng(meta['seed_gyro']).normal(0, meta['noise_std'], len(data))
    np.testing.assert_allclose(noise, expected_noise, rtol=0, atol=1e-12)
    terminated, truncated = bool(np.any(col('terminated_after_step'))), bool(np.any(col('truncated_after_step')))
    full = (len(data) == plan['common']['steps'] and meta['stop_reason'] == 'budget_reached'
            and not terminated and not truncated)
    if meta['stop_reason'] == 'interrupted_or_error':
        raise ValueError('An interrupted process is not a completed experimental trial')
    if not full and not (terminated or truncated):
        raise ValueError('Short run without a recorded termination')
    if terminated or truncated:
        if not (col('terminated_after_step')[-1] or col('truncated_after_step')[-1]):
            raise ValueError('Trial continued after its terminal transition')
    yaw = np.unwrap(col('gt_yaw'))
    relative_yaw = yaw - yaw[0]
    feedback_error = col('pid_feedback_yaw') - relative_yaw
    dx, dy = col('gt_x') - col('gt_x')[0], col('gt_y') - col('gt_y')[0]
    forward, lateral = dx * np.cos(yaw[0]) + dy * np.sin(yaw[0]), -dx * np.sin(yaw[0]) + dy * np.cos(yaw[0])
    control = col('pid_correction')
    step_displacement = np.hypot(np.diff(dx), np.diff(dy))
    row = {'case_id': case['case_id'], 'arm': case['arm'], 'seed': case['seed'],
           'trace_file': str(path.parent.name + '/' + path.name),
           'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
           'samples': len(data), 'duration_s': len(data) * meta['dt'],
           'last_sample_time_s': (len(data) - 1) * meta['dt'],
           'full_duration': full, 'terminated': terminated, 'truncated': truncated,
           'stop_reason': meta['stop_reason'],
           'termination_events': json.loads(meta['termination_events_json']),
           'heading_rms_deg': float(np.rad2deg(np.sqrt(np.mean(relative_yaw**2)))),
           'heading_final_abs_deg': float(abs(np.rad2deg(relative_yaw[-1]))),
           'feedback_rms_deg': float(np.rad2deg(np.sqrt(np.mean(feedback_error**2)))),
           'control_rms_rad_s': float(np.sqrt(np.mean(control**2))),
           'control_peak_rad_s': float(np.max(abs(control))),
           'control_saturation_pct': float(100 * np.mean(abs(control) >= 1.6 - 1e-7)),
           'max_control_step_rad_s': float(np.max(abs(np.diff(control)))) if len(data) > 1 else 0.0,
           'forward_m': float(forward[-1]), 'lateral_abs_m': float(abs(lateral[-1])),
           'path_length_m': float(np.sum(step_displacement)),
           'max_position_step_m': float(np.max(step_displacement)) if len(data) > 1 else 0.0,
           'reference_updates': int(col('reference_updated').sum()),
           'reference_rejections': int(col('reference_rejected').sum()),
           'initial_pose': [float(col(n)[0]) for n in ('gt_yaw', 'gt_x', 'gt_y')],
           'noise_stream_max_error': float(np.max(abs(noise - expected_noise)))}
    series = {'time_s': np.arange(len(data)) * meta['dt'], 'heading_deg': np.rad2deg(relative_yaw),
              'feedback_error_deg': np.rad2deg(feedback_error), 'control': control,
              'forward_m': forward, 'lateral_m': lateral}
    return row, series


def analyze(input_dir):
    plan = json.loads((input_dir / 'protocol.json').read_text(encoding='utf-8'))
    rows, trajectories = [], {}
    for case in plan['cases']:
        run_dir = input_dir / case['case_id']
        record = json.loads((run_dir / 'run_record.json').read_text(encoding='utf-8'))
        if record['returncode'] != 0:
            raise ValueError(f'Process failure: {case["case_id"]}')
        paths = list(run_dir.glob('trace*.npz'))
        if len(paths) != 1:
            raise ValueError(f'Expected one trace: {case["case_id"]}')
        row, series = describe_trace(paths[0], case, plan)
        rows.append(row)
        trajectories[case['case_id']] = series
    groups, pairs = {}, []
    for arm in LABELS:
        subset = [r for r in rows if r['arm'] == arm]
        completed = [r for r in subset if r['full_duration']]
        groups[arm] = {'planned_n': len(subset), 'full_duration_n': len(completed),
            'terminated_n': sum(r['terminated'] for r in subset),
            'truncated_n': sum(r['truncated'] for r in subset), 'completed_only_metrics': {}}
        for metric in METRICS:
            values = [r[metric] for r in completed]
            groups[arm]['completed_only_metrics'][metric] = {
                'mean': float(np.mean(values)) if values else None,
                'sd': float(np.std(values, ddof=1)) if len(values) > 1 else None}
    for seed in sorted({r['seed'] for r in rows}):
        by_arm = {r['arm']: r for r in rows if r['seed'] == seed}
        poses = [r['initial_pose'] for r in by_arm.values()]
        np.testing.assert_allclose(poses, np.broadcast_to(poses[0], (3, 3)), rtol=0, atol=1e-9)
        b, c = by_arm['B'], by_arm['C']
        if b['full_duration'] and c['full_duration']:
            pairs.append({'seed': seed, **{metric + '_C_minus_B': c[metric] - b[metric] for metric in METRICS}})
    return {'kind': 'pilot_descriptive_analysis', 'rows': rows, 'groups': groups, 'paired_C_minus_B': pairs,
            'checks': {'configuration_and_code_identity': True, 'same_initial_pose_by_seed': True,
                       'independent_gyro_stream_replayed_exactly': True, 'single_segment_per_trial': True,
                       'actual_pid_feedback_delay_and_source': True},
            'sampling_note': plan['sampling_note'], 'analysis_rule': plan['analysis_rule']}, trajectories


def make_plots(output, result, trajectories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    seeds = sorted({r['seed'] for r in result['rows']})
    colors = {'A': '#6b7280', 'B': '#dc7d25', 'C': '#2079b0'}
    fig, axes = plt.subplots(len(seeds), 3, figsize=(13, 8.5), sharex=True)
    for i, seed in enumerate(seeds):
        for arm in LABELS:
            curve = trajectories[f'{arm}_seed{seed}']
            for j, field in enumerate(('heading_deg', 'feedback_error_deg', 'control')):
                axes[i, j].plot(curve['time_s'], curve[field], color=colors[arm], lw=1, label=arm)
                axes[i, j].grid(alpha=.2)
                axes[i, j].set_xlabel('Time (s)')
        axes[i, 0].set_ylabel(f'Seed {seed}\nHeading deviation (deg)')
        axes[i, 1].set_ylabel('PID estimate error (deg)')
        axes[i, 2].set_ylabel('Yaw command (rad/s)')
    axes[0, 0].set_title('Physical heading')
    axes[0, 1].set_title('Actual feedback vs physical heading')
    axes[0, 2].set_title('Controller output')
    axes[0, 0].legend()
    fig.suptitle('Closed-loop pilot | A: noisy integral  B: bias-corrected integral  C: fused heading', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(output / 'heading_pilot_traces.png', dpi=170)
    fig.savefig(output / 'heading_pilot_traces.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9))
    for ax, metric, label in zip(axes, ('heading_rms_deg', 'feedback_rms_deg', 'control_rms_rad_s'),
                                ('Physical heading RMS (deg)', 'PID estimate error RMS (deg)', 'Yaw command RMS (rad/s)')):
        for seed in seeds:
            vals = [next(r for r in result['rows'] if r['arm'] == arm and r['seed'] == seed) for arm in LABELS]
            if all(r['full_duration'] for r in vals):
                ax.plot(range(3), [r[metric] for r in vals], '-o', lw=1, ms=5, label=f'Seed {seed}')
        ax.set_xticks(range(3), list(LABELS))
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Individual paired outcomes | n = 3 per arm | completed trials only', fontsize=12)
    fig.tight_layout()
    fig.savefig(output / 'heading_pilot_paired.png', dpi=180)
    fig.savefig(output / 'heading_pilot_paired.pdf')
    plt.close(fig)


def write_report(output, result):
    lines = ['# 航向反馈闭环预实验', '',
        '三种接法、每种三个固定种子；每次计划 3000 步（60 s）。策略固定为 model_12000.pt，未重新训练。', '',
        'A：加噪角速度积分；B：融合器补偿零偏后独立积分；C：融合航向直接反馈。B、C 使用相同的角速度处理与参考设置。', '',
        '共同设置：平地、固定机体、地面摩擦配置 0.8、前进指令 0.5 m/s、Kp=0.91、Ki=0、Kd=0.01；'
        '陀螺噪声 0.15 rad/s、零偏 0.02 rad/s；参考标准差 0.05 rad、1 Hz、无丢包。', '',
        '## 每次运行', '',
        '| 方案 | 种子 | 时长 s | 结束原因 | 真实航向 RMS ° | 最终绝对航向偏差 ° | 反馈误差 RMS ° | 控制 RMS rad/s | 前进 m |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in sorted(result['rows'], key=lambda r: (r['arm'], r['seed'])):
        lines.append(f'| {r["arm"]} | {r["seed"]} | {r["duration_s"]:.2f} | {r["stop_reason"]} | '
                     f'{r["heading_rms_deg"]:.3f} | {r["heading_final_abs_deg"]:.3f} | {r["feedback_rms_deg"]:.3f} | '
                     f'{r["control_rms_rad_s"]:.4f} | {r["forward_m"]:.3f} |')
    lines += ['', '## 完成 60 秒的运行：均值 ± 样本标准差', '',
        '| 方案 | 完整运行/计划 | 终止/超时 | 真实航向 RMS ° | 反馈误差 RMS ° | 控制 RMS rad/s | 前进 m |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm, group in result['groups'].items():
        cells = []
        for key in ('heading_rms_deg', 'feedback_rms_deg', 'control_rms_rad_s', 'forward_m'):
            m = group['completed_only_metrics'][key]
            cells.append('—' if m['mean'] is None else f'{m["mean"]:.4f} ± {m["sd"]:.4f}' if m['sd'] is not None else f'{m["mean"]:.4f}（SD未定义）')
        lines.append(f'| {arm} | {group["full_duration_n"]}/{group["planned_n"]} | {group["terminated_n"]}/{group["truncated_n"]} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 按种子配对：C 减 B', '',
        '| 种子 | 真实航向 RMS 差 ° | 反馈误差 RMS 差 ° | 控制 RMS 差 rad/s | 前进距离差 m |',
        '|---|---:|---:|---:|---:|']
    for pair in result['paired_C_minus_B']:
        lines.append(f'| {pair["seed"]} | {pair["heading_rms_deg_C_minus_B"]:+.4f} | '
                     f'{pair["feedback_rms_deg_C_minus_B"]:+.4f} | {pair["control_rms_rad_s_C_minus_B"]:+.4f} | '
                     f'{pair["forward_m_C_minus_B"]:+.4f} |')
    lines += ['', '前两项为负表示 C 在该次配对中的误差更小；控制量变化需结合运动表现解释。', '',
        '## 核对与适用范围', '',
        '- 配置、权重/源码哈希、种子、实际 PID 反馈来源与一步延迟均通过检查。同种子初始位姿一致，注入陀螺噪声可由保存种子逐点复现。',
        '- 启用现有 fell_over（机体倾斜超过 70°）和 time_out 条件，首次终止即结束该次。没有用重生片段补齐时长。该判据不能覆盖所有步态退化。',
        '- 每行是物理步之前的采样；3000 步对应运行 60 s，最后一行的采样时刻是 59.98 s。终止后的物理状态未混入轨迹。',
        '- 参考是仿真航向真值相对起点的变化叠加高斯噪声；本轮没有验证真实传感器。',
        '- n=3 只作描述性预实验，不做显著性、等效性或普适改进结论。提前终止记录完整保留，完整时长组统计单独列出。', '',
        '![按种子配对的结果](heading_pilot_paired.png)', '',
        '![全程轨迹](heading_pilot_traces.png)', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.input_dir / 'analysis'
    result, trajectories = analyze(args.input_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    make_plots(output, result, trajectories)
    write_report(output, result)
    print(json.dumps(result['groups'], ensure_ascii=False, indent=2))
    print(f'Report: {(output / "report.md").resolve()}')


if __name__ == '__main__':
    main()
