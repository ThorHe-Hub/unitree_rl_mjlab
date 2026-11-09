"""Analyze the preregistered 30-trial zero-bias batch, including stability logs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from analyze_heading_pilot import describe_trace, LABELS, METRICS as HEADING_METRICS

STABILITY_METRICS = ['tilt_rms_deg', 'tilt_peak_deg', 'tilt_p95_deg',
                     'height_min_m', 'height_mean_m', 'height_std_m', 'height_range_m']
METRICS = [*HEADING_METRICS, *STABILITY_METRICS]


def paired_statistics(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite paired outcomes')
    n = len(values)
    if n == 0:
        return {'n': 0, 'mean': None, 'sd': None, 'ci95_t': None, 'c_lower_n': 0}
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else None
    half = float(student_t.ppf(.975, n-1) * sd / np.sqrt(n)) if n > 1 else None
    return {'n': n, 'mean': mean, 'sd': sd, 'ci95_t': [mean-half, mean+half] if half is not None else None,
            'c_lower_n': int(np.sum(values < 0))}


def read_trial(path, case, plan):
    row, series = describe_trace(path, case, plan)
    with np.load(path, allow_pickle=False) as z:
        config = json.loads(z['config_json'].item())
        for name in ('fusion_q_bias', 'fusion_sigma_bias_prior', 'abs_outage_start_s', 'abs_outage_duration_s'):
            if name in plan['common']:
                expected = plan['common'][name]
                if name not in z or z[name].item() != expected or config.get(name) != expected:
                    raise ValueError(f'Unexpected fusion parameter: {name}')
        c = {str(name): i for i, name in enumerate(z['columns'])}
        # Schemas 5 and 6 only append columns; every column read below exists in all three.
        schema = z['trace_schema_version'].item()
        width = {4: 43, 5: 47, 6: 49}.get(schema)
        if width is None:
            raise ValueError('This batch requires schema 4-6 stability records')
        if len(c) != width or z['data'].shape != (row['samples'], width):
            raise ValueError(f'Expected {width} distinct schema {schema} columns')
        for key, expected in {'task_id': plan['common']['task_id'],
                              'requested_steps': plan['common']['steps'],
                              'completed_steps': row['samples'],
                              'completed_duration_s': row['duration_s']}.items():
            if z[key].item() != expected:
                raise ValueError(f'Unexpected trial metadata: {key}')
        if z['stability_timing'].item() != 'pre_physics_step_same_as_gt_yaw':
            raise ValueError('Unexpected stability sampling time')
        if z['base_tilt_definition'].item() != 'angle_between_root_link_z_and_world_vertical_rad' or z['base_height_definition'].item() != 'root_link_world_z_m_not_total_robot_height':
            raise ValueError('Unexpected stability definition')
        for name in ('terminated_after_step', 'truncated_after_step'):
            flags = z['data'][:, c[name]]
            if not np.isin(flags, [0, 1]).all() or np.any(flags[:-1]):
                raise ValueError('Invalid terminal flags or continued after done')
        from pinned_input_audit import audit_inputs
        row.update(audit_inputs(z, plan['common']))
        tilt = z['data'][:, c['base_tilt']]
        height = z['data'][:, c['base_height_w']]
    if not np.isfinite(tilt).all() or not np.isfinite(height).all() or np.any((tilt < 0) | (tilt > np.pi)):
        raise ValueError('Invalid body stability samples')
    deg = np.rad2deg(tilt)
    row.update(tilt_rms_deg=float(np.sqrt(np.mean(deg**2))), tilt_peak_deg=float(deg.max()),
               tilt_p95_deg=float(np.percentile(deg, 95)), height_min_m=float(height.min()),
               height_mean_m=float(height.mean()), height_std_m=float(height.std()),
               height_range_m=float(np.ptp(height)), initial_height_m=float(height[0]))
    series.update(tilt_deg=deg.copy(), height_m=height.copy())
    return row, series


def analyze(root, partial=False):
    plan = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
    if plan['common']['gyro_bias'] != 0 or len(plan['cases']) != 30:
        raise ValueError('Expected the fixed zero-bias 30-trial protocol')
    rows, trajectories = [], {}
    for case in plan['cases']:
        directory = root / case['case_id']
        record_file = directory / 'run_record.json'
        if partial and not record_file.exists():
            continue
        record = json.loads(record_file.read_text(encoding='utf-8'))
        if record['returncode'] != 0:
            raise ValueError(f'Infrastructure failure: {case["case_id"]}')
        paths = list(directory.glob('trace*.npz'))
        if len(paths) != 1:
            raise ValueError(f'Expected one trace: {case["case_id"]}')
        row, series = read_trial(paths[0], case, plan)
        if record['trace_sha256'] != row['sha256']:
            raise ValueError(f'Trace bytes changed: {case["case_id"]}')
        rows.append(row)
        trajectories[case['case_id']] = series
    groups, pairs = {}, []
    for arm in LABELS:
        subset = [row for row in rows if row['arm'] == arm]
        completed = [row for row in subset if row['full_duration']]
        groups[arm] = {'planned_n': 10, 'recorded_n': len(subset), 'full_duration_n': len(completed),
                       'terminated_n': sum(r['terminated'] for r in subset),
                       'truncated_n': sum(r['truncated'] for r in subset), 'completed_only_metrics': {}}
        for metric in METRICS:
            values = [r[metric] for r in completed]
            groups[arm]['completed_only_metrics'][metric] = {
                'mean': float(np.mean(values)) if values else None,
                'sd': float(np.std(values, ddof=1)) if len(values) > 1 else None}
    for seed in sorted({r['seed'] for r in rows}):
        by_arm = {r['arm']: r for r in rows if r['seed'] == seed}
        poses = [r['initial_pose'] + [r['initial_height_m']] for r in by_arm.values()]
        np.testing.assert_allclose(poses, np.broadcast_to(poses[0], (len(poses), 4)), rtol=0, atol=1e-9)
        if 'B' in by_arm and 'C' in by_arm and by_arm['B']['full_duration'] and by_arm['C']['full_duration']:
            pairs.append({'seed': seed, **{metric: by_arm['C'][metric] - by_arm['B'][metric] for metric in METRICS}})
    paired = {metric: paired_statistics([p[metric] for p in pairs]) for metric in METRICS}
    return {'kind': 'zero_bias_fixed_input_rerun_analysis', 'complete': len(rows) == 30,
        'protocol_sha256': hashlib.sha256((root / 'protocol.json').read_bytes()).hexdigest(),
        'rows': rows, 'groups': groups, 'paired_C_minus_B': pairs, 'paired_statistics': paired,
        'primary_outcome': plan['primary_outcome'], 'analysis_rule': plan['analysis_rule'],
        'checks': {'source_command_phase_consistency_all_rows': True, 'configuration_and_code_identity': True, 'same_initial_pose_and_height_by_seed': True,
                   'independent_gyro_stream_replayed': True, 'single_segment_per_trial': True,
                   'actual_pid_feedback_source_and_delay': True, 'stability_schema_timing_and_finiteness': True}}, trajectories


def plots(output, result, trajectories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    seeds = sorted({r['seed'] for r in result['rows']})
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    specs = [('heading_rms_deg', 'Physical heading RMS (deg)'), ('feedback_rms_deg', 'Feedback error RMS (deg)'),
             ('control_rms_rad_s', 'Yaw command RMS (rad/s)'), ('tilt_p95_deg', '95th percentile tilt (deg)'),
             ('height_min_m', 'Minimum root-link world z (m)'), ('forward_m', 'Forward displacement (m)')]
    for ax, (metric, label) in zip(axes.flat, specs):
        for seed in seeds:
            vals = [next(r for r in result['rows'] if r['seed'] == seed and r['arm'] == arm) for arm in LABELS]
            if all(r['full_duration'] for r in vals):
                ax.plot(range(3), [r[metric] for r in vals], '-o', lw=.9, ms=4, alpha=.85, label=str(seed))
        ax.set_xticks(range(3), list(LABELS))
        ax.set_ylabel(label)
        ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(title='Seed', ncol=2, fontsize=8)
    fig.suptitle('Zero-bias batch | 10 paired rerun seeds per arm | paired individual outcomes', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(output / 'zero_bias_paired.png', dpi=170)
    fig.savefig(output / 'zero_bias_paired.pdf')
    plt.close(fig)
    colors = {'A': '#6b7280', 'B': '#dc7d25', 'C': '#2079b0'}
    # Show every seed in two pages, with consistent arm colors and no selection of representative successes.
    for page in range(2):
        subset = seeds[page*5:(page+1)*5]
        fig, axes = plt.subplots(len(subset), 3, figsize=(13, 12), squeeze=False, sharex=True)
        for i, seed in enumerate(subset):
            for arm in LABELS:
                curve = trajectories[f'{arm}_seed{seed}']
                for j, field in enumerate(('heading_deg', 'tilt_deg', 'height_m')):
                    axes[i, j].plot(curve['time_s'], curve[field], lw=.85, color=colors[arm], label=arm)
                    axes[i, j].grid(alpha=.2)
            axes[i, 0].set_ylabel(f'Seed {seed}\nHeading (deg)')
            axes[i, 1].set_ylabel('Body tilt (deg)')
            axes[i, 2].set_ylabel('Root-link z (m)')
        for ax in axes[-1]:
            ax.set_xlabel('Time (s)')
        axes[0, 0].legend(ncol=3)
        fig.suptitle(f'Zero-bias trajectories, page {page+1}/2 | A noisy integral, B corrected integral, C fused heading', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .98))
        fig.savefig(output / f'zero_bias_traces_{page+1}.png', dpi=160)
        fig.savefig(output / f'zero_bias_traces_{page+1}.pdf')
        plt.close(fig)


def report(output, result):
    lines = ['# 固定指令输入修复后的零偏复验', '',
        '三个方案各使用 24000–24009 共十个与旧批次相同的种子，每次计划 3000 步（60 s）。预实验的三个种子未纳入本批。', '',
        'A：加噪后直接积分；B：补偿零偏后独立积分；C：融合航向直接反馈。策略为 model_12000.pt，参数固定。'
        '陀螺白噪声标准差 0.15 rad/s，注入零偏为 0；合成航向参考标准差 0.05 rad、1 Hz、无丢包。', '',
        '## 完整时长运行的组统计', '',
        '下表为均值 ± 样本标准差；每个种子权重相同。倾角为根链接 z 轴偏离世界竖直方向的夹角，高度为根链接世界 z 坐标，并非机器人总身高。', '',
        '| 方案 | 完成/计划 | 终止/超时 | 航向 RMS ° | 反馈误差 RMS ° | 纠偏指令 RMS rad/s | 倾角 P95 ° | 最低高度 m |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm, g in result['groups'].items():
        cells = []
        for metric in ('heading_rms_deg', 'feedback_rms_deg', 'control_rms_rad_s', 'tilt_p95_deg', 'height_min_m'):
            v = g['completed_only_metrics'][metric]
            cells.append('—' if v['mean'] is None else f'{v["mean"]:.4f} ± {v["sd"]:.4f}' if v['sd'] is not None else f'{v["mean"]:.4f}（SD未定义）')
        lines.append(f'| {arm} | {g["full_duration_n"]}/10 | {g["terminated_n"]}/{g["truncated_n"]} | ' + ' | '.join(cells) + ' |')
    primary = result['paired_statistics']['heading_rms_deg']
    lines += ['', '## 预先指定的主要比较：C 减 B', '',
        f'完整 60 s 的有效配对数为 {primary["n"]}，其中 {primary["c_lower_n"]} 个种子的 C 航向 RMS 更低。']
    if primary['ci95_t'] is not None:
        lo, hi = primary['ci95_t']
        lines += [f'配对差值均值为 {primary["mean"]:+.4f}°；95% 配对 t 区间为 [{lo:+.4f}°, {hi:+.4f}°]。']
    lines += ['该区间假定不同种子的配对差值独立，并采用小样本 t 近似。其余指标和比较作描述性分析，未作多重检验后的显著性声明。', '',
        '| 种子 | 航向 RMS 差 ° | 反馈误差 RMS 差 ° | 纠偏指令 RMS 差 rad/s | 倾角 P95 差 ° | 最低高度差 m |',
        '|---|---:|---:|---:|---:|---:|']
    for p in result['paired_C_minus_B']:
        lines.append(f'| {p["seed"]} | {p["heading_rms_deg"]:+.4f} | {p["feedback_rms_deg"]:+.4f} | '
                     f'{p["control_rms_rad_s"]:+.4f} | {p["tilt_p95_deg"]:+.4f} | {p["height_min_m"]:+.4f} |')
    lines += ['', '## 全部运行记录', '',
        '| 方案 | 种子 | 时长 s | 结束原因 | 航向 RMS ° | 前进 m | 倾角峰值 ° | 最低高度 m |',
        '|---|---:|---:|---|---:|---:|---:|---:|']
    for r in sorted(result['rows'], key=lambda r: (r['arm'], r['seed'])):
        lines.append(f'| {r["arm"]} | {r["seed"]} | {r["duration_s"]:.2f} | {r["stop_reason"]} | '
                     f'{r["heading_rms_deg"]:.3f} | {r["forward_m"]:.3f} | {r["tilt_peak_deg"]:.3f} | {r["height_min_m"]:.4f} |')
    lines += ['', '## 核对和适用范围', '',
        '- 全部样本逐步核对观测入口、策略输入、环境指令、站立标志和 0.6 s 周期相位；固定前进期间不允许零相位。',
        '- 权重、源码、配置、独立噪声种子、同种子初始位姿/高度、实际反馈来源与一步延迟均已核对。',
        '- 所有终止保留，首次终止即结束该次。若发生失败，完整时长配对的结果仅适用于双方完成的运行，不能单独代表总体优劣。',
        '- 稳定性与航向真值同为物理步之前的采样；60 s 运行的末个样本时刻为 59.98 s。终止转换之后的状态不混入轨迹。',
        '- 倾角、高度和未触发 70° 跌倒判据是有限的稳定性证据，不能证明所有步态品质等效。纠偏指令大小不是能耗。',
        '- 零偏为 0 指没有额外注入常值陀螺零偏，并不排除机器人动力学、坐标变换和其他估计误差。参考仍为合成观测。',
        '- 本批只包含零偏为 0 的 30 次运行；0.02 与 0.04 rad/s 的正式批次另行安排。', '',
        '![配对结果](zero_bias_paired.png)', '',
        '![全程轨迹 1](zero_bias_traces_1.png)', '',
        '![全程轨迹 2](zero_bias_traces_2.png)', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('--partial', action='store_true', help='Read-only progress audit, without final files or plots')
    args = parser.parse_args()
    result, trajectories = analyze(args.input_dir, args.partial)
    if args.partial:
        print(json.dumps({'recorded': len(result['rows']), 'rows': [{k: r[k] for k in ('case_id', 'stop_reason', 'heading_rms_deg', 'tilt_peak_deg', 'height_min_m')} for r in result['rows']]}, indent=2))
        return
    output = args.input_dir / 'analysis'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    plots(output, result, trajectories)
    report(output, result)
    print(json.dumps({'groups': result['groups'], 'primary': result['paired_statistics']['heading_rms_deg']}, indent=2))
    print(f'Report: {(output / "report.md").resolve()}')


if __name__ == '__main__':
    main()
