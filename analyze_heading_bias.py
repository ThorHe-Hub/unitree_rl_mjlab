"""Audit and compare the existing zero-bias baseline with user-run .02/.04 batches."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from analyze_heading_fixed_input import analyze as analyze_baseline
from heading_bias_batch import BASELINE, BASELINE_SHA256, load_plan, inspect_case, summarize, write_json, METRICS
from run_heading_fixed_input import digest

PROJECT = Path(__file__).resolve().parent
RESULTS = PROJECT/'unitree_heading_results'


def check_comparable(base, other):
    for key in ('cases','input_contract','trace_schema_version','stability_definition','dependency_versions','analysis_rule'):
        if other[key] != base[key]:
            raise ValueError(f'Incomparable batch: {key}')
    for key in ('source_tree_sha256','checkpoint_sha256'):
        if other['identity'][key] != base['identity'][key]:
            raise ValueError(f'Incomparable batch: {key}')
    a,b = dict(base['common']),dict(other['common'])
    a.pop('gyro_bias'); b.pop('gyro_bias')
    if a != b:
        raise ValueError('Non-bias experiment parameters differ')
    if other.get('baseline_protocol_sha256') != BASELINE_SHA256:
        raise ValueError('Wrong baseline provenance')


def collect_batch(root, plan, partial=False):
    rows, trajectories, attempts = [], {}, []
    for case in plan['cases']:
        directory = root/case['case_id']
        for path in sorted(directory.glob('attempt_*')):
            record = path/'run_record.json'
            meta = json.loads(record.read_text(encoding='utf-8')) if record.exists() else {}
            attempts.append(dict(case_id=case['case_id'], attempt=path.name,
                status=meta.get('status','unfinished'), returncode=meta.get('returncode')))
        if not (directory/'completed.json').exists():
            if partial:
                continue
            raise ValueError(f'Case is not finalized: {directory}; run the launcher with --resume')
        result = inspect_case(directory, case, plan)
        if result is None:
            raise ValueError(f'Invalid finalized case: {directory}')
        row, curve = result
        rows.append(row)
        trajectories[case['case_id']] = curve
    result = summarize(rows, plan)
    result.update(protocol_sha256=digest(root/'protocol.json'), attempts=attempts)
    return result, trajectories


def analyze(roots, partial=False):
    if digest(roots[0]/'protocol.json') != BASELINE_SHA256:
        raise ValueError('Expected the reviewed fixed-input zero-bias baseline')
    base = load_plan(roots[0])
    baseline, curves = analyze_baseline(roots[0])
    baseline.update(gyro_bias=0., attempts=[])
    results, trajectories, missing = [baseline], {0.:curves}, []
    for bias, root in zip((.02,.04), roots[1:]):
        if partial and not (root/'protocol.json').exists():
            missing.append(dict(gyro_bias=bias, path=str(root)))
            continue
        plan = load_plan(root)
        if plan['common']['gyro_bias'] != bias:
            raise ValueError(f'Wrong bias directory: expected {bias}')
        check_comparable(base, plan)
        result, curves = collect_batch(root, plan, partial)
        results.append(result)
        trajectories[bias] = curves
    # Pose/height equality across arms AND across bias levels, for every recorded seed.
    poses = {}
    for result in results:
        for row in result['rows']:
            pose = row['initial_pose']+[row['initial_height_m']]
            np.testing.assert_allclose(pose, poses.setdefault(row['seed'],pose), atol=1e-9, rtol=0)
    return dict(kind='three_level_heading_bias_comparison', complete=len(results)==3 and all(r['complete'] for r in results),
        batches=results, missing_batches=missing,
        checks=dict(core_code_checkpoint_nonbias_parameters_seeds_and_dependency_versions_match=True,
            same_logged_initial_pose_and_height_across_bias_levels=True,
            protocol_and_source_archives_verified=True, actual_inputs_noise_streams_and_feedback_checked=True)), trajectories


def fmt(value):
    return '—' if value is None else f'{value:.4f}'


def save_report(out, result):
    write_json(out/'summary.json',result)
    fields=['gyro_bias','arm','seed','full_duration','stop_reason','samples','duration_s',*METRICS,'trace_file','sha256']
    with (out/'trials.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for batch in result['batches']:
            for row in sorted(batch['rows'],key=lambda r:(r['arm'],r['seed'])):
                writer.writerow({k:batch['gyro_bias'] if k=='gyro_bias' else row[k] for k in fields})
    lines=['# 三档常值零偏对照', '',
        'A：加噪陀螺积分；B：补偿零偏后独立积分；C：融合航向反馈。'
        '0 rad/s 使用已完成的输入修复版本；0.02/0.04 rad/s 使用相同核心代码、model_12000、参数与种子。', '',
        '## 完成率和完整时长指标', '',
        '均值 ± 样本 SD 仅由完整 60 s 的运行计算。所有跌倒、超时和中断记录保留；若有失败，完整时长指标不能单独代表总体表现。', '',
        '| 零偏 rad/s | 方案 | 完成/计划 | 跌倒/超时 | 航向 RMS ° | 指令 RMS rad/s | 最大单步变化 rad/s | 倾角峰值 ° |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for batch in result['batches']:
        for arm,g in batch['groups'].items():
            cells=[]
            for metric in ('heading_rms_deg','control_rms_rad_s','max_control_step_rad_s','tilt_peak_deg'):
                v=g['completed_only_metrics'][metric]
                cells.append(f'{fmt(v["mean"])} ± {fmt(v["sd"])}')
            lines.append(f'| {batch["gyro_bias"]:.2f} | {arm} | {g["full_duration_n"]}/10 | {g["terminated_n"]}/{g["truncated_n"]} | '+' | '.join(cells)+' |')
    lines+=['', '## 主要比较：各零偏档位内 C 减 B', '',
        '| 零偏 rad/s | 完整配对数 | C 更低的种子数 | 航向 RMS 差均值 ° | 样本 SD ° | 95% 配对 t 区间 ° |',
        '|---:|---:|---:|---:|---:|---|']
    for batch in result['batches']:
        p=batch['paired_statistics']['heading_rms_deg']
        ci='—' if p['ci95_t'] is None else f'[{p["ci95_t"][0]:+.4f}, {p["ci95_t"][1]:+.4f}]'
        lines.append(f'| {batch["gyro_bias"]:.2f} | {p["n"]} | {p["c_lower_n"]} | {fmt(p["mean"])} | {fmt(p["sd"])} | {ci} |')
    lines+=['', '区间采用各档内部独立种子差值的小样本 t 近似。三档复用同一组种子，不能合并成 30 个独立种子；'
        '三个区间是分别计算的边际区间，不作联合多重检验显著性声明。其余指标和跨零偏变化作描述分析。', '',
        '## 运行与解释边界', '',
        '原始数据均按物理步前采样，末个样本为 59.98 s。倾角和根链接世界高度不是完整步态质量评价；控制指令大小不等于能耗。'
        '航向参考为合成带噪观测，结论限于当前仿真条件。', '',
        '全部逐次指标见 trials.csv，所有运行尝试状态与配对差值见 summary.json。'
        '每个新批次目录保留协议、源码快照、completed.json 和所有 attempt 子目录。', '',
        '![三档概览](overview.png)', '']
    for bias in (0.,.02,.04):
        for page in (1,2):
            name=f'trajectories_b{int(round(bias*100)):03d}_{page}.png'
            lines += [f'![零偏 {bias:.2f} 全部轨迹 {page}]({name})','']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def plots(out,result,trajectories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(A='#6b7280',B='#dc7d25',C='#2079b0')
    biases=[b['gyro_bias'] for b in result['batches']]
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    specs=[('heading_rms_deg','Heading RMS (deg)'),('control_rms_rad_s','Yaw command RMS (rad/s)'),
           ('max_control_step_rad_s','Max command step (rad/s)'),('tilt_peak_deg','Peak root tilt (deg)'),
           ('height_min_m','Minimum root z (m)')]
    for ax,(metric,label) in zip(axes.flat,specs):
        for arm,color in colors.items():
            vals=[b['groups'][arm]['completed_only_metrics'][metric] for b in result['batches']]
            ax.errorbar(biases,[v['mean'] if v['mean'] is not None else np.nan for v in vals],
                yerr=[v['sd'] or 0 for v in vals],label=arm,color=color,fmt='-o',capsize=3)
        ax.set_ylabel(label); ax.set_xlabel('Injected bias (rad/s)'); ax.set_xticks(biases); ax.grid(alpha=.2)
    ax=axes.flat[-1]
    for seed in range(24000,24010):
        values=[next((p['heading_rms_deg'] for p in b['paired_C_minus_B'] if p['seed']==seed),np.nan) for b in result['batches']]
        ax.plot(biases,values,'-o',lw=.8,ms=3,label=str(seed))
    ax.axhline(0,color='black',lw=.8); ax.set_ylabel('Paired C minus B heading RMS (deg)')
    ax.set_xlabel('Injected bias (rad/s)'); ax.set_xticks(biases); ax.grid(alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.95))
    fig.suptitle('Three bias levels | mean and sample SD of completed trials | failures reported separately')
    fig.tight_layout(rect=(0,0,1,.91))
    fig.savefig(out/'overview.png',dpi=170); fig.savefig(out/'overview.pdf'); plt.close(fig)
    for batch in result['batches']:
        bias=batch['gyro_bias']
        for page in range(2):
            fig,axes=plt.subplots(5,3,figsize=(13,12),sharex=True)
            for i,seed in enumerate(range(24000+page*5,24005+page*5)):
                for arm,color in colors.items():
                    curve=trajectories[bias][f'{arm}_seed{seed}']
                    for j,key in enumerate(('heading_deg','tilt_deg','height_m')):
                        axes[i,j].plot(curve['time_s'],curve[key],color=color,lw=.8,label=arm)
                        axes[i,j].grid(alpha=.2)
                for j,label in enumerate(('Heading (deg)','Root tilt (deg)','Root z (m)')):
                    axes[i,j].set_ylabel(f'{seed}\n{label}')
            for ax in axes[-1]: ax.set_xlabel('Time (s)')
            handles,labels=axes[0,0].get_legend_handles_labels()
            fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.965))
            fig.suptitle(f'Bias {bias:.2f} rad/s | all trajectories, page {page+1}/2 | short failures retained')
            fig.tight_layout(rect=(0,0,1,.94))
            name=f'trajectories_b{int(round(bias*100)):03d}_{page+1}'
            fig.savefig(out/(name+'.png'),dpi=150); fig.savefig(out/(name+'.pdf')); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir',type=Path,default=PROJECT/BASELINE)
    parser.add_argument('--bias-002-dir',type=Path,default=RESULTS/'bias_002')
    parser.add_argument('--bias-004-dir',type=Path,default=RESULTS/'bias_004')
    parser.add_argument('--output-dir',type=Path,default=RESULTS/'comparison')
    parser.add_argument('--partial',action='store_true',help='Read-only progress; missing batches allowed; no final report')
    args=parser.parse_args()
    roots=[p.resolve() for p in (args.baseline_dir,args.bias_002_dir,args.bias_004_dir)]
    result,trajectories=analyze(roots,args.partial)
    if args.partial:
        print(json.dumps(dict(batches=[dict(gyro_bias=b['gyro_bias'],recorded=len(b['rows']),
            full_duration=sum(r['full_duration'] for r in b['rows']),
            terminated=sum(r['terminated'] for r in b['rows'])) for b in result['batches']],
            missing_batches=result['missing_batches']),indent=2))
        return
    out=args.output_dir.resolve()
    if any(out==r or out.is_relative_to(r) for r in roots):
        raise ValueError('Write comparison output outside the original batch directories')
    out.mkdir(parents=True,exist_ok=True)
    plots(out,result,trajectories)
    save_report(out,result)
    print(f'[COMPLETE] 90 trial outcomes audited. Report: {out/"report.md"}')


if __name__=='__main__':
    main()
