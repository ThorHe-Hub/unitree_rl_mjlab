"""轨迹分组统计与审计（仅依赖 numpy）。

python agg.py                         # 根目录与 traces/，历史 3000 步非激励批次
python agg.py --cohort identification # 开环辨识，按激励种子分别统计
python agg.py --cohort legacy         # 早期轨迹，按精确步数分别统计
python agg.py --cohort recorded       # 新记录格式，按反馈方式和代码/权重版本分组
python agg.py --cohort all            # 全部轨迹，不同批次不合并
python agg.py --audit-dir build/data_audit

main 是历史数据的筛选规则，不代表已核验 fixed_body、pin_cmd 或 PID 状态。
主表 ± 为样本标准差；2SE 只保留原稿的探索性判据，不作等效性判定。
"""
import argparse
import glob
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MET = ["|航向净变化| rad", "|横向漂移| m", "前进 m", "路径长度 m",
       "位置跳变次数", "|积分器偏差| rad", "PID修正 RMS",
       "|实际反馈误差| rad", "实际反馈误差 RMS rad"]
PRIMARY, SECONDARY = 5, 0
DIMS = ["策略", "摩擦", "白噪声", "零偏", "闭环", "低通Hz", "τ", "参考σ",
        "参考Hz", "丢包", "Kp", "Ki", "vx", "批次", "步数", "dt",
        "激励幅值", "激励种子", "固定机体", "固定指令", "任务", "Kd", "反馈方式",
        "权重SHA256", "源代码SHA256", "记录版本", "参考延迟", "反馈时序"]
FILT_DIM = DIMS.index("闭环")
# 为复现旧表保留这些历史约定；每次补值都在 manifest 中记录，不能当作实测元数据。
DEFAULTS = {"gyro_bias": 0.0, "filt": "akf", "lpf_hz": 0.0, "bias_tau": 10.0,
            "abs_sigma": 0.05, "abs_hz": 1.0, "abs_dropout": 0.0,
            "kp": 0.91, "ki": 0.0, "pin_vx": 0.5}
AUDIT_FIELDS = ["fixed_body", "pin_cmd", "seed", "task_id", "kd", "feedback_mode",
                "sweep_vyaw", "sweep_seed", "checkpoint_sha256", "source_tree_sha256", *DEFAULTS]
REQUIRED = ["gt_yaw", "gt_x", "gt_y", "integrated_yaw", "pid_correction"]


def discover(patterns):
    """裸文件名模式同时搜索两种布局；显式路径严格按用户输入搜索。"""
    found = set()
    for pattern in patterns or ["trace*.npz"]:
        candidates = glob.glob(pattern)
        if Path(pattern).parent == Path("."):
            candidates += glob.glob(str(Path("traces") / pattern))
        found.update(Path(p).resolve() for p in candidates if Path(p).is_file())
    return sorted(found)


def metrics(d, columns):
    c = {str(name): i for i, name in enumerate(columns)}
    if len(c) != len(columns) or any(name not in c for name in REQUIRED):
        raise ValueError("列名重复或缺少评价指标所需列")
    if d.ndim != 2 or len(d) < 2 or d.shape[1] != len(columns):
        raise ValueError("轨迹形状不合法或不足两步")
    if not np.isfinite(d[:, [c[n] for n in REQUIRED]]).all():
        raise ValueError("评价指标所需列含 NaN/Inf")
    if "segment_id" in c and (not np.isfinite(d[:, c["segment_id"]]).all()
                               or len(np.unique(d[:, c["segment_id"]])) != 1):
        raise ValueError("轨迹包含自动重置的多个区段，不能跨区段计算位移/误差；请先按 segment_id 拆分")
    gt = np.unwrap(d[:, c["gt_yaw"]])
    psi0 = gt[0]
    dx = d[-1, c["gt_x"]] - d[0, c["gt_x"]]
    dy = d[-1, c["gt_y"]] - d[0, c["gt_y"]]
    lat = -dx * np.sin(psi0) + dy * np.cos(psi0)
    fwd = dx * np.cos(psi0) + dy * np.sin(psi0)
    ierr = d[:, c["integrated_yaw"]] - (gt - gt[0]) - d[0, c["integrated_yaw"]]
    step = np.hypot(np.diff(d[:, c["gt_x"]]), np.diff(d[:, c["gt_y"]]))
    jumps = step > 0.1
    # 保留旧算法：大于 0.1 m 的位移跳变不计入路径长度。
    # 这是重生/跳变的代理指标，不能单独证明没有跌倒或步态未退化。
    values = [abs(gt[-1] - gt[0]), abs(lat), fwd, step[~jumps].sum(),
              float(jumps.sum()), abs(ierr[-1]),
              np.sqrt(np.mean(d[:, c["pid_correction"]] ** 2))]
    if "pid_feedback_yaw" in c:
        fb = d[:, c["pid_feedback_yaw"]]
        if not np.isfinite(fb).all():
            raise ValueError("PID 实际反馈含 NaN/Inf")
        err = fb - (gt - gt[0])
        values.extend([abs(err[-1]), np.sqrt(np.mean(err ** 2))])
    else:
        # Historical logs contain the post-update integral, not actual PID input.
        # Do not invent a feedback trace from it.
        values.extend([np.nan, np.nan])
    return np.asarray(values), float(step.max())


def load_record(path):
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        d, columns = z["data"], z["columns"]
        meta = {k: z[k].item() for k in z.files if k not in ("data", "columns")}
        values, max_step = metrics(d, columns)
        steps = len(d)
    assumed = {k: v for k, v in DEFAULTS.items() if k not in meta}
    effective = {**DEFAULTS, **meta}
    # sweep_vyaw 是首选依据；旧文件若只在文件名标注激励，则保守隔离。
    sweep = float(meta.get("sweep_vyaw", 0.0))
    if sweep != 0 or ("sweep_vyaw" not in meta and "_sweep" in path.stem):
        cohort = "identification"
        basis = "sweep_vyaw" if "sweep_vyaw" in meta else "filename_sweep"
    else:
        if meta.get("trace_schema_version", 1) >= 2:
            cohort, basis = "recorded", "explicit_trace_schema_v2"
        else:
            cohort = "main" if steps == 3000 else "legacy"
            basis = "historical_steps_rule_not_proof_of_closed_loop"
    ck = str(meta.get("checkpoint", "unknown")).replace("\\", "/")
    parts = [p for p in ck.split("/") if p]
    # 保留训练目录、运行目录及文件名，避免同一运行中不同 checkpoint 被合并。
    checkpoint = "/".join(parts[-3:]) or "unknown"
    key = (checkpoint, effective.get("friction", "unknown"),
           effective.get("noise_std", "unknown"), effective["gyro_bias"], effective["filt"],
           effective["lpf_hz"], effective["bias_tau"], effective["abs_sigma"],
           effective["abs_hz"], effective["abs_dropout"], effective["kp"],
           effective["ki"], effective["pin_vx"], cohort, steps,
           meta.get("dt", "unknown"), meta.get("sweep_vyaw", "unrecorded"),
           meta.get("sweep_seed", "unrecorded") if cohort == "identification" else "not_applicable",
           meta.get("fixed_body", "unrecorded"), meta.get("pin_cmd", "unrecorded"),
           meta.get("task_id", "unrecorded"), meta.get("kd", "unrecorded"),
           meta.get("feedback_mode", "unrecorded"), meta.get("checkpoint_sha256", "unrecorded"),
           meta.get("source_tree_sha256", "unrecorded"), meta.get("trace_schema_version", 1),
           meta.get("reference_delay_steps", "unrecorded"), meta.get("feedback_timing", "unrecorded"))
    # 对非激励历史批次，未记录激励字段与明确为 0 按同一历史筛选规则处理。
    if cohort != "identification":
        key = key[:16] + (0.0,) + key[17:]
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "steps": steps, "cohort": cohort, "cohort_basis": basis, "metadata": meta,
            "assumed_defaults": assumed, "missing_metadata": [k for k in AUDIT_FIELDS if k not in meta],
            "key": key, "metrics": values.tolist(), "max_step_m": max_step}


def collect(patterns, cohort="main"):
    paths = discover(patterns)
    if not paths:
        raise ValueError("没有找到 trace*.npz；裸模式同时查找根目录与 traces/")
    records, seen = [], {}
    for p in paths:
        record = load_record(p)
        digest = record["sha256"]
        record["duplicate_of"] = seen.get(digest)
        seen.setdefault(digest, record["path"])
        record["selected"] = not record["duplicate_of"] and (cohort == "all" or record["cohort"] == cohort)
        records.append(record)
    return records


def summarize(records):
    groups = defaultdict(list)
    for r in records:
        if r["selected"]:
            groups[tuple(r["key"])].append(r)
    stats = {}
    for k, runs in groups.items():
        values = np.array([r["metrics"] for r in runs])
        sd = values.std(0, ddof=1) if len(runs) > 1 else np.full(len(MET), np.nan)
        stats[k] = (values.mean(0), sd, len(runs))
    return groups, stats


def se(a, b, metric):
    return np.sqrt(a[1][metric] ** 2 / a[2] + b[1][metric] ** 2 / b[2])


def print_stats(records, stats):
    counts = Counter(r["cohort"] for r in records if not r["duplicate_of"])
    print(f"发现 {len(records)} 个文件；去除逐字节相同的副本后 {sum(counts.values())} 条轨迹：{dict(counts)}")
    print(f"本次选择 {sum(v[2] for v in stats.values())} 条，{len(stats)} 组。± = 样本标准差 SD；n 按实际文件数。")
    print("main 按历史格式、3000 步且无激励标记筛选，不能据此确认固定机体、固定指令或闭环状态。")
    print("缺失字段的历史默认值逐文件记录于审计清单；零位置跳变不等于已验证零跌倒。")
    if not stats:
        print("所选批次没有轨迹。")
        return
    keys = sorted(stats, key=lambda k: tuple(str(v) for v in k))
    varying = [i for i in range(len(DIMS)) if len({k[i] for k in keys}) > 1]
    fixed = [i for i in range(len(DIMS)) if i not in varying]
    visible_metrics = [i for i in range(len(MET)) if any(np.isfinite(v[0][i]) for v in stats.values())]
    print("固定分组值：" + "  ".join(f"{DIMS[i]}={keys[0][i]}" for i in fixed))
    def fmt(k):
        return " ".join(f"{str(k[i]):>7}" for i in varying)
    print(" ".join(DIMS[i] for i in varying) + "  n  " + " | ".join(MET[i] for i in visible_metrics))
    for k in keys:
        m, s, n = stats[k]
        print(f"{fmt(k)} {n:3d} " + " ".join(f"{m[i]:.3f}±{s[i]:.3f}" for i in visible_metrics))
    row_dims = [i for i in varying if i != FILT_DIM]
    if len(row_dims) <= 1 and FILT_DIM in varying:
        row_dim = row_dims[0] if row_dims else None
        rows = sorted({k[row_dim] for k in keys}, key=str) if row_dim is not None else [None]
        cols = sorted({k[FILT_DIM] for k in keys})
        print("\n交叉对照：列 = 闭环方案；均值±SD；与 dirty 比较采用探索性 2SE 判据。")
        for t in (PRIMARY, SECONDARY):
            print(MET[t] + " | " + " | ".join(cols))
            for row in rows:
                cells = {k[FILT_DIM]: k for k in keys if row_dim is None or k[row_dim] == row}
                label = f"{DIMS[row_dim]}={row}" if row_dim is not None else "同条件"
                print(label + " | " + " | ".join(
                    f"{stats[cells[col]][0][t]:.3f}±{stats[cells[col]][1][t]:.3f}" if col in cells else "—"
                    for col in cols))
                if "dirty" not in cells:
                    continue
                base = stats[cells["dirty"]]
                for col in cols:
                    if col == "dirty" or col not in cells:
                        continue
                    v = stats[cells[col]]
                    delta, threshold = v[0][t] - base[0][t], 2 * se(v, base, t)
                    verdict = "样本不足" if not np.isfinite(threshold) else (
                        "超过2SE" if abs(delta) > threshold else "未超过2SE")
                    print(f"  {col} vs dirty: Δ={delta:+.4f}, 2SE={threshold:.4f}, {verdict}")
    print("\n描述性比值：组均值 |航向净变化| / |积分器偏差|；不是独立性或因果性检验。")
    for k in keys:
        m = stats[k][0]
        ratio = m[SECONDARY] / m[PRIMARY] if m[PRIMARY] > 1e-9 else np.nan
        print(f"  {fmt(k)}  {ratio:.5f}")
    if any(v[2] < 2 for v in stats.values()):
        print("注意：n=1 的组 SD/SE 未定义，显示 nan，不按零方差处理。")


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def export_audit(directory, records, groups, stats):
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "selection_rule": "identification=sweep (first priority); recorded=schema>=2 without sweep; main=historical schema, 3000 steps without sweep; legacy=historical schema, other lengths",
                "records": records}
    summary = [{"conditions": dict(zip(DIMS, k)), "n": v[2], "metrics": {
        name: {"mean": v[0][i], "sd": v[1][i], "se": v[1][i] / np.sqrt(v[2])}
        for i, name in enumerate(MET)}, "files": [r["path"] for r in groups[k]]}
        for k, v in stats.items()]
    for name, value in [("manifest.json", manifest), ("summary.json", summary)]:
        (out / name).write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("patterns", nargs="*", help="文件或通配符；包含路径时不额外搜索 traces/")
    parser.add_argument("--cohort", choices=["main", "identification", "legacy", "recorded", "all"], default="main")
    parser.add_argument("--audit-dir", type=Path, help="输出逐文件清单与含 SD/SE 的分组统计 JSON")
    args = parser.parse_args(argv)
    try:
        records = collect(args.patterns, args.cohort)
        groups, stats = summarize(records)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"读取失败（未静默丢弃异常文件）：{exc}\n")
    print_stats(records, stats)
    if args.audit_dir:
        export_audit(args.audit_dir, records, groups, stats)
        print(f"审计文件：{args.audit_dir.resolve()}")
    return 0 if stats else 1


if __name__ == "__main__":
    raise SystemExit(main())
