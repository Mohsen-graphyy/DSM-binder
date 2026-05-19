#!/usr/bin/env python3
"""
DSMBind evaluation pipeline — CPU/GPU compatible.

Supports all built-in benchmarks and custom data (e.g. TCR-pMHC JSONL files
produced by preprocess_tcr_pmhc.py).

Quick start
-----------
# Built-in benchmarks with default checkpoints:
    python evaluate.py --task drug
    python evaluate.py --task antibody
    python evaluate.py --task protein
    python evaluate.py --task all

# Custom TCR-pMHC data (after running preprocess_tcr_pmhc.py):
    python evaluate.py --task antibody \\
        --data data/tcr_pmhc/dataset.jsonl \\
        --checkpoint ckpts/model.antibody.allatom

# Compare multiple checkpoints on the same data:
    python evaluate.py --task antibody \\
        --checkpoint ckpts/model.antibody.allatom ckpts/model.skempi.allatom

# Force CPU even on a GPU machine:
    python evaluate.py --task all --device cpu

Output
------
results/
  predictions_<run_id>.csv   — per-sample: pdb_id, task, dataset, checkpoint,
                               predicted_score, label
  summary_<run_id>.json      — aggregate metrics per (task, dataset, checkpoint)
  plots/                     — regression plots saved as PNG
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
import matplotlib

matplotlib.use("Agg")  # headless; avoids Tk dependency on servers
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from tqdm import tqdm


# ── Device & CPU-patch ────────────────────────────────────────────────────────


def setup_device(device_arg: str) -> torch.device:
    """Return the target device and apply a no-op .cuda() patch when not using CUDA."""
    if device_arg == "auto":
        use_cuda = torch.cuda.is_available()
    elif device_arg == "cuda":
        if not torch.cuda.is_available():
            print(
                "WARNING: --device cuda requested but CUDA is not available. Falling back to CPU."
            )
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = False

    if not use_cuda:
        # Patch .cuda() so all hardcoded .cuda() calls in the codebase become no-ops.
        torch.Tensor.cuda = lambda self, *a, **kw: self
        nn.Module.cuda = lambda self, *a, **kw: self

    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")
    return device


# ── Checkpoint loading ─────────────────────────────────────────────────────────


def load_checkpoint(ckpt_path: str, device: torch.device):
    """Return (model_state_dict, model_args) from a DSMBind checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, (tuple, list)):
        model_state, _opt_state, model_args = ckpt
    else:
        model_state = ckpt.get("model_state_dict", ckpt)
        model_args = ckpt.get("args", None)
    return model_state, model_args


# ── Metrics ───────────────────────────────────────────────────────────────────


def compute_regression_metrics(scores, labels) -> dict:
    scores = np.array(scores, dtype=float)
    labels = np.array(labels, dtype=float)
    if len(scores) < 2:
        return {"n": len(scores)}
    return {
        "n": len(scores),
        "spearman_r": float(scipy.stats.spearmanr(scores, labels).statistic),
        "pearson_r": float(scipy.stats.pearsonr(scores, labels)[0]),
    }


def compute_auc(scores, binary_labels) -> dict:
    from sklearn.metrics import roc_auc_score

    try:
        return {"auc": float(roc_auc_score(binary_labels, scores))}
    except Exception:
        return {}


def print_metrics(tag: str, metrics: dict):
    print(f"\n{'-' * 55}")
    print(f"  {tag}")
    for k, v in metrics.items():
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"    {k}: {val}")
    print(f"{'-' * 55}")


# ── Plotting ──────────────────────────────────────────────────────────────────


def save_regplot(scores, labels, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    sns.regplot(x=scores, y=labels, ax=ax, scatter_kws={"alpha": 0.5})
    ax.set_xlabel("Predicted binding energy")
    ax.set_ylabel("Experimental affinity / ΔΔG")
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


# ── Embedding cache ───────────────────────────────────────────────────────────


def _embed_cache_path(data: list, fields: list, cache_dir: Path) -> Path:
    """Stable cache filename based on a content hash of all embedded sequences."""
    import hashlib
    seqs = sorted({d[f] for d in data for f in fields if d.get(f)})
    h = hashlib.md5("\n".join(seqs).encode()).hexdigest()[:10]
    tag = "_".join(fields)
    return cache_dir / f"esm_{tag}_{h}.pt"


def get_embeddings(data: list, fields: list, cache_dir: Path) -> dict:
    """
    Return ESM embeddings for all sequences in *data[fields]*.

    On first call the embeddings are computed (slow on CPU) and written to a
    .pt file in *cache_dir*.  Every subsequent call with the same data loads
    the cache instantly — no re-computation.
    """
    from bindenergy.utils.ioutils import load_esm_embedding

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _embed_cache_path(data, fields, cache_dir)

    if cache_path.exists():
        print(f"[embed cache] Loading from {cache_path.name} ...")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    n_seqs = len({d[f] for d in data for f in fields if d.get(f)})
    print(
        f"[embed cache] Computing ESM embeddings for {n_seqs} unique sequences "
        f"(fields: {fields}).\n"
        f"              This is slow on CPU. Result will be cached to:\n"
        f"              {cache_path.resolve()}"
    )
    embedding = load_esm_embedding(data, fields)
    torch.save(embedding, cache_path)
    print(f"[embed cache] Saved -> {cache_path.name}")
    return embedding


# ── Task: Drug / Ligand ───────────────────────────────────────────────────────


def evaluate_drug(parsed_args, device, ckpt_paths, out_dir: Path):
    from bindenergy.models.drug import DrugAllAtomEnergyModel
    from bindenergy.data.drug import DrugDataset

    # Resolve datasets
    if parsed_args.data:
        datasets = {
            Path(parsed_args.data).stem: DrugDataset(
                parsed_args.data, parsed_args.patch_size
            )
        }
    else:
        candidates = {
            "casf16": "data/drug/test_casf16.pkl",
            "equibind": "data/drug/test_equibind.pkl",
            "fep": "data/drug/test_fep.pkl",
        }
        datasets = {
            name: DrugDataset(path, parsed_args.patch_size)
            for name, path in candidates.items()
            if os.path.exists(path)
        }

    if not datasets:
        print("No drug datasets found. Skipping drug task.")
        return [], []

    all_data = [entry for ds in datasets.values() for entry in ds.data]
    embedding = get_embeddings(all_data, ["target_seq"], out_dir / "cache")

    rows, summaries = [], []

    for ckpt_path in ckpt_paths:
        ckpt_name = Path(ckpt_path).name
        model_state, model_args = load_checkpoint(ckpt_path, device)
        model = DrugAllAtomEnergyModel(model_args).to(device)
        model.load_state_dict(model_state, strict=False)
        model.eval()

        for ds_name, dataset in datasets.items():
            scores, labels = [], []
            with torch.no_grad():
                for entry in tqdm(dataset.data, desc=f"drug/{ds_name} [{ckpt_name}]"):
                    binder, target = DrugDataset.make_bind_batch(
                        [entry], embedding, model_args
                    )
                    pred = model.predict(binder, target)
                    # DSMBind score: higher = better binding; affinity: lower = tighter
                    scores.append(-pred.item())
                    labels.append(entry.get("affinity"))

            has_labels = all(l is not None for l in labels)
            tag = f"drug | {ds_name} | {ckpt_name}"

            for i, entry in enumerate(dataset.data):
                rows.append(
                    {
                        "pdb_id": entry.get("pdb", str(i)),
                        "task": "drug",
                        "dataset": ds_name,
                        "checkpoint": ckpt_name,
                        "predicted_score": scores[i],
                        "label": labels[i] if has_labels else None,
                    }
                )

            if has_labels:
                m = compute_regression_metrics(scores, labels)
                print_metrics(tag, m)
                summaries.append({"tag": tag, **m})
                save_regplot(
                    scores,
                    labels,
                    f"Drug {ds_name} [{ckpt_name}]\nSpearman R = {m.get('spearman_r', float('nan')):.4f}",
                    out_dir / "plots" / f"drug_{ds_name}_{ckpt_name}.png",
                )

    return rows, summaries


# ── Task: Antibody / TCR-pMHC ─────────────────────────────────────────────────


def evaluate_antibody(parsed_args, device, ckpt_paths, out_dir: Path):
    from bindenergy.models.energy import AllAtomEnergyModel
    from bindenergy.data.antibody import AntibodyDataset

    cdr_type = parsed_args.cdr_type

    if parsed_args.data:
        datasets = {
            Path(parsed_args.data).stem: AntibodyDataset(
                parsed_args.data, cdr_type, parsed_args.epitope_size
            )
        }
    else:
        candidates = {
            "sabdab": "data/antibody/test_sabdab.jsonl",
            "HER2": "data/antibody/test_HER2.jsonl",
        }
        datasets = {
            name: AntibodyDataset(path, cdr_type, parsed_args.epitope_size)
            for name, path in candidates.items()
            if os.path.exists(path)
        }

    if not datasets:
        print("No antibody datasets found. Skipping antibody task.")
        return [], []

    all_data = [entry for ds in datasets.values() for entry in ds.data]
    embedding = get_embeddings(all_data, ["antibody_seq", "antigen_seq"], out_dir / "cache")

    rows, summaries = [], []

    for ckpt_path in ckpt_paths:
        ckpt_name = Path(ckpt_path).name
        model_state, model_args = load_checkpoint(ckpt_path, device)
        model = AllAtomEnergyModel(model_args).to(device)
        model.load_state_dict(model_state, strict=False)
        model.eval()

        for ds_name, dataset in datasets.items():
            scores, labels = [], []
            with torch.no_grad():
                for entry in tqdm(
                    dataset.data, desc=f"antibody/{ds_name} [{ckpt_name}]"
                ):
                    binder, target = AntibodyDataset.make_local_batch(
                        [entry], embedding, model_args
                    )
                    pred = model.predict(binder, target)
                    scores.append(-pred.item())
                    labels.append(entry.get("affinity"))

            has_labels = all(l is not None for l in labels)
            tag = f"antibody | {ds_name} | {ckpt_name}"

            for i, entry in enumerate(dataset.data):
                rows.append(
                    {
                        "pdb_id": entry.get("pdb", str(i)),
                        "task": "antibody",
                        "dataset": ds_name,
                        "checkpoint": ckpt_name,
                        "predicted_score": scores[i],
                        "label": labels[i] if has_labels else None,
                    }
                )

            if has_labels:
                m = compute_regression_metrics(scores, labels)
                print_metrics(tag, m)
                summaries.append({"tag": tag, **m})
                save_regplot(
                    scores,
                    labels,
                    f"Antibody {ds_name} [{ckpt_name}]\nSpearman R = {m.get('spearman_r', float('nan')):.4f}",
                    out_dir / "plots" / f"antibody_{ds_name}_{ckpt_name}.png",
                )

    return rows, summaries


# ── Task: Protein-Protein / SKEMPI ────────────────────────────────────────────


def evaluate_protein(parsed_args, device, ckpt_paths, out_dir: Path):
    from bindenergy.models.energy import AllAtomEnergyModel
    from bindenergy.data.protein import ProteinDataset

    data_path = parsed_args.data or "data/skempi/skempi_all.pkl"
    if not os.path.exists(data_path):
        print(f"Protein dataset not found at '{data_path}'. Skipping protein task.")
        return [], []

    ds_name = Path(data_path).stem
    dataset = ProteinDataset(data_path, parsed_args.patch_size)

    # Dataset statistics (helps the user understand the workload up-front)
    wt_entries  = [e for e in dataset.data if len(e["pdb"][1]) == 0]
    mut_entries = [e for e in dataset.data if len(e["pdb"][1]) > 0]
    print(
        f"\nDataset: {len(dataset.data)} valid entries "
        f"({len(wt_entries)} wild-type, {len(mut_entries)} mutants)"
    )

    embedding = get_embeddings(
        dataset.data, ["binder_full", "target_full"], out_dir / "cache"
    )

    # Partial-results file so an interrupt doesn't lose all progress
    partial_path = out_dir / f"partial_protein_{ds_name}.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, summaries = [], []

    for ckpt_path in ckpt_paths:
        ckpt_name = Path(ckpt_path).name
        model_state, model_args = load_checkpoint(ckpt_path, device)
        model = AllAtomEnergyModel(model_args).to(device)
        model.load_state_dict(model_state, strict=False)
        model.eval()

        tag = f"protein | {ds_name} | {ckpt_name}"

        with torch.no_grad():
            # Pass 1: wild-type energies (now visible with its own progress bar)
            wt_energy = {}
            for entry in tqdm(wt_entries, desc=f"  WT energies [{ckpt_name}]"):
                pdb, _mut, _ddg = entry["pdb"]
                binder, target = ProteinDataset.make_local_batch(
                    [entry], embedding, model_args, "binder", "target"
                )
                wt_energy[pdb] = (
                    model.predict(binder, target) + model.predict(target, binder)
                )

            # Pass 2: mutant ΔΔG — saves a partial CSV every 100 samples
            scores, labels, pdb_ids = [], [], []
            partial_rows = []
            for i, entry in enumerate(
                tqdm(mut_entries, desc=f"  Mutants [{ckpt_name}]")
            ):
                pdb, mutation, ddg = entry["pdb"]
                if pdb not in wt_energy:
                    continue
                binder, target = ProteinDataset.make_local_batch(
                    [entry], embedding, model_args, "binder", "target"
                )
                score = (
                    model.predict(binder, target)
                    + model.predict(target, binder)
                    - wt_energy[pdb]
                )
                s = -score.item()
                scores.append(s)
                labels.append(ddg)
                pid = f"{pdb}_{mutation}"
                pdb_ids.append(pid)
                partial_rows.append(
                    {
                        "pdb_id": pid,
                        "task": "protein",
                        "dataset": ds_name,
                        "checkpoint": ckpt_name,
                        "predicted_score": s,
                        "label": ddg,
                    }
                )
                # Flush partial results every 100 samples
                if (i + 1) % 100 == 0:
                    pd.DataFrame(partial_rows).to_csv(partial_path, index=False)
            # Final flush
            if partial_rows:
                pd.DataFrame(partial_rows).to_csv(partial_path, index=False)
                print(f"  Partial results saved -> {partial_path.resolve()}")

        for i, pid in enumerate(pdb_ids):
            rows.append(
                {
                    "pdb_id": pid,
                    "task": "protein",
                    "dataset": ds_name,
                    "checkpoint": ckpt_name,
                    "predicted_score": scores[i],
                    "label": labels[i],
                }
            )

        if labels:
            m = compute_regression_metrics(scores, labels)
            print_metrics(tag, m)
            summaries.append({"tag": tag, **m})
            save_regplot(
                scores,
                labels,
                f"Protein {ds_name} [{ckpt_name}]\nSpearman R = {m.get('spearman_r', float('nan')):.4f}",
                out_dir / "plots" / f"protein_{ds_name}_{ckpt_name}.png",
            )

    return rows, summaries


# ── Persistence ───────────────────────────────────────────────────────────────


def save_outputs(rows, summaries, out_dir: Path, run_id: str) -> Path:
    """Write predictions CSV and summary JSON. Returns the CSV path."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always write the CSV (empty DataFrame if no rows, so the file always exists)
    csv_path = out_dir / f"predictions_{run_id}.csv"
    pd.DataFrame(rows if rows else []).to_csv(csv_path, index=False)
    print(f"\nPredictions -> {csv_path.resolve()}")

    # Always write the JSON summary
    summary_path = out_dir / f"summary_{run_id}.json"
    payload = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "results": summaries,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Summary    -> {summary_path.resolve()}")

    return csv_path


# ── Analysis ─────────────────────────────────────────────────────────────────


def analyze_results(csv_path: Path, out_dir: Path, run_id: str):
    """
    Read the predictions CSV and produce:
      - A formatted text report  (results/analysis_<run_id>.txt)
      - A checkpoint-comparison bar chart  (results/plots/comparison_<run_id>.png)

    Metrics computed per (task, dataset, checkpoint) group:
      n, Spearman R, Spearman p-value, Pearson R, RMSE,
      top-10% recall (fraction of true top-10% binders ranked in top-10% by score)
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        print("No predictions to analyze.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("Predictions CSV is empty - nothing to analyze.")
        return

    has_labels = df["label"].notna().all()
    groups = ["task", "dataset", "checkpoint"]

    report_lines = [
        f"DSMBind Evaluation Analysis",
        f"Run ID  : {run_id}",
        f"File    : {csv_path.resolve()}",
        f"Samples : {len(df)}",
        f"Labeled : {'yes' if has_labels else 'no (scores only)'}",
        "",
    ]

    # ── Per-group metrics table ───────────────────────────────────────────────
    group_records = []

    for key, gdf in df.groupby(groups):
        task, dataset, ckpt = key
        scores = gdf["predicted_score"].to_numpy()
        rec = {"task": task, "dataset": dataset, "checkpoint": ckpt, "n": len(scores)}

        if has_labels and gdf["label"].notna().all():
            labels = gdf["label"].to_numpy()
            sp = scipy.stats.spearmanr(scores, labels)
            pr = scipy.stats.pearsonr(scores, labels)
            rmse = float(np.sqrt(np.mean((scores - labels) ** 2)))

            # Top-10% recall: how many true top-10% binders are predicted in top-10%
            k = max(1, len(scores) // 10)
            # Lower label = tighter binding; higher score = better predicted binding
            true_top  = set(np.argsort(labels)[:k])          # smallest labels
            pred_top  = set(np.argsort(scores)[-k:])          # largest scores
            recall_10 = len(true_top & pred_top) / k

            rec.update({
                "spearman_r": round(float(sp.statistic), 4),
                "spearman_p": float(sp.pvalue),
                "pearson_r":  round(float(pr[0]), 4),
                "rmse":       round(rmse, 4),
                "top10_recall": round(recall_10, 4),
            })
        group_records.append(rec)

    # Print/write the per-group table
    report_lines.append("=" * 70)
    report_lines.append("Per-group results")
    report_lines.append("=" * 70)
    for r in group_records:
        report_lines.append(
            f"\n  Task: {r['task']}  |  Dataset: {r['dataset']}  |  Checkpoint: {r['checkpoint']}"
        )
        report_lines.append(f"    n               = {r['n']}")
        if "spearman_r" in r:
            p_str = f"{r['spearman_p']:.2e}"
            report_lines.append(f"    Spearman R      = {r['spearman_r']:.4f}  (p={p_str})")
            report_lines.append(f"    Pearson R       = {r['pearson_r']:.4f}")
            report_lines.append(f"    RMSE            = {r['rmse']:.4f}")
            report_lines.append(f"    Top-10% recall  = {r['top10_recall']:.4f}")

    # ── Cross-checkpoint comparison table ─────────────────────────────────────
    if has_labels and len(group_records) > 1:
        ckpts_all    = sorted({r["checkpoint"] for r in group_records})
        datasets_all = sorted({f"{r['task']}/{r['dataset']}" for r in group_records})

        if len(ckpts_all) > 1:
            report_lines.append("")
            report_lines.append("=" * 70)
            report_lines.append("Checkpoint comparison (Spearman R)")
            report_lines.append("=" * 70)
            col_w = max(25, max(len(c) for c in ckpts_all) + 2)
            row_w = max(20, max(len(d) for d in datasets_all) + 2)
            header = f"{'Dataset':<{row_w}}" + "".join(f"{c:<{col_w}}" for c in ckpts_all)
            report_lines.append(header)
            report_lines.append("-" * len(header))
            idx = {(r["task"] + "/" + r["dataset"], r["checkpoint"]): r.get("spearman_r", float("nan"))
                   for r in group_records}
            for ds in datasets_all:
                row = f"{ds:<{row_w}}"
                for c in ckpts_all:
                    val = idx.get((ds, c), float("nan"))
                    row += f"{val:<{col_w}.4f}" if not np.isnan(val) else f"{'N/A':<{col_w}}"
                report_lines.append(row)

    # ── Save text report ──────────────────────────────────────────────────────
    report_path = out_dir / f"analysis_{run_id}.txt"
    report_text = "\n".join(report_lines)
    print(report_text)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    print(f"\nAnalysis report -> {report_path.resolve()}")

    # ── Comparison bar chart ──────────────────────────────────────────────────
    if not has_labels or "spearman_r" not in group_records[0]:
        return

    gdf_plot = pd.DataFrame(group_records)
    gdf_plot = gdf_plot[gdf_plot["spearman_r"].notna()]
    if gdf_plot.empty:
        return

    tasks_present = gdf_plot["task"].unique()
    n_tasks = len(tasks_present)
    fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 4), squeeze=False)

    for ax, task in zip(axes[0], tasks_present):
        sub = gdf_plot[gdf_plot["task"] == task]
        ckpts_sub    = sub["checkpoint"].unique()
        datasets_sub = sub["dataset"].unique()

        x = np.arange(len(datasets_sub))
        bar_w = 0.8 / max(len(ckpts_sub), 1)

        for i, ck in enumerate(ckpts_sub):
            vals = [
                sub[(sub["dataset"] == ds) & (sub["checkpoint"] == ck)]["spearman_r"].values
                for ds in datasets_sub
            ]
            vals = [v[0] if len(v) else float("nan") for v in vals]
            offset = (i - len(ckpts_sub) / 2 + 0.5) * bar_w
            bars = ax.bar(x + offset, vals, bar_w, label=ck, alpha=0.85)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.3f}",
                        ha="center", va="bottom", fontsize=7,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(datasets_sub, rotation=20, ha="right")
        ax.set_ylabel("Spearman R")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Task: {task}")
        if len(ckpts_sub) > 1:
            ax.legend(fontsize=7, loc="lower right")

    plt.suptitle("Checkpoint comparison - Spearman R", fontsize=11)
    plt.tight_layout()
    comp_path = out_dir / "plots" / f"comparison_{run_id}.png"
    comp_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(comp_path, dpi=150)
    plt.close()
    print(f"Comparison chart -> {comp_path.resolve()}")


# ── Default checkpoints per task ──────────────────────────────────────────────

DEFAULT_CKPTS = {
    "drug": ["ckpts/model.drug.allatom"],
    "antibody": ["ckpts/model.antibody.allatom"],
    "protein": ["ckpts/model.skempi.allatom"],
}


# ── Entry point ───────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DSMBind evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--task",
        required=True,
        choices=["drug", "antibody", "protein", "all"],
        help="Evaluation task.  'all' runs every task that has a valid checkpoint + data.",
    )
    p.add_argument(
        "--checkpoint",
        nargs="+",
        metavar="PATH",
        help="One or more checkpoint files.  Defaults to the task-appropriate ckpts/ file.",
    )
    p.add_argument(
        "--data",
        default=None,
        metavar="PATH",
        help="Custom data file (JSONL for antibody/TCR, PKL for drug/protein).  "
        "If omitted the built-in test sets are used.",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device.  'auto' uses CUDA when available (default).",
    )
    p.add_argument(
        "--output",
        default="results",
        metavar="DIR",
        help="Output directory for CSV, JSON summary, and plots (default: results/).",
    )
    p.add_argument(
        "--patch-size",
        type=int,
        default=50,
        metavar="N",
        help="Number of residues in the binding-site patch (default: 50).",
    )
    p.add_argument(
        "--epitope-size",
        type=int,
        default=50,
        metavar="N",
        help="Max epitope residues for antibody task (default: 50).",
    )
    p.add_argument(
        "--cdr-type",
        default="123456",
        help="CDR loop labels to include as paratope, e.g. '3' for CDR-H3 only "
        "(default: '123456' = all CDR loops).",
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    device = setup_device(args.device)
    out_dir = Path(args.output)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    tasks = ["drug", "antibody", "protein"] if args.task == "all" else [args.task]

    all_rows, all_summaries = [], []

    for task in tasks:
        ckpts = args.checkpoint or DEFAULT_CKPTS.get(task, [])
        ckpts = [c for c in ckpts if os.path.exists(c)]
        if not ckpts:
            print(f"\nNo valid checkpoint found for task '{task}' - skipping.")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Task: {task}   Checkpoints: {[Path(c).name for c in ckpts]}")
        print(f"{'=' * 60}")

        if task == "drug":
            rows, summaries = evaluate_drug(args, device, ckpts, out_dir)
        elif task == "antibody":
            rows, summaries = evaluate_antibody(args, device, ckpts, out_dir)
        elif task == "protein":
            rows, summaries = evaluate_protein(args, device, ckpts, out_dir)
        else:
            rows, summaries = [], []

        all_rows.extend(rows)
        all_summaries.extend(summaries)

    csv_path = save_outputs(all_rows, all_summaries, out_dir, run_id)
    analyze_results(csv_path, out_dir, run_id)
    print(f"\nDone.  Run ID: {run_id}")


if __name__ == "__main__":
    main()
