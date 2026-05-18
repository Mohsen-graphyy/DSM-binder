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
    print(f"\n{'─' * 55}")
    print(f"  {tag}")
    for k, v in metrics.items():
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"    {k}: {val}")
    print(f"{'─' * 55}")


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


# ── Task: Drug / Ligand ───────────────────────────────────────────────────────


def evaluate_drug(parsed_args, device, ckpt_paths, out_dir: Path):
    from bindenergy.models.drug import DrugAllAtomEnergyModel
    from bindenergy.data.drug import DrugDataset
    from bindenergy.utils.ioutils import load_esm_embedding  # noqa (ioutils re-export)

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
    embedding = load_esm_embedding(all_data, ["target_seq"])

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
    from bindenergy.utils.ioutils import load_esm_embedding

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
    embedding = load_esm_embedding(all_data, ["antibody_seq", "antigen_seq"])

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
    from bindenergy.utils.ioutils import load_esm_embedding

    data_path = parsed_args.data or "data/skempi/skempi_all.pkl"
    if not os.path.exists(data_path):
        print(f"Protein dataset not found at '{data_path}'. Skipping protein task.")
        return [], []

    ds_name = Path(data_path).stem
    dataset = ProteinDataset(data_path, parsed_args.patch_size)
    embedding = load_esm_embedding(dataset.data, ["binder_full", "target_full"])

    rows, summaries = [], []

    for ckpt_path in ckpt_paths:
        ckpt_name = Path(ckpt_path).name
        model_state, model_args = load_checkpoint(ckpt_path, device)
        model = AllAtomEnergyModel(model_args).to(device)
        model.load_state_dict(model_state, strict=False)
        model.eval()

        tag = f"protein | {ds_name} | {ckpt_name}"

        with torch.no_grad():
            # First pass: wild-type energies
            wt_energy = {}
            for entry in dataset.data:
                pdb, mutation, _ddg = entry["pdb"]
                if len(mutation) == 0:
                    binder, target = ProteinDataset.make_local_batch(
                        [entry], embedding, model_args, "binder", "target"
                    )
                    wt_energy[pdb] = model.predict(binder, target) + model.predict(
                        target, binder
                    )

            # Second pass: mutant energies → ΔΔG prediction
            scores, labels, pdb_ids = [], [], []
            for entry in tqdm(dataset.data, desc=f"protein/{ds_name} [{ckpt_name}]"):
                pdb, mutation, ddg = entry["pdb"]
                if len(mutation) > 0 and pdb in wt_energy:
                    binder, target = ProteinDataset.make_local_batch(
                        [entry], embedding, model_args, "binder", "target"
                    )
                    score = (
                        model.predict(binder, target)
                        + model.predict(target, binder)
                        - wt_energy[pdb]
                    )
                    scores.append(-score.item())
                    labels.append(ddg)
                    pdb_ids.append(f"{pdb}_{mutation}")

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


def save_outputs(rows, summaries, out_dir: Path, run_id: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"predictions_{run_id}.csv"
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\nPredictions → {csv_path}")

    summary_path = out_dir / f"summary_{run_id}.json"
    payload = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "results": summaries,
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Summary    → {summary_path}")


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
        epilog=__doc__,
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
            print(f"\nNo valid checkpoint found for task '{task}' — skipping.")
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

    save_outputs(all_rows, all_summaries, out_dir, run_id)
    print(f"\nDone.  Run ID: {run_id}")


if __name__ == "__main__":
    main()
