# DSMBind

Official repository for:

- Jin et al., *DSMBind: SE(3) denoising score matching for unsupervised binding energy prediction and nanobody design*, Biorxiv 2023
- Jin et al., *Unsupervised protein-ligand binding energy prediction with Neural Euler's Rotation Equations*, NeurIPS 2023

---

## Table of Contents

1. [Quick overview](#quick-overview)
2. [Step 1 — Clone and create virtual environment](#step-1--clone-and-create-virtual-environment)
3. [Step 2 — Install dependencies](#step-2--install-dependencies)
4. [Step 3 — Download the ESM-2 model](#step-3--download-the-esm-2-model)
5. [Step 4 — Download the data](#step-4--download-the-data)
6. [Step 5 — Download the checkpoints](#step-5--download-the-checkpoints)
7. [Step 6 — Configure the ESM checkpoint path](#step-6--configure-the-esm-checkpoint-path)
8. [Step 7 — Run the evaluation](#step-7--run-the-evaluation)
9. [Understanding the output](#understanding-the-output)
10. [CLI reference](#cli-reference)
11. [Using your own data (TCR-pMHC)](#using-your-own-data-tcr-pmhc)
12. [Tips for CPU-only machines](#tips-for-cpu-only-machines)

---

## Quick overview

`evaluate.py` is a unified evaluation pipeline that runs binding energy prediction
across three tasks using pre-trained DSMBind checkpoints:

| Task | Dataset | Checkpoint |
|---|---|---|
| Drug / ligand | CASF-2016, EquiBind, FEP | `ckpts/model.drug.allatom` |
| Antibody-antigen | SAbDab, HER2 | `ckpts/model.antibody.allatom` |
| Protein-protein | SKEMPI | `ckpts/model.skempi.allatom` |

The pipeline works on **CPU and GPU** without any code changes.  
ESM-2 embeddings are **cached to disk** after the first run, so repeat runs are fast.

---

## Step 1 — Clone and create virtual environment

```bash
git clone https://github.com/wengong-jin/DSMBind.git
cd DSMBind

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

---

## Step 2 — Install dependencies

```bash
pip install torch torchvision                    # pytorch (>=2.0 recommended)
pip install biotite tqdm numpy pandas scipy      # core science stack
pip install scikit-learn matplotlib seaborn      # metrics and plotting
pip install rdkit                                # for drug/ligand tasks

# ESM-2 (Facebook Research)
pip install fair-esm

# SRU++ (required for the encoder)
git clone https://github.com/asappresearch/sru
cd sru
git checkout 3.0.0-dev
pip install .
cd ..

# Chemprop (for ligand graph encoder)
pip install chemprop

# Install the DSMBind package itself
pip install -e .
```

> **Note:** `sidechainnet` and `prody` are only needed if you want to preprocess
> your own PDB files with `preprocess_tcr_pmhc.py`. They are not required to run
> `evaluate.py` on the built-in datasets.

---

## Step 3 — Download the ESM-2 model

The pipeline uses **ESM-2 (3B parameters)** to embed protein sequences.
Download the model weights manually:

**Download link:**  
https://drive.google.com/file/d/1lkq1vTDsaG_bw_-40WUJDlKXLjisk4FQ/view?usp=sharing

Save the file to:

```
C:\Users\<your-username>\.cache\torch\hub\checkpoints\esm2_t36_3B_UR50D.pt
```

On Linux / macOS:

```
~/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt
```

> The file is ~6 GB. Make sure you have enough disk space.

---

## Step 4 — Download the data

**Download link:**  
https://drive.google.com/file/d/1jjiuqltc8gBj3_928Kf_3CPlxBP6I4S1/view?usp=drive_link

Extract the archive and place the contents inside the `data/` folder:

```
DSMBind/
  data/
    antibody/
      test_sabdab.jsonl
      test_HER2.jsonl
      train_data.jsonl
      val_data.jsonl
    drug/
      test_casf16.pkl
      test_equibind.pkl
      test_fep.pkl
      train_refine.pkl
    skempi/
      skempi_all.pkl
```

The original data can also be found at https://zenodo.org/records/10402853.

---

## Step 5 — Download the checkpoints

Pre-trained checkpoints are stored in the `ckpts/` folder.
If they are not included in the repository, download them from the same Google Drive
link as the data (Step 4) and place them as follows:

```
DSMBind/
  ckpts/
    model.drug.allatom       ← drug/ligand task
    model.antibody.allatom   ← antibody-antigen task
    model.skempi.allatom     ← protein-protein (SKEMPI) task
```

---

## Step 6 — Configure the ESM checkpoint path

Open **`bindenergy/utils/ioutils.py`** and set `_LOCAL_ESM_CKPT` to the path
where you saved the ESM-2 weights in Step 3:

```python
# bindenergy/utils/ioutils.py  (line ~36)
_LOCAL_ESM_CKPT = r"C:/Users/<your-username>/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt"
```

If you use `bindenergy/apps/antibody/screen.py`, update the same variable there:

```python
# bindenergy/apps/antibody/screen.py
_LOCAL_ESM_CKPT = r"C:/Users/<your-username>/.cache/torch/hub/checkpoints/esm2_t36_3B_UR50D.pt"
```

Replace `<your-username>` with your actual Windows username (e.g. `hajmo`).

---

## Step 7 — Run the evaluation

Make sure the virtual environment is active, then run from the project root:

```bash
# Drug/ligand binding  (CASF-2016, EquiBind, FEP)
python evaluate.py --task drug --device cpu

# Antibody-antigen binding  (SAbDab, HER2)
python evaluate.py --task antibody --device cpu

# Protein-protein mutations  (SKEMPI)
python evaluate.py --task protein --device cpu

# All three tasks at once
python evaluate.py --task all --device cpu
```

On a machine with a GPU, replace `--device cpu` with `--device cuda` (or omit
`--device` entirely — it auto-detects).

> **First run:** ESM-2 embeddings are computed from scratch and saved to
> `results/cache/`. This is slow on CPU (minutes to hours depending on dataset
> size). All subsequent runs load the cache instantly.

---

## Understanding the output

After a run, the `results/` directory looks like this:

```
results/
  predictions_20240518_143022.csv      <- every sample: pdb_id, score, label
  summary_20240518_143022.json         <- Spearman R, Pearson R, n per group
  analysis_20240518_143022.txt         <- full report (also printed to screen)
  cache/
    esm_target_seq_<hash>.pt           <- cached ESM embeddings (reused next run)
  plots/
    drug_casf16_model.drug.allatom.png
    antibody_sabdab_model.antibody.allatom.png
    comparison_20240518_143022.png     <- checkpoint comparison bar chart
```

### predictions CSV columns

| Column | Description |
|---|---|
| `pdb_id` | Structure identifier |
| `task` | `drug`, `antibody`, or `protein` |
| `dataset` | e.g. `casf16`, `sabdab`, `skempi_all` |
| `checkpoint` | Checkpoint filename used |
| `predicted_score` | DSMBind energy score (higher = better binding) |
| `label` | Experimental affinity / ΔΔG (if available) |

### summary JSON

```json
{
  "run_id": "20240518_143022",
  "timestamp": "2024-05-18T14:30:22",
  "results": [
    {
      "tag": "drug | casf16 | model.drug.allatom",
      "n": 285,
      "spearman_r": 0.6234,
      "pearson_r": 0.5987
    }
  ]
}
```

### analysis report

Printed to the terminal and saved as `analysis_<run_id>.txt`.  
Includes per-group metrics (Spearman R, Pearson R, RMSE, p-value, top-10% recall)
and a cross-checkpoint comparison table when multiple checkpoints are evaluated.

### Protein task — partial saves

The SKEMPI evaluation is long on CPU. Every 100 mutant samples a checkpoint file
is written to `results/partial_protein_skempi_all.csv`, so you never lose more
than 100 samples of work if you interrupt the run.

---

## CLI reference

```
python evaluate.py --help

  --task          drug | antibody | protein | all
  --checkpoint    path(s) to checkpoint file(s)   [default: task-appropriate ckpts/]
  --data          custom data file (JSONL or PKL)  [default: built-in test sets]
  --device        auto | cpu | cuda               [default: auto]
  --output        output directory                [default: results/]
  --patch-size    binding-site patch size         [default: 50]
  --epitope-size  max epitope residues            [default: 50]
  --cdr-type      which CDR loops to use          [default: 123456 = all]
```

**Examples:**

```bash
# Force CPU on a GPU machine
python evaluate.py --task antibody --device cpu

# Custom data file
python evaluate.py --task antibody \
    --data data/tcr_pmhc/dataset.jsonl \
    --checkpoint ckpts/model.antibody.allatom

# Compare multiple checkpoints side by side
python evaluate.py --task antibody \
    --checkpoint ckpts/model.antibody.allatom ckpts/model.skempi.allatom

# Change output directory
python evaluate.py --task all --output results/run_001
```

---

## Using your own data (TCR-pMHC)

The antibody model can score TCR-pMHC complexes directly — TCR is structurally
analogous to an antibody, and pMHC plays the role of the antigen.

### Step A — Prepare a manifest CSV

```csv
pdb_id,tcr_alpha_chain,tcr_beta_chain,pmhc_chains,affinity
1ao7,A,B,CD,-8.5
3dxa,D,E,AB,
```

- `pmhc_chains` — all chain IDs as one string, e.g. `CD` means chains C and D.
- `affinity` — optional; leave blank if you have no experimental label.

### Step B — Run the preprocessor

```bash
python preprocess_tcr_pmhc.py \
    --manifest data/tcr_pmhc/manifest.csv \
    --pdb-dir  data/tcr_pmhc/pdbs/ \
    --output   data/tcr_pmhc/dataset.jsonl
```

Use `--cdr-mode imgt` if your PDB files have IMGT residue numbering.  
Use the default `--cdr-mode all` (safer) for sequentially numbered PDBs.

### Step C — Evaluate

```bash
python evaluate.py --task antibody \
    --data data/tcr_pmhc/dataset.jsonl \
    --checkpoint ckpts/model.antibody.allatom
```

---

## Tips for CPU-only machines

| Situation | Recommendation |
|---|---|
| First run is very slow | Normal — ESM embeddings are being computed and cached. Subsequent runs are fast. |
| SKEMPI task takes many hours | Use `results/partial_protein_skempi_all.csv` for results so far. Transfer the cache file to a GPU machine for faster scoring. |
| Want GPU results | Copy the whole project folder to a teammate's GPU machine. The cache is portable — embeddings computed on CPU load fine on GPU. |
| Out of memory | Reduce `--patch-size` or `--epitope-size` (e.g. `--patch-size 30`). |
