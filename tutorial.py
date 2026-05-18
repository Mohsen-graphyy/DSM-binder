# %%
import numpy as np
import scipy
import random
import glob
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve

from bindenergy import *

sns.set(rc={"figure.dpi": 300, "savefig.dpi": 300})


# %%
from bindenergy.models.drug import DrugAllAtomEnergyModel

# %% [markdown]
# ### Protein-Ligand Binding

# %%
import torch
import torch.nn as nn

# ── CPU-only fallback ────────────────────────────────────────────────────────
# This PyTorch build has no CUDA support.  Patch .cuda() on both Tensor and
# Module so all hardcoded .cuda() calls in the model/data code silently become
# no-ops and everything runs on CPU instead of crashing.
if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self
# ────────────────────────────────────────────────────────────────────────────

# %% [markdown]
# Load model (DrugAllAtomEnergyModel considers the full-atom structure of a protein-ligand complex, including side-chains).

# %%
# %%
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %%
fn = "ckpts/model.drug.allatom"
model = None
if os.path.exists(fn):
    checkpoint = torch.load(fn, map_location=device, weights_only=False)
    if isinstance(checkpoint, tuple) or isinstance(checkpoint, list):
        model_ckpt, opt_ckpt, model_args = checkpoint
    else:
        model_ckpt = checkpoint.get("model_state_dict", checkpoint)
        model_args = checkpoint.get("args", None)

    # %%
    model = DrugAllAtomEnergyModel(model_args).to(device)
    model.load_state_dict(model_ckpt, strict=False)
    model.eval()
    print("Model loaded successfully!")
else:
    print(f"Skipping DrugAllAtomEnergyModel: {fn} not found")

# %% [markdown]
# In our NeurIPS paper, we used the PDBBind core (CASF 2016) as the test set. We use the docking test set from EquiBind (Starks et al., ICML 2022)as our validation set.
#
# In our new biorxiv paper, we further evaluate DSMBind on Merck free energy perturbation (FEP) benchmark.

# %%
test_casf16, test_equibind, test_fep = None, None, None
embedding = None
if os.path.exists("data/drug/test_casf16.pkl"):
    test_casf16 = DrugDataset("data/drug/test_casf16.pkl", 50)
    test_equibind = DrugDataset("data/drug/test_equibind.pkl", 50)
    test_fep = DrugDataset("data/drug/test_fep.pkl", 50)

    # %%
    embedding = load_esm_embedding(
        test_equibind.data + test_casf16.data + test_fep.data, ["target_seq"]
    )
else:
    print("Warning: Drug datasets not found. Skipping data loading.")

# %% [markdown]
# Inference script (note: DSMBind predicted score is the higher the better, while binding affinity is the lower the better)


# %%
def pdbbind_evaluate(model, data, embedding, args):
    model.eval()
    score = []
    label = []
    with torch.no_grad():
        for entry in tqdm(data):
            binder, target = DrugDataset.make_bind_batch([entry], embedding, args)
            pred = model.predict(binder, target)
            score.append(-1.0 * pred.item())
            label.append(entry["affinity"])
    return scipy.stats.spearmanr(score, label)[0], score, label


# %% [markdown]
# Make predictions on CASF-2016/Equibind test sets

# %%
if model is not None and test_casf16 is not None:
    casf16_corr, casf16_score, casf16_label = pdbbind_evaluate(
        model, test_casf16, embedding, model_args
    )
    equibind_corr, equibind_score, equibind_label = pdbbind_evaluate(
        model, test_equibind, embedding, model_args
    )

    # %%
    sns.regplot(x=casf16_score, y=casf16_label)
    plt.xlabel("Predicted binding energy")
    plt.ylabel("Experimental binding affinity ($\log_{10}$)")
    plt.title(f"CASF 2016 Spearman R = {casf16_corr:.4f}")
    plt.show()

    # %%
    sns.regplot(x=equibind_score, y=equibind_label)
    plt.xlabel("Predicted binding energy")
    plt.ylabel("Experimental binding affinity ($\log_{10}$)")
    plt.title(f"Equibind test Spearman R = {equibind_corr:.4f}")
    plt.show()

    # %% [markdown]
    # ##### Merck FEP Benchmark Evaluation

    # %%
    score = defaultdict(list)
    label = defaultdict(list)
    with torch.no_grad():
        for entry in tqdm(test_fep):
            pdb = entry["pdb"]
            binder, target = DrugDataset.make_bind_batch([entry], embedding, model_args)
            pred = model.predict(binder, target)
            score[pdb].append(pred.item())
            label[pdb].append(-1.0 * entry["affinity"])

    # %%
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
    fig.text(0.5, 0.04, "Predicted binding energy", ha="center")
    fig.text(0.08, 0.5, "Experimental binding affinity", va="center", rotation="vertical")
    for i, pdb in enumerate(score.keys()):
        i, j = i // 4, i % 4
        sns.regplot(x=score[pdb], y=label[pdb], ax=axes[i, j])
        corr = scipy.stats.spearmanr(score[pdb], label[pdb])[0]
        axes[i, j].set_title(f"{pdb}: Spearman R={corr:.3f}")
    plt.show()
else:
    print("Skipping PDBBind and FEP evaluation due to missing model or data.")

# %% [markdown]
# ### Antibody-Antigen Binding


# %%
def sabdab_evaluate(model, data, embedding, args):
    model.eval()
    pred, label = [], []
    for ab in tqdm(data):
        binder, target = AntibodyDataset.make_local_batch([ab], embedding, args)
        score = model.predict(binder, target)
        pred.append(-1.0 * score.item())
        label.append(ab["affinity"])
    return scipy.stats.spearmanr(pred, label)[0], pred, label


# %% [markdown]
# In our NeurIPS and biorxiv paper, we evaluate on two test sets:
# * The first test set is from SAbDab. It has 566 antibody-antigen complexes with binding affinity labels
# * The second test set is from . It has 424 HER2-trastuzumab variants (CDR3 mutation) with binding affinity labels

# %%
if os.path.exists("data/antibody/test_sabdab.jsonl") and os.path.exists("ckpts/model.antibody.allatom"):
    test_sabdab = AntibodyDataset(
        "data/antibody/test_sabdab.jsonl", cdr_type="123456", epitope_size=50
    )
    test_HER2 = AntibodyDataset(
        "data/antibody/test_HER2.jsonl", cdr_type="123456", epitope_size=50
    )

    # %%
    embedding = load_esm_embedding(
        test_sabdab.data + test_HER2.data, ["antibody_seq", "antigen_seq"]
    )

    # %%
    model_ckpt, opt_ckpt, model_args = torch.load("ckpts/model.antibody.allatom", map_location=device, weights_only=False)
    model = AllAtomEnergyModel(model_args).to(device)
    model.load_state_dict(model_ckpt, strict=False)
    model.eval()

    # %%
    test_corr, pred, label = sabdab_evaluate(model, test_sabdab, embedding, model_args)

    # %%
    sns.regplot(x=pred, y=label)
    plt.xlabel("Predicted binding energy")
    plt.ylabel("Experiment binding affinity")
    plt.title(f"SAbDab test correlation: Spearman R = {test_corr:.4f}")
    plt.show()

    # %%
    pred, label = [], []
    for ab in tqdm(test_HER2):
        binder, target = AntibodyDataset.make_local_batch([ab], embedding, model_args)
        score = model.predict(binder, target)
        pred.append(score.item())
        label.append(
            int(ab["affinity"] < -8.7)
        )  # better than wildtype trastuzumab binding affinity

    fpr, tpr, _ = roc_curve(label, pred)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc_score(label, pred),
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("HER2 AUROC")
    plt.legend(loc="lower right")
    plt.show()
else:
    print("Skipping Antibody-Antigen evaluation due to missing model or data.")

# %% [markdown]
# #### SKEMPI Results

# %%
if os.path.exists("data/skempi/skempi_all.pkl") and os.path.exists("ckpts/model.skempi.allatom"):
    test_data = ProteinDataset("data/skempi/skempi_all.pkl", 50)
    embedding = load_esm_embedding(test_data.data, ["binder_full", "target_full"])

    # %%
    model_ckpt, _, model_args = torch.load("ckpts/model.skempi.allatom", map_location=device, weights_only=False)
    model = AllAtomEnergyModel(model_args).to(device)
    model.load_state_dict(model_ckpt, strict=False)
    model.eval()

    # %%
    with torch.no_grad():
        wt_map = {}
        pred, label = [], []
        for entry in test_data.data:
            pdb, mutation, ddg = entry["pdb"]
            if len(mutation) == 0:
                binder, target = ProteinDataset.make_local_batch(
                    [entry], embedding, model_args, "binder", "target"
                )
                wt_map[pdb] = model.predict(binder, target) + model.predict(target, binder)

        for entry in test_data.data:
            pdb, mutation, ddg = entry["pdb"]
            if len(mutation) > 0:
                binder, target = ProteinDataset.make_local_batch(
                    [entry], embedding, model_args, "binder", "target"
                )
                score = model.predict(binder, target) + model.predict(target, binder)
                score = score - wt_map[pdb]
                pred.append(-1.0 * score.item())
                label.append(ddg)

    # %%
    test_corr = scipy.stats.spearmanr(pred, label)[0]
    sns.regplot(x=pred, y=label)
    plt.xlabel("Predicted DDG")
    plt.ylabel("Experimental DDG")
    plt.title(f"SKEMPI test correlation: Spearman R = {test_corr:.4f}")
    plt.show()
else:
    print("Skipping SKEMPI evaluation due to missing model or data.")

# %% [markdown]
# ### Ligand virtual screening

# %%
if os.path.exists("ckpts/model.recA"):
    fn = "ckpts/model.recA"
    model_ckpt, opt_ckpt, model_args = torch.load(fn, map_location=device, weights_only=False)
    model = DrugAllAtomEnergyModel(model_args).to(device)
    model.load_state_dict(model_ckpt, strict=False)
    model.eval()

    # %%
    if os.path.exists("data/recA/3cmt.pdb") and len(glob.glob("data/recA/FABind/*.sdf")) > 0:
        sdf_list = sorted(glob.glob("data/recA/FABind/*.sdf"))
        result = model.virtual_screen("data/recA/3cmt.pdb", sdf_list, batch_size=1)
        print(pd.DataFrame(result))

    # %%
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    if os.path.exists("data/recA/recA.csv"):
        with open("data/recA/recA.csv") as f:
            label_map = {}
            for line in f.readlines()[1:]:
                smiles, label = line.split(",")
                smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)
                label = int(label)
                label_map[smiles] = label

        embedding = None
        for fn in glob.glob("/home/wjin/bindenergy/ckpts/drug/seed*/full-d*/model.ckpt.*"):
            if not os.path.exists(fn): continue
            model_ckpt, opt_ckpt, model_args = torch.load(fn, map_location=device, weights_only=False)
            model = DrugAllAtomEnergyModel(model_args).to(device)
            model.load_state_dict(model_ckpt, strict=False)
            model.eval()
            sdf_list = sorted(glob.glob("data/recA/FABind/*.sdf"))
            if len(sdf_list) > 0:
                result = model.virtual_screen(
                    "data/recA/3cmt.pdb", sdf_list, batch_size=1, embedding=embedding
                )
                x, y = [], []
                for _, smiles, score in result:
                    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)
                    if smiles in label_map:
                        x.append(score)
                        y.append(label_map[smiles])
                if len(y) > 0:
                    print(fn, roc_auc_score(y, x))
else:
    print("Skipping Ligand virtual screening due to missing model.recA checkpoint.")


# %% [markdown]
# ### Inference on test_data 

# %%
if os.path.exists("test_data/protein.pdb") and os.path.exists("test_data/ligand.sdf") and os.path.exists("ckpts/model.drug.allatom"):
    print("Running inference on test_data...")
    try:
        fn = "ckpts/model.drug.allatom"
        checkpoint = torch.load(fn, map_location=device, weights_only=False)
        if isinstance(checkpoint, tuple) or isinstance(checkpoint, list):
            model_ckpt, opt_ckpt, model_args = checkpoint
        else:
            model_ckpt = checkpoint.get("model_state_dict", checkpoint)
            model_args = checkpoint.get("args", None)
            
        model = DrugAllAtomEnergyModel(model_args).to(device)
        model.load_state_dict(model_ckpt, strict=False)
        model.eval()
        
        sdf_list = ["test_data/ligand.sdf"]
        result = model.virtual_screen("test_data/protein.pdb", sdf_list, batch_size=1)
        print("Inference successful:")
        print(pd.DataFrame(result))
    except Exception as e:
        print(f"Error during test_data inference: {e}")

# %%
