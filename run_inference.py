import argparse
import json
import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Local imports – these will work when the script is run from the DSMBind root
from bindenergy.models.drug import (
    DrugEnergyModel,
    DrugAllAtomEnergyModel,
    DrugDecoyModel,
)
from bindenergy.data.drug import DrugDataset
from bindenergy.utils.utils import load_esm_embedding

# Helper to map model name string to class
MODEL_CLASSES = {
    "DrugEnergyModel": DrugEnergyModel,
    "DrugAllAtomEnergyModel": DrugAllAtomEnergyModel,
    "DrugDecoyModel": DrugDecoyModel,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DSMBind model inference on test data"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Computation device",
    )
    parser.add_argument(
        "--models", default="all", help='Comma‑separated list of models or "all"'
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint (optional)",
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Generate 3‑D visualisation with nglview (optional)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Skip heavy computations for CI testing"
    )
    return parser.parse_args()


def get_device(choice):
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def discover_checkpoints(root_dir="ckpts"):
    ckpt_paths = []
    for ext in ("*.ckpt", "*.pth"):
        ckpt_paths.extend(Path(root_dir).rglob(ext))
    return [str(p) for p in ckpt_paths]


def load_model(model_name, ckpt_path, device, args):
    ModelCls = MODEL_CLASSES[model_name]
    model = ModelCls(args).to(device)
    # Load state dict – checkpoint may contain a dict with 'model_ckpt' key or be a raw state_dict
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_ckpt" in ckpt:
        state = ckpt["model_ckpt"]
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()
    return model


def prepare_test_data(args):
    # Minimal synthetic data – a short peptide (3 residues) and a simple ligand (benzene)
    # The actual DSMBind pipeline expects a binder tuple (coords, mol_batch, bind_A, None)
    # and a target tuple (coords, target_S, target_A, None).  We reuse the helper in DrugDataset.
    # Create a dummy entry dictionary compatible with DrugDataset.process.
    dummy_entry = {
        "binder_mol": None,  # placeholder – will be replaced by a simple RDKit molecule
        "target_seq": "AAA",
        "target_coords": None,
    }
    # Load a tiny ligand from RDKit (benzene)
    from rdkit.Chem import MolFromSmiles

    dummy_entry["binder_mol"] = MolFromSmiles("c1ccccc1")
    # Generate a dummy embedding (required by the dataset).  We reuse the utility which can handle a list.
    embedding = load_esm_embedding(dummy_entry, ["target_seq"])
    processed = DrugDataset.process([dummy_entry], args.patch_size)
    binder, target = DrugDataset.make_bind_batch(processed, embedding, args)
    return binder, target


def compute_contact_map(binder, target, device):
    bind_X, _, bind_A, _ = binder
    tgt_X, _, tgt_A, _ = target
    B, N, _ = bind_X.shape
    M = tgt_X.shape[1]
    mask = torch.cat(
        [bind_A[:, :, 1].clamp(max=1).float(), tgt_A[:, :, 1].clamp(max=1).float()],
        dim=1,
    )
    mask_2D = mask.unsqueeze(2) * mask.unsqueeze(1)
    X = torch.cat((bind_X[:, :, 1], tgt_X[:, :, 1]), dim=1)
    D = (X[:, :, None, :] - X[:, None, :, :]).norm(dim=-1)
    return mask_2D * (D < 8.0).float()  # use a generic threshold


def plot_energy_bar(energies, model_name, out_dir):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(range(len(energies))), y=energies, palette="viridis")
    plt.title(f"Predicted energies – {model_name}")
    plt.xlabel("Ligand index")
    plt.ylabel("Energy")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"energy_{model_name}.png"))
    plt.close()


def plot_contact_heatmap(contact, model_name, out_dir):
    plt.figure(figsize=(5, 5))
    sns.heatmap(contact.cpu().numpy(), cmap="magma")
    plt.title(f"Contact map – {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"contact_{model_name}.png"))
    plt.close()


def main():
    args = parse_args()
    device = get_device(args.device)

    # Load model arguments – the original training script stored args in the checkpoint; we reuse a dummy namespace
    class DummyArgs:
        def __init__(self):
            self.hidden_size = 128
            self.threshold = 8.0
            self.patch_size = 32
            self.mpn_depth = 3
            self.dropout = 0.1
            self.mpn_depth = 3
            self.mpn_depth = 3

    model_args = DummyArgs()

    # Determine which models to run
    if args.models == "all":
        selected_models = list(MODEL_CLASSES.keys())
    else:
        selected_models = [m.strip() for m in args.models.split(",")]

    # Discover checkpoints – if a specific path is given we only use that one
    if args.checkpoint:
        ckpt_paths = [args.checkpoint]
    else:
        ckpt_paths = discover_checkpoints()
    if not ckpt_paths:
        raise FileNotFoundError("No checkpoint files found in ckpts/")

    os.makedirs("figures", exist_ok=True)
    results = {}
    for model_name in tqdm(selected_models, desc="Models"):
        # Pick the first checkpoint that matches the model name, otherwise fall back to the first one
        ckpt_path = None
        for p in ckpt_paths:
            if model_name.lower() in p.lower():
                ckpt_path = p
                break
        if not ckpt_path:
            ckpt_path = ckpt_paths[0]
        model = load_model(model_name, ckpt_path, device, model_args)
        binder, target = prepare_test_data(model_args)
        binder = tuple(
            t.to(device) if isinstance(t, torch.Tensor) else t for t in binder
        )
        target = tuple(
            t.to(device) if isinstance(t, torch.Tensor) else t for t in target
        )
        # Inference – many models expose a `predict` method
        with torch.no_grad():
            if hasattr(model, "predict"):
                energy = model.predict(binder, target)
            else:
                raise AttributeError(f"{model_name} has no predict method")
        energy_val = energy.squeeze().cpu().item()
        results[model_name] = {"checkpoint": ckpt_path, "energy": energy_val}
        # Visualisations (skip in dry‑run)
        if not args.dry_run:
            contact = compute_contact_map(binder, target, device)
            plot_energy_bar([energy_val], model_name, "figures")
            plot_contact_heatmap(contact, model_name, "figures")
    # Save JSON summary
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Inference completed. Results saved to results.json")


if __name__ == "__main__":
    main()
