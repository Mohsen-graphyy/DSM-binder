import os
import glob
import re

CKPT_DIR = os.path.join(os.path.dirname(__file__), "ckpts")

MODEL_MAP = {
    "DrugEnergyModel": "DrugEnergyModel",
    "DrugAllAtomEnergyModel": "DrugAllAtomEnergyModel",
    "DrugDecoyModel": "DrugDecoyModel",
}


def list_checkpoints():
    ckpts = []
    for path in glob.glob(os.path.join(CKPT_DIR, "**", "*"), recursive=True):
        if os.path.isfile(path) and path.endswith((".ckpt", ".pth")):
            ckpts.append(path)
    return ckpts


def infer_model_class(ckpt_path):
    # Guess model class from filename using keywords
    name = os.path.basename(ckpt_path).lower()
    for key in MODEL_MAP:
        if key.lower() in name:
            return MODEL_MAP[key]
    return "UnknownModel"


def generate_table():
    rows = []
    for ckpt in list_checkpoints():
        model = infer_model_class(ckpt)
        rows.append(
            f"| `{os.path.relpath(ckpt, os.getcwd())}` | `{model}` | Auto‑detected from filename |"
        )
    header = "| Checkpoint | Model Class | Description | |---|---|---|"
    table = header + "\n" + "\n".join(rows)
    return table


if __name__ == "__main__":
    print(generate_table())
