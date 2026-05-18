# DSMBind Senior Mentoring Walkthrough

Welcome to the **DSMBind** onboarding! As a senior developer joining this project, this guide will orient you to the core architecture, data flows, and where to look when you want to modify the codebase for future experiments.

[DSMBind](https://github.com/wengong-jin/DSMBind) is used for unsupervised binding energy prediction and nanobody design using a technique based on *Denoising Score Matching* with SE(3) frame averaging to ensure rotational/translational symmetry (equivariance).

## 1. High-Level Architecture Overview
The repository structures its components into modules handling data loading, feature encoding, frame averaging, and predictive modeling:
* `bindenergy/data/`: Utilities for parsing structural data, managing vocabulary, constants like atomic radii/types, and creating batches. It separates domains into `protein.py`, `drug.py`, and `antibody.py` based on the interaction type.
* `bindenergy/models/`: Your core model logic and architectures. Key building blocks like `FAEncoder` (Frame Averaging Encoder), `AllAtomEncoder`, and `MPNEncoder` reside here.
* `bindenergy/apps/`: Training pipelines and application scripts. E.g., `apps/drug/energy_train.py` handles the training loop for drug-protein binding energy prediction.

## 2. Walkthrough of the Core Models

### `frame.py` (The Mathematical Core)
This file implements the `FrameAveraging` base class, handling SE(3) equivariance. Instead of building complex equivariant network layers, DSMBind standardizes the input point clouds (from proteins and ligands) into a common reference frame (derived by computing a covariance matrix and doing eigendecomposition), runs predictions, and maps them back. 
- *If you want to modify how 3D rotation logic or input equivariance works, you start here.*

### `energy.py` (Protein-Protein / Antibody-Antigen Binding)
This file houses `FAEnergyModel` and `AllAtomEnergyModel`:
*   `AllAtomEnergyModel.forward()` adds noise to backbones and side chains (score matching), and then predicts the score.
*   The models calculate energy maps internally based on pair-wise distance cutoffs (`self.threshold`), then use `torch.autograd.grad` to pull out forces (gradients of energy with respect to coordinates).
*   *If you want to change how energy is aggregated or how protein-protein scoring works, edit this file.*

### `drug.py` (Protein-Ligand Binding)
Since you currently have `bindenergy/models/drug.py` open, let's take a closer look:
*   It combines the 3D frame averaging for the protein side `FAEncoder`/`AllAtomEncoder` with a **Message Passing Neural Network (`MPNEncoder`)** from Chemprop for the 2D ligand graph.
*   `DrugEnergyModel` & `DrugAllAtomEnergyModel` predict how strongly a small molecule (drug/ligand) binds to the protein target.
*   `virtual_screen()`: Built-in utility to scan through an SDF files list of ligands and score them against a single protein PDB target.
*   *If your future changes involve the **drug/ligand representation** (e.g., adding 3D conformations for ligands, or changing the graph convolution logic), you will be editing `drug.py`.*

## 3. How to Make Future Changes

As a senior developer looking to modify this model, here are the paths you would take depending on your goal:

### Adding New Features to Protein Representation
1.  **Modify Data**: Edit `bindenergy/data/loader.py` and `bindenergy/data/protein.py` to extract your new features from PDB/SDF files and add them to the batch tensor.
2.  **Update Encoders**: Modify `bindenergy/models/frame.py` (`FAEncoder` or `AllAtomEncoder`). Note the `self.encoder` SRUpp inputs dimensions. You will need to increase the input dimension to accommodate your new features.

### Adjusting the Loss Function / Adding Decoys
The project currently relies strongly on *Denoising Score Matching Loss* (`wloss` for angular velocity, `tloss` for translation, `uloss` for sidechains).
1.  **Modify the Loss**: Go into the `forward` function of the respective energy model (e.g., `DrugAllAtomEnergyModel` in `drug.py`). Add your custom loss terms to the return statement.
2.  **Update `apps`**: Make sure the training scripts (like `apps/drug/energy_train.py`) pass any necessary new labels.

### Experimenting with Different Ligand Graph Models
If you want to swap out the Chemprop-based `MPNEncoder` for something else (e.g., a GAT, GIN, or 3D coordinate-aware network for ligands):
1.  Rewrite the `MPNEncoder` class in `drug.py`.
2.  Ensure that its output (`bind_S`) produces node embeddings compatible with the concatenation step in the `DrugEnergyModel`.

## 4. Next Steps for Onboarding
1.  **Read the Data Pipeline**: Review exactly what `binder` and `target` tuples contain. They typically pack `(Coordinates, Sequence/Graph, Mask, ...)`. 
2.  **Run Inference**: Run through the `tutorial.ipynb`. It's the designated entry point for inference. Add breakpoints using `import pdb; pdb.set_trace()` inside `models/drug.py`'s `predict` function so you can inspect the exact tensor shapes flowing through the attention/pooling layers.
3.  **Local Dev Setup**: Set up your conda environment using the instructions in `README.md`. Make sure Chemprop, SRU++, and ESM-2 are installed as they are external heavy dependencies.

Feel free to let me know which area you'd like to dive into deep first, and we can start refactoring or writing new code together!
