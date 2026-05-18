#!/usr/bin/env python3
"""
Convert TCR-pMHC PDB files into the JSONL format expected by DSMBind's
antibody model.  The resulting file can be passed directly to evaluate.py.

Mapping
-------
  TCR alpha-chain  -->  antibody heavy chain  (CDR labels 1 / 2 / 3)
  TCR beta-chain   -->  antibody light chain  (CDR labels 4 / 5 / 6)
  pMHC chains      -->  antigen

CDR labeling modes (--cdr-mode)
--------------------------------
  imgt   Use IMGT residue numbers already present in the PDB to assign CDR loops
         (CDR1: 27-38, CDR2: 56-65, CDR3: 105-117).  Works only when the PDB
         was deposited with IMGT numbering (common for SAbDab/TCRdb structures).
  all    Mark every TCR residue as a CDR loop.  Safe fallback for PDBs with
         sequential or non-IMGT numbering.  (default)

Manifest CSV format
-------------------
  pdb_id,tcr_alpha_chain,tcr_beta_chain,pmhc_chains,affinity
  1ao7,A,B,CD,-8.5
  3dxa,D,E,AB,

Rules
-----
  * At least one TCR chain (alpha or beta) must be present.
  * pmhc_chains is a string of single-character chain IDs, e.g. "CD".
  * affinity is optional; leave blank if unknown.
  * One .pdb file per row, named  <pdb_id>.pdb  inside --pdb-dir.

Example
-------
  python preprocess_tcr_pmhc.py \\
      --manifest data/tcr_pmhc/manifest.csv \\
      --pdb-dir  data/tcr_pmhc/pdbs/ \\
      --output   data/tcr_pmhc/dataset.jsonl

Then evaluate with:
  python evaluate.py --task antibody \\
      --data data/tcr_pmhc/dataset.jsonl \\
      --checkpoint ckpts/model.antibody.allatom
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Residue name tables (from DSMBind constants)
RESTYPE_1to3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
RESTYPE_3to1 = {v: k for k, v in RESTYPE_1to3.items()}
ALPHABET = ['#', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
            'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# 14 atoms per residue in the exact order DSMBind expects
RES_ATOM14 = [
    [''] * 14,
    ['N','CA','C','O','CB','','','','','','','','',''],
    ['N','CA','C','O','CB','CG','CD','NE','CZ','NH1','NH2','','',''],
    ['N','CA','C','O','CB','CG','OD1','ND2','','','','','',''],
    ['N','CA','C','O','CB','CG','OD1','OD2','','','','','',''],
    ['N','CA','C','O','CB','SG','','','','','','','',''],
    ['N','CA','C','O','CB','CG','CD','OE1','NE2','','','','',''],
    ['N','CA','C','O','CB','CG','CD','OE1','OE2','','','','',''],
    ['N','CA','C','O','','','','','','','','','',''],
    ['N','CA','C','O','CB','CG','ND1','CD2','CE1','NE2','','','',''],
    ['N','CA','C','O','CB','CG1','CG2','CD1','','','','','',''],
    ['N','CA','C','O','CB','CG','CD1','CD2','','','','','',''],
    ['N','CA','C','O','CB','CG','CD','CE','NZ','','','','',''],
    ['N','CA','C','O','CB','CG','SD','CE','','','','','',''],
    ['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ','','',''],
    ['N','CA','C','O','CB','CG','CD','','','','','','',''],
    ['N','CA','C','O','CB','OG','','','','','','','',''],
    ['N','CA','C','O','CB','OG1','CG2','','','','','','',''],
    ['N','CA','C','O','CB','CG','CD1','CD2','NE1','CE2','CE3','CZ2','CZ3','CH2'],
    ['N','CA','C','O','CB','CG','CD1','CD2','CE1','CE2','CZ','OH','',''],
    ['N','CA','C','O','CB','CG1','CG2','','','','','','',''],
]


# ── CDR labeling ──────────────────────────────────────────────────────────────

def _imgt_cdr(resnum: int) -> str:
    """Map an IMGT residue number to a CDR label character ('1','2','3') or '0'."""
    if 27 <= resnum <= 38:
        return '1'
    if 56 <= resnum <= 65:
        return '2'
    if 105 <= resnum <= 117:
        return '3'
    return '0'


# ── Biotite-based coordinate extraction ───────────────────────────────────────

def _load_structure(pdb_path: str):
    """Return a biotite AtomArray for model 1 of the PDB file."""
    try:
        import biotite.structure.io.pdb as pdb_io
    except ImportError:
        sys.exit("ERROR: biotite is required.  Install with:  pip install biotite")
    pdb_file = pdb_io.PDBFile.read(pdb_path)
    return pdb_file.get_structure(model=1)


def _residue_coords(res_atoms, one_letter: str) -> list:
    """
    Return a list of 14 [x, y, z] coordinates for one residue, ordered by
    RES_ATOM14.  Missing atoms are represented as [0, 0, 0].
    """
    res_idx     = ALPHABET.index(one_letter)
    atom14_names = RES_ATOM14[res_idx]

    # Build a name→coord lookup for atoms present in this residue
    coord_map = {}
    for i in range(len(res_atoms)):
        name = res_atoms.atom_name[i]
        coord_map[name] = res_atoms.coord[i].tolist()

    return [
        coord_map.get(name, [0.0, 0.0, 0.0]) if name else [0.0, 0.0, 0.0]
        for name in atom14_names
    ]


def _parse_tcr_chain(structure, chain_id: str, chain_role: str, cdr_mode: str):
    """
    Extract sequence, 14-atom coordinates, and CDR labels for one TCR chain.

    chain_role : 'alpha' → CDR labels 1/2/3; 'beta' → CDR labels 4/5/6
    Returns (seq_str, coords_list, cdr_str) or raises ValueError.
    """
    import biotite.structure as struc

    chain_atoms = structure[structure.chain_id == chain_id]
    chain_atoms = chain_atoms[struc.filter_amino_acids(chain_atoms)]
    if len(chain_atoms) == 0:
        raise ValueError(f"Chain '{chain_id}' has no standard amino-acid residues.")

    res_ids, res_names = struc.get_residues(chain_atoms)

    seq, coords, cdr = [], [], []
    for res_id, res_name in zip(res_ids, res_names):
        one_letter = RESTYPE_3to1.get(res_name)
        if one_letter is None or one_letter not in ALPHABET[1:]:
            continue  # skip non-standard residues

        res_atoms = chain_atoms[chain_atoms.res_id == res_id]
        seq.append(one_letter)
        coords.append(_residue_coords(res_atoms, one_letter))

        if cdr_mode == "imgt":
            label = _imgt_cdr(int(res_id))
        else:                    # "all" — every residue is treated as CDR
            label = '3'

        # β-chain CDR labels shift from 1/2/3 to 4/5/6
        if chain_role == "beta" and label != '0':
            label = str(int(label) + 3)
        cdr.append(label)

    if not seq:
        raise ValueError(f"Chain '{chain_id}' produced an empty sequence after filtering.")

    return ''.join(seq), coords, ''.join(cdr)


def _parse_pmhc_chains(structure, chain_ids: list):
    """
    Concatenate all pMHC chains into a single antigen sequence + coordinate list.
    Returns (seq_str, coords_list).
    """
    import biotite.structure as struc

    all_seq, all_coords = [], []
    for chain_id in chain_ids:
        chain_atoms = structure[structure.chain_id == chain_id]
        chain_atoms = chain_atoms[struc.filter_amino_acids(chain_atoms)]
        if len(chain_atoms) == 0:
            print(f"    WARNING: pMHC chain '{chain_id}' is empty — skipped.")
            continue

        res_ids, res_names = struc.get_residues(chain_atoms)
        for res_id, res_name in zip(res_ids, res_names):
            one_letter = RESTYPE_3to1.get(res_name)
            if one_letter is None or one_letter not in ALPHABET[1:]:
                continue
            res_atoms = chain_atoms[chain_atoms.res_id == res_id]
            all_seq.append(one_letter)
            all_coords.append(_residue_coords(res_atoms, one_letter))

    if not all_seq:
        raise ValueError(f"pMHC chains {chain_ids} produced no valid residues.")

    return ''.join(all_seq), all_coords


# ── Per-entry processing ───────────────────────────────────────────────────────

def process_entry(row: dict, pdb_dir: str, cdr_mode: str) -> dict | None:
    pdb_id      = row["pdb_id"].strip()
    pdb_path    = str(Path(pdb_dir) / f"{pdb_id}.pdb")
    alpha_chain = row.get("tcr_alpha_chain", "").strip()
    beta_chain  = row.get("tcr_beta_chain",  "").strip()
    pmhc_str    = row.get("pmhc_chains",     "").strip()
    affinity    = row.get("affinity",        "").strip()

    if not Path(pdb_path).exists():
        print(f"  SKIP {pdb_id}: PDB file not found at {pdb_path}")
        return None

    if not alpha_chain and not beta_chain:
        print(f"  SKIP {pdb_id}: no TCR chain specified.")
        return None

    pmhc_chain_ids = list(pmhc_str)           # "CD" → ['C', 'D']
    if not pmhc_chain_ids:
        print(f"  SKIP {pdb_id}: pmhc_chains is empty.")
        return None

    try:
        structure = _load_structure(pdb_path)
    except Exception as e:
        print(f"  SKIP {pdb_id}: could not load PDB — {e}")
        return None

    ab_seq, ab_coords, ab_cdr = "", [], ""

    for chain_id, role in [(alpha_chain, "alpha"), (beta_chain, "beta")]:
        if not chain_id:
            continue
        try:
            seq, coords, cdr = _parse_tcr_chain(structure, chain_id, role, cdr_mode)
            ab_seq    += seq
            ab_coords += coords
            ab_cdr    += cdr
        except Exception as e:
            print(f"  SKIP {pdb_id}: TCR {role}-chain '{chain_id}' — {e}")
            return None

    if not ab_seq:
        print(f"  SKIP {pdb_id}: TCR chains are empty after parsing.")
        return None

    try:
        ag_seq, ag_coords = _parse_pmhc_chains(structure, pmhc_chain_ids)
    except Exception as e:
        print(f"  SKIP {pdb_id}: pMHC parsing failed — {e}")
        return None

    entry = {
        "pdb":            pdb_id,
        "antibody_seq":   ab_seq,
        "antibody_cdr":   ab_cdr,
        "antibody_coords": ab_coords,
        "antigen_seq":    ag_seq,
        "antigen_coords": ag_coords,
    }
    if affinity:
        try:
            entry["affinity"] = float(affinity)
        except ValueError:
            print(f"  WARNING {pdb_id}: affinity '{affinity}' is not a number — ignored.")

    return entry


# ── Entry point ───────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Preprocess TCR-pMHC PDB files for DSMBind",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--manifest", required=True, metavar="CSV",
                   help="CSV file with columns: pdb_id, tcr_alpha_chain, "
                        "tcr_beta_chain, pmhc_chains[, affinity]")
    p.add_argument("--pdb-dir", required=True, metavar="DIR",
                   help="Directory containing PDB files named <pdb_id>.pdb")
    p.add_argument("--output", required=True, metavar="JSONL",
                   help="Output JSONL file path")
    p.add_argument("--cdr-mode", default="all", choices=["imgt", "all"],
                   help="CDR labeling mode.  'imgt' requires IMGT-numbered PDBs.  "
                        "'all' includes every TCR residue as paratope (default: all).")
    return p


def main():
    args = build_parser().parse_args()

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    with open(out_path, "w") as out_f:
        for row in tqdm(rows, desc="Processing PDB files"):
            entry = process_entry(row, args.pdb_dir, args.cdr_mode)
            if entry is not None:
                out_f.write(json.dumps(entry) + "\n")
                n_ok += 1

    print(f"\nWrote {n_ok}/{len(rows)} entries → {out_path}")
    if n_ok == 0:
        print("WARNING: no entries were written.  Check your manifest and PDB files.")
    else:
        print(f"\nNext step:")
        print(f"  python evaluate.py --task antibody \\")
        print(f"      --data {out_path} \\")
        print(f"      --checkpoint ckpts/model.antibody.allatom")


if __name__ == "__main__":
    main()
