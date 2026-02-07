# envs/core/molecule_ops.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from rdkit import Chem
from rdkit.Chem import rdchem

BondType = rdchem.BondType

@dataclass
class EditResult:
    ok: bool                 # whether the edit produced a valid molecule
    mol: Chem.Mol            # sanitized (valid) molecule if ok, otherwise best-effort
    reason: str = ""         # error message if not ok


def sanitize_mol(mol: Chem.Mol) -> EditResult:
    """
    RDKit validity check:
    - Chem.SanitizeMol(...) throws if valence/aromaticity/etc is invalid.
    """
    try:
        Chem.SanitizeMol(mol)
        return EditResult(True, mol, "")
    except Exception as e:
        return EditResult(False, mol, str(e))


def start_molecule(initial_atom: str = "C") -> Chem.RWMol:
    """
    Create a small starting molecule (1 atom) to avoid 'empty molecule' edge cases.
    Editing is done with RWMol (editable), per RDKit conventions.
    """
    rw = Chem.RWMol()
    rw.AddAtom(Chem.Atom(initial_atom))
    return rw


def add_atom_with_bond(
    rw_mol: Chem.RWMol,
    attach_idx: int,
    new_atom_symbol: str,
    bond_type: BondType,
) -> EditResult:
    """
    Safe edit: copy -> mutate -> sanitize.
    If sanitize fails, we return ok=False and don't mutate original rw_mol.
    """
    rw = Chem.RWMol(rw_mol)  # copy (so we can safely reject invalid edits)
    new_idx = rw.AddAtom(Chem.Atom(new_atom_symbol))
    rw.AddBond(int(attach_idx), int(new_idx), bond_type)

    mol = rw.GetMol()
    return sanitize_mol(mol)
