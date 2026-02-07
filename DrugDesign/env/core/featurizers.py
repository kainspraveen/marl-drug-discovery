# envs/core/featurizers.py

from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

FP_SIZE = 2048


def mol_to_morgan_fp(mol: Chem.Mol, radius: int = 2, fp_size: int = FP_SIZE) -> np.ndarray:
    """
    Convert an RDKit Mol -> Morgan fingerprint vector.

    Returns:
      fp: np.ndarray of shape (fp_size,), dtype float32
    """
    if mol is None:
        return np.zeros((fp_size,), dtype=np.float32)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    fp = gen.GetFingerprint(mol)  # RDKit ExplicitBitVect-like
    arr = np.array(fp, dtype=np.float32)  # shape (fp_size,)
    return arr


def smiles_to_morgan_fp(smiles: str, radius: int = 2, fp_size: int = FP_SIZE) -> np.ndarray:
    """
    Convert a SMILES string -> Morgan fingerprint vector.

    Returns:
      fp: np.ndarray of shape (fp_size,), dtype float32
    """
    mol = Chem.MolFromSmiles(smiles)
    return mol_to_morgan_fp(mol, radius=radius, fp_size=fp_size)
