# envs/core/reward.py

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem import QED


def qed_score(mol: Chem.Mol) -> float:
    """
    Compute a simple drug-likeness score using RDKit's QED.

    Returns:
      float in [0, 1] typically (higher is "more drug-like")
    """
    if mol is None:
        return 0.0
    try:
        return float(QED.qed(mol))
    except Exception:
        # If RDKit can't compute descriptors for some reason, be safe.
        return 0.0


def terminal_reward(mol: Chem.Mol, *, size_penalty: float = 0.0) -> float:
    """
    Terminal-only reward used for the RL episode.

    - Base reward: QED(mol)
    - Optional penalty: discourage very large molecules (simple regularizer)

    size_penalty:
      If > 0, subtract size_penalty * num_atoms.
    """
    if mol is None:
        return 0.0

    r = qed_score(mol)

    if size_penalty > 0.0:
        r -= float(size_penalty) * float(mol.GetNumAtoms())

    return r
