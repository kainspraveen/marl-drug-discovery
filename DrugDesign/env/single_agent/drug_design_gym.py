# DrugDesign/env/single_agent/drug_design_gym.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

from rdkit import Chem
from rdkit.Chem import rdchem

from DrugDesign.env.core.molecule_ops import start_molecule, add_atom_with_bond
from DrugDesign.env.core.featurizers import mol_to_morgan_fp, FP_SIZE
from DrugDesign.env.core.reward import terminal_reward


BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
]
NUM_BOND_CHOICES_PLUS_STOP = len(BOND_TYPES) + 1
STOP_LOCAL = len(BOND_TYPES)  # last local index means STOP


# Simple valence caps (not perfect chemistry; just a practical “avoid obvious invalids”)
MAX_VALENCE = {
    "C": 4,
    "N": 3,   # ignoring [N+] etc for simplicity
    "O": 2,
    "F": 1,
    "S": 6,
    "Cl": 1,
}


def _bond_order(b: rdchem.BondType) -> int:
    if b == rdchem.BondType.SINGLE:
        return 1
    if b == rdchem.BondType.DOUBLE:
        return 2
    if b == rdchem.BondType.TRIPLE:
        return 3
    return 0


@dataclass(frozen=True)
class DecodedAction:
    atom_type_id: int
    attach_idx: int
    bond_or_stop: int  # 0..len(BOND_TYPES) where STOP_LOCAL means stop


class DrugDesignSingleAgentEnv(gym.Env):
    """
    Single-agent molecule building environment.

    Observation:
      - Morgan fingerprint vector of shape (FP_SIZE,)

    Action (Discrete):
      - Flattened index over (atom_type_id, attach_idx, bond_or_stop)
        where bond_or_stop in {0,1,2,STOP_LOCAL} and STOP_LOCAL means STOP.
    """

    metadata = {"render_modes": ["human"]}

    ATOM_TYPES = ["C", "N", "O", "F", "S", "Cl"]

    def __init__(
        self,
        max_atoms: int = 30,
        max_steps: int = 30,
        invalid_penalty: float = 0.1,
        stop_min_atoms: int = 1,
        step_penalty: float = 0.0,
        sanitize_on_reset: bool = True,
    ):
        super().__init__()

        self.max_atoms = int(max_atoms)
        self.max_steps = int(max_steps)
        self.invalid_penalty = float(invalid_penalty)
        self.stop_min_atoms = int(stop_min_atoms)
        self.step_penalty = float(step_penalty)
        self.sanitize_on_reset = bool(sanitize_on_reset)

        self._A = len(self.ATOM_TYPES)
        self._N = self.max_atoms
        self._B = NUM_BOND_CHOICES_PLUS_STOP
        self._num_actions = self._A * self._N * self._B
        self.action_space = Discrete(self._num_actions)

        self.observation_space = Box(low=0.0, high=1.0, shape=(FP_SIZE,), dtype=np.float32)

        self._step_count = 0
        self._rw_mol: Optional[Chem.RWMol] = None
        self._mol: Optional[Chem.Mol] = None

    # -----------------------
    # Encoding / decoding
    # -----------------------
    def _encode(self, a: int, i: int, b: int) -> int:
        return ((a * self._N) + i) * self._B + b

    def _decode(self, idx: int) -> DecodedAction:
        b = idx % self._B
        tmp = idx // self._B
        i = tmp % self._N
        a = tmp // self._N
        return DecodedAction(atom_type_id=int(a), attach_idx=int(i), bond_or_stop=int(b))

    # -----------------------
    # Obs / info
    # -----------------------
    def _get_obs(self) -> np.ndarray:
        assert self._mol is not None
        return mol_to_morgan_fp(self._mol)

    def _get_info(self) -> Dict[str, Any]:
        assert self._mol is not None
        return {"smiles": Chem.MolToSmiles(self._mol), "num_atoms": self._mol.GetNumAtoms()}

    # -----------------------
    # Simple chemistry guardrail
    # -----------------------
    def _used_bond_order_sum(self, atom: Chem.Atom) -> int:
        """
        Version-stable replacement for deprecated RDKit valence helpers:
        sum of bond orders around this atom.
        """
        return sum(_bond_order(b.GetBondType()) for b in atom.GetBonds())

    def _would_exceed_valence(self, attach_atom: Chem.Atom, new_symbol: str, bond_type: rdchem.BondType) -> bool:
        """
        Cheap pre-check to avoid obviously impossible bonds before calling RDKit mutate/sanitize.
        Not chemically perfect, but reduces sanitize failures a lot.
        """
        attach_symbol = attach_atom.GetSymbol()
        if attach_symbol not in MAX_VALENCE or new_symbol not in MAX_VALENCE:
            return False

        bo = _bond_order(bond_type)

        # IMPORTANT: avoid deprecated GetTotalValence/GetExplicitValence calls.
        current_attach_valence = self._used_bond_order_sum(attach_atom)

        # new atom has exactly one new bond of order bo
        new_atom_valence_after = bo

        if current_attach_valence + bo > MAX_VALENCE[attach_symbol]:
            return True
        if new_atom_valence_after > MAX_VALENCE[new_symbol]:
            return True

        # halogens only single bonds
        if new_symbol in {"F", "Cl"} and bo != 1:
            return True

        return False

    # -----------------------
    # Action masks
    # -----------------------
    def get_action_mask(self) -> np.ndarray:
        """
        Gymnasium-compatible mask for Discrete.sample(mask=...):
          - dtype np.int8, shape (num_actions,)
          - 1 allowed, 0 disallowed
        """
        assert self._mol is not None and self._rw_mol is not None
        num_atoms = self._mol.GetNumAtoms()

        mask = np.zeros((self._num_actions,), dtype=np.int8)
        allow_stop = num_atoms >= self.stop_min_atoms

        if num_atoms <= 0:
            if allow_stop:
                for a in range(self._A):
                    for i in range(self._N):
                        mask[self._encode(a, i, STOP_LOCAL)] = 1
            return mask

        valid_attach_max = min(num_atoms, self._N)

        for a in range(self._A):
            new_symbol = self.ATOM_TYPES[a]

            for i in range(valid_attach_max):
                attach_atom = self._mol.GetAtomWithIdx(i)

                if allow_stop:
                    mask[self._encode(a, i, STOP_LOCAL)] = 1

                for b_local, bond_type in enumerate(BOND_TYPES):
                    if not self._would_exceed_valence(attach_atom, new_symbol, bond_type):
                        mask[self._encode(a, i, b_local)] = 1

        return mask

    def action_masks(self) -> np.ndarray:
        """sb3-contrib MaskablePPO hook: boolean mask over Discrete actions."""
        return self.get_action_mask().astype(bool)

    # -----------------------
    # Gymnasium API
    # -----------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0

        self._rw_mol = start_molecule("C")
        mol = self._rw_mol.GetMol()
        if self.sanitize_on_reset:
            Chem.SanitizeMol(mol)
        self._mol = mol

        info = self._get_info()
        info["action_mask"] = self.get_action_mask()
        return self._get_obs(), info

    def step(self, action: int):
        assert self._rw_mol is not None and self._mol is not None

        self._step_count += 1
        terminated = False
        truncated = False

        reward = 0.0
        reward -= self.step_penalty

        decoded = self._decode(int(action))
        atom_type_id = decoded.atom_type_id
        attach_idx = decoded.attach_idx
        bond_or_stop = decoded.bond_or_stop

        num_atoms = self._mol.GetNumAtoms()

        # STOP
        if bond_or_stop == STOP_LOCAL:
            if num_atoms < self.stop_min_atoms:
                info = self._get_info()
                info.update({"invalid": True, "reason": "stop_too_early", "action_mask": self.get_action_mask()})
                return self._get_obs(), -self.invalid_penalty, terminated, truncated, info

            terminated = True
            reward += terminal_reward(self._mol)
            info = self._get_info()
            info["action_mask"] = self.get_action_mask()
            return self._get_obs(), reward, terminated, truncated, info

        # Max steps -> truncate
        if self._step_count >= self.max_steps:
            truncated = True
            reward += terminal_reward(self._mol)
            info = self._get_info()
            info["action_mask"] = self.get_action_mask()
            return self._get_obs(), reward, terminated, truncated, info

        # Attach idx must exist
        if attach_idx >= num_atoms:
            info = self._get_info()
            info.update({"invalid": True, "reason": "attach_idx_out_of_range", "action_mask": self.get_action_mask()})
            return self._get_obs(), -self.invalid_penalty, terminated, truncated, info

        atom_symbol = self.ATOM_TYPES[atom_type_id]
        bond_type = BOND_TYPES[bond_or_stop]

        attach_atom = self._mol.GetAtomWithIdx(attach_idx)
        if self._would_exceed_valence(attach_atom, atom_symbol, bond_type):
            info = self._get_info()
            info.update({"invalid": True, "reason": "valence_cap_precheck", "action_mask": self.get_action_mask()})
            return self._get_obs(), -self.invalid_penalty, terminated, truncated, info

        # Apply edit via helper (does sanitize inside)
        res = add_atom_with_bond(self._rw_mol, attach_idx, atom_symbol, bond_type)

        if not res.ok:
            info = self._get_info()
            info.update({"invalid": True, "reason": res.reason, "action_mask": self.get_action_mask()})
            return self._get_obs(), -self.invalid_penalty, terminated, truncated, info

        self._mol = res.mol
        self._rw_mol = Chem.RWMol(self._mol)

        if self._mol.GetNumAtoms() >= self.max_atoms:
            truncated = True
            reward += terminal_reward(self._mol)

        info = self._get_info()
        info["action_mask"] = self.get_action_mask()
        return self._get_obs(), reward, terminated, truncated, info
