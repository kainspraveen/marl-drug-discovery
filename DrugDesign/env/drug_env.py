import functools
import sys
import os
import random
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box, Dict
from rdkit import Chem
from rdkit.Chem import Draw, RWMol, MolToSmiles, Atom, Bond, AllChem, Descriptors, QED, BondType, MolFromSmarts, RDConfig

# Optional Scorer
try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
except ImportError:
    print("Warning: sascorer not found. SA Score will be 1.0 (ignored).")
    sascorer = None

class DrugDesignEnv(ParallelEnv):
    metadata = {"name": "marl_env_drug_design_v0", "render_modes": ["human"], "is_parallelizable": True}

    def __init__(self):
        super().__init__()
        self.agents = ["scaffold", "functional", "finetune"]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, range(len(self.agents))))

        # Define separate action subsets
        self.agent_actions = {
            "scaffold": [0, 6, 7, 8, 9],   # Add C, Bonds, Rings
            "functional": [1, 2, 3, 4, 6], # Add N, O, S, F, Single Bond
            "finetune": [4, 5, 10]         # Add F, Cl, Do Nothing
        }

        self.MAX_ATOMS = 50
        self.NUM_ATOM_FEATURES = 12 
        self.NUM_BOND_FEATURES = 4 

        self.action_spaces = {agent: Discrete(11) for agent in self.agents}
        self.observation_spaces = {
            agent: Dict({
                "x": Box(low=0, high=1, shape=(self.MAX_ATOMS, self.NUM_ATOM_FEATURES), dtype=np.float32),
                "edge_index": Box(low=0, high=self.MAX_ATOMS, shape=(2, self.MAX_ATOMS * 4), dtype=np.int64),
                "edge_attr": Box(low=0, high=1, shape=(self.MAX_ATOMS * 4, self.NUM_BOND_FEATURES), dtype=np.float32),
                "mask": Box(low=0, high=1, shape=(self.MAX_ATOMS,), dtype=np.int8),
                "action_mask": Box(low=0, high=1, shape=(11,), dtype=np.int8),
            })
            for agent in self.agents
        }
        
        self.action_dispatch = {
            0: lambda: self._add_atom('C'),
            1: lambda: self._add_atom('N'),
            2: lambda: self._add_atom('O'),
            3: lambda: self._add_atom('S'),
            4: lambda: self._add_atom('F'),
            5: lambda: self._add_atom('Cl'),
            6: lambda: self._add_bond(BondType.SINGLE),
            7: lambda: self._add_bond(BondType.DOUBLE),
            8: lambda: self._add_bond(BondType.TRIPLE),
            9: self._add_ring,
            10: self._do_nothing,
        }
        self.mol = None

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        # Start with Benzene
        self.mol = RWMol(Chem.MolFromSmiles("c1ccccc1")) 
        Chem.SanitizeMol(self.mol)
        return {agent: self.observe(agent) for agent in self.agents}, {agent: {} for agent in self.agents}

    def step(self, actions):
        for agent in self.agents:
            if agent in actions:
                action_idx = actions[agent]
                self.action_dispatch[action_idx]()
        
        valid_chemistry = True
        try:
            self.mol.UpdatePropertyCache(strict=False)
            Chem.SanitizeMol(self.mol)
        except:
            valid_chemistry = False

        reward = 0.0
        terminated = False
        
        if not valid_chemistry:
            reward = -10.0 
            terminated = True 
        else:
            mol = self.mol
            num_atoms = mol.GetNumAtoms()
            atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
            
            # Constraints
            if num_atoms > 0:
                num_carbon = atoms.count('C')
                carbon_ratio = num_carbon / num_atoms
                if carbon_ratio < 0.5: reward -= 2.0
            
            num_halogens = atoms.count('Cl') + atoms.count('F') + atoms.count('Br')
            num_sulfur = atoms.count('S')
            
            if num_halogens > 3: reward -= 0.5 * (num_halogens - 3)
            if num_sulfur > 2:   reward -= 0.5 * (num_sulfur - 2)

            # Metrics
            try:
                qed_score = QED.qed(mol)
                reward += qed_score * 2.0
            except: pass

            if sascorer:
                try:
                    sa = sascorer.calculateScore(mol)
                    if sa < 4.0: reward += 1.0
                    elif sa > 5.0: reward -= (sa - 5.0) * 0.5
                except: pass

            mw = Descriptors.MolWt(mol)
            if mw < 200: reward -= 1.0
            elif mw > 600: reward -= 1.0
            else: reward += 1.0

            if not self._check_unwanted_substructures(mol):
                reward -= 5.0

        if self.mol.GetNumAtoms() >= self.MAX_ATOMS:
            terminated = True
            
        rewards = {agent: reward for agent in self.agents}
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def observe(self, agent):
        self.mol.UpdatePropertyCache(strict=False)
        mol = self.mol.GetMol()
        
        x = np.zeros((self.MAX_ATOMS, self.NUM_ATOM_FEATURES), dtype=np.float32)
        mask = np.zeros((self.MAX_ATOMS,), dtype=np.int8)
        max_edges = self.MAX_ATOMS * 4 
        edge_index = np.full((2, max_edges), -1, dtype=np.int64)
        edge_attr = np.zeros((max_edges, self.NUM_BOND_FEATURES), dtype=np.float32)

        for i, atom in enumerate(mol.GetAtoms()):
            if i >= self.MAX_ATOMS: break
            x[i] = self._get_atom_features(atom)
            mask[i] = 1 

        edge_count = 0
        for bond in mol.GetBonds():
            if edge_count >= max_edges: break
            idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index[0, edge_count], edge_index[1, edge_count] = idx1, idx2
            edge_attr[edge_count] = self._get_bond_features(bond)
            edge_count += 1
            if edge_count < max_edges:
                edge_index[0, edge_count], edge_index[1, edge_count] = idx2, idx1
                edge_attr[edge_count] = self._get_bond_features(bond)
                edge_count += 1
        
        action_mask = self._get_action_mask(agent)
        
        return {"x": x, "edge_index": edge_index, "edge_attr": edge_attr, "mask": mask, "action_mask": action_mask}
    
    def _add_atom(self, atom_symbol):
        if self.mol.GetNumAtoms() >= 50: return
        new_idx = self.mol.AddAtom(Atom(atom_symbol))
        available_atoms = [atom.GetIdx() for atom in self.mol.GetAtoms() 
                           if atom.GetIdx() != new_idx and self._get_free_valence(atom) > 0]
        if available_atoms:
            target_idx = random.choice(available_atoms)
            self.mol.AddBond(new_idx, target_idx, BondType.SINGLE)
        else:
            self.mol.RemoveAtom(new_idx)

    def _add_bond(self, bond_type):
        num_atoms = self.mol.GetNumAtoms()
        if num_atoms < 2: return
        order_map = {BondType.SINGLE: 1, BondType.DOUBLE: 2, BondType.TRIPLE: 3}
        required_valence = order_map.get(bond_type, 1)
        possible_bonds = []
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                if self.mol.GetBondBetweenAtoms(i, j) is None:
                    atom_i = self.mol.GetAtomWithIdx(i)
                    atom_j = self.mol.GetAtomWithIdx(j)
                    if (self._get_free_valence(atom_i) >= required_valence and 
                        self._get_free_valence(atom_j) >= required_valence):
                        possible_bonds.append((i, j))
        if possible_bonds:
            idx1, idx2 = random.choice(possible_bonds)
            self.mol.AddBond(idx1, idx2, bond_type)

    def _add_ring(self):
        if self.mol.GetNumAtoms() >= self.MAX_ATOMS - 6: return
        available = [a.GetIdx() for a in self.mol.GetAtoms() if self._get_free_valence(a) >= 1]
        if not available: return
        anchor_idx = random.choice(available)
        ring_indices = []
        for _ in range(6):
            atom = Atom('C')
            atom.SetIsAromatic(True) 
            idx = self.mol.AddAtom(atom)
            ring_indices.append(idx)
        for i in range(6):
            self.mol.AddBond(ring_indices[i], ring_indices[(i+1)%6], BondType.AROMATIC)
        self.mol.AddBond(anchor_idx, ring_indices[0], BondType.SINGLE)

    def _do_nothing(self): pass

    def _get_atom_features(self, atom):
        possible_atoms = ['C', 'N', 'O', 'F', 'S', 'Cl']
        atom_type = [0] * (len(possible_atoms) + 1)
        symbol = atom.GetSymbol()
        if symbol in possible_atoms: atom_type[possible_atoms.index(symbol)] = 1
        else: atom_type[-1] = 1
        is_aromatic = [1 if atom.GetIsAromatic() else 0]
        hybridization = [0] * 4
        hyb = str(atom.GetHybridization())
        if 'SP3' in hyb: hybridization[0] = 1
        elif 'SP2' in hyb: hybridization[1] = 1
        elif 'SP' in hyb: hybridization[2] = 1
        else: hybridization[3] = 1
        return np.array(atom_type + is_aromatic + hybridization, dtype=np.float32)

    def _get_bond_features(self, bond):
        bond_type = bond.GetBondType()
        features = [0] * 4
        if bond_type == BondType.SINGLE: features[0] = 1
        elif bond_type == BondType.DOUBLE: features[1] = 1
        elif bond_type == BondType.TRIPLE: features[2] = 1
        elif bond_type == BondType.AROMATIC: features[3] = 1
        return np.array(features, dtype=np.float32)
    
    def _check_unwanted_substructures(self, mol):
        if mol.HasSubstructMatch(MolFromSmarts('[N]-[Cl]')): return False
        if mol.HasSubstructMatch(MolFromSmarts('[N]-[F]')): return False
        if mol.HasSubstructMatch(MolFromSmarts('[O]-[O]')): return False
        if mol.HasSubstructMatch(MolFromSmarts('CCCCCCCC')): return False
        return True

    def _get_action_mask(self, agent):
        mask = np.zeros(11, dtype=np.int8)
        allowed_actions = self.agent_actions[agent]
        for action_idx in allowed_actions: mask[action_idx] = 1
        if self.mol.GetNumAtoms() >= self.MAX_ATOMS:
            mask[0:6] = 0; mask[9] = 0
        if self.mol.GetNumAtoms() < 2: mask[6:9] = 0
        has_free_valence = any(self._get_free_valence(atom) > 0 for atom in self.mol.GetAtoms())
        if not has_free_valence: mask[6:9] = 0
        return mask
    
    def render(self):
        self.mol.UpdatePropertyCache(strict=False)
        smiles = MolToSmiles(self.mol)
        print(f"Current Molecule: {smiles}")
        try:
            mol = self.mol.GetMol()
            if mol.GetNumAtoms() > 0:
                img = Draw.MolToImage(mol)
                img.save("current_molecule.png")
        except: pass

    def _get_free_valence(self, atom):
        pt = AllChem.GetPeriodicTable()
        return pt.GetDefaultValence(atom.GetAtomicNum()) - atom.GetExplicitValence()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent): return self.observation_spaces[agent]
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent): return self.action_spaces[agent]