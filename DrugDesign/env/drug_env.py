"""
Multi-agent drug design environment using PettingZoo

"""
import functools
import sys
import os
import random
from copy import copy
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, RWMol, MolToSmiles, Atom, Bond, AllChem, rdFingerprintGenerator, MolFromSmiles, Descriptors, Crippen, QED, BondType, MolFromSmarts
from rdkit.Chem import FilterCatalog, RDConfig
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium.spaces import Discrete, Box, Dict

try:
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
except ImportError:
    print("Warning: sascorer not found. SA Score will be 1.0 (ignored).")
    sascorer = None

class DrugDesignEnv(ParallelEnv):
    """
    A multi-agent environment for drug design using reinforcement learning.
    Each agent represents a different component of the drug design process.
    """
    metadata = {"name": "marl_env_drug_design_v0", "render_modes": ["human"], "is_parallelizable": True}

    FRAGMENTS = [
        "C",        # Methyl
        "CC",       # Ethyl  
        "c1ccccc1", # Benzene
        "C(=O)O",   # Carboxylic acid
        "C(=O)N",   # Amide
        "O",        # Hydroxyl
        "N",        # Amine
        "F",        # Fluorine
        "S",        # Thiol
        "C1CCCCC1", # Cyclohexane
    ]
    ACTION_SPACE_OPTIONAL = {
        "affinity": {  # Optimize binding affinity
            0: "add_hydrophobic_group",    # -CH3, phenyl
            1: "add_hbond_donor",          # -OH, -NH2
            2: "add_hbond_acceptor",       # =O, -O-
            3: "add_aromatic_ring",
            4: "increase_rigidity",
            5: "do_nothing",
        },
        "toxicity": {  # Reduce toxicity
            0: "remove_reactive_group",    # Remove aldehydes, epoxides
            1: "add_solubility_group",     # Add -OH, -COOH
            2: "block_metabolism_site",    # Protect vulnerable positions
            3: "reduce_lipophilicity",
            4: "do_nothing",
        },
        "sa": {  # Improve synthetic accessibility
            0: "simplify_structure",       # Remove complex stereocenters
            1: "use_common_building_block",
            2: "reduce_ring_complexity",
            3: "add_standard_linker",
            4: "do_nothing",
        },
    }

    def __init__(self, num_agents=None):
        # self.num_agents = num_agents
        # # self.agents = [f"agent_{i}" for i in range(num_agents)]
        # self.agents = ["designer_1", "designer_2", "designer_3", "editor"]
        # self.state = self.reset()
        # self.max_steps = 100
        # self.current_step = 0
        super().__init__()
        self.agents = ["affinity", "toxicity", "sa"]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, range(len(self.agents))))

        # === CONSTANTS ===
        self.MAX_ATOMS = 50
        self.NUM_ATOM_FEATURES = 12  # 7 (types) + 1 (aromatic) + 4 (hybridization)
        self.NUM_BOND_FEATURES = 4   # Single, Double, Triple, Aromatic

        # Action: 0-10 (add atoms, bonds, etc.)
        self.action_spaces = {agent: Discrete(11) for agent in self.agents}

        # We use a Dictionary space to hold Node features and Edge Indices
        self.observation_spaces = {
            agent: Dict({
                # Node Features: (Max_Atoms, Num_Features)
                "x": Box(low=0, high=1, shape=(self.MAX_ATOMS, self.NUM_ATOM_FEATURES), dtype=np.float32),
                
                # Adjacency Matrix / Edge Index: (2, Max_Edges) 
                # We allocate space for a fully connected graph as worst case (Max_Atoms * Max_Atoms)
                # But typically padding with -1 or handle in the 'observe' function
                "edge_index": Box(low=0, high=self.MAX_ATOMS, shape=(2, self.MAX_ATOMS * 4), dtype=np.int64),
                
                # Edge Attributes: (Max_Edges, Num_Bond_Features)
                "edge_attr": Box(low=0, high=1, shape=(self.MAX_ATOMS * 4, self.NUM_BOND_FEATURES), dtype=np.float32),
                
                # Mask: To tell the GNN which nodes are real and which are padding
                "mask": Box(low=0, high=1, shape=(self.MAX_ATOMS,), dtype=np.int8),

                # Action Mask (1=Valid, 0=Invalid)
                # Size = 11 (number of actions in your Discrete space)
                "action_mask": Box(low=0, high=1, shape=(11,), dtype=np.int8),
            })
            for agent in self.agents
        }
        
        # Map Actions to Chemistry Methods
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

    def reset(self, seed = None, options = None): #options input incase we try to use standard wrappers
        """
        Reset to a single Carbon atom
        """
        self.agents = self.possible_agents[:]
        self.mol = RWMol()
        self.mol.AddAtom(Atom("C")) #Start with single Carbon atom => RdKit will add 4-H automatically because Valence of 4

        print("=== RESET MOLECULE ===")

        #Return observations and infos
        return {agent: self.observe(agent) for agent in self.agents}, {agent: {} for agent in self.agents}

    def step(self, actions):
        """
        Execute one step where ALL agents act simultaneously.
        """
        # Apply Actions sequentially (Priority: Affinity -> Toxicity -> SA)
        # Gives priority to specific agents
        for agent in self.agents:
            if agent in actions:
                action_idx = actions[agent]
                #Apply chemical change
                self.action_dispatch[action_idx]()
        
        self.mol.UpdatePropertyCache(strict=False)
        
        # Calculate Global Reward 
        # Must check QED, SASTc, LogP, Molecular Weight, etc.
        current_smiles = MolToSmiles(self.mol)
        qed = QED.qed(self.mol)
        #sastc = SA.sascore(self.mol)
        logp = Crippen.MolLogP(self.mol)
        mw = Descriptors.MolWt(self.mol)
        
        # Check if if molecule is valid (ex: Carbon with 5 bonds not accepted. We punish RL model for this)
        try:
            Chem.SanitizeMol(self.mol)
            valid_chemistry = True
        except:
            valid_chemistry = False

        reward = 0.0

        # Calculate Rewards
        if not valid_chemistry:
            reward = -5.0 #Penalize heavily for invalid molecules
        elif not self._check_unwanted_substructures(self.mol):
            reward = -5.0 #Penalize toxic / unstablle substructures
        else: 
            reward += 0.5 #Reward for valid chemistry
            
            # QED (Drug-likeness) [0.0 - 1.0]
            qed_score = QED.qed(self.mol)
            reward += qed_score * 2.0 

            # SA Score (Synthetic Accessibility) [1 (Easy) -> 10 (Hard)]
            if sascorer:
                sa = sascorer.calculateScore(self.mol)
                # Normalize: We want low SA. 
                # If SA > 5 (Hard), penalize. If SA < 3 (Easy), reward.
                reward -= (sa - 3.0) * 0.3 

            # Molecular Weight (Target: 200-500 Da)
            mw = Descriptors.MolWt(self.mol)
            if mw < 200: reward -= 0.5 # Too small
            elif mw > 500: reward -= 0.5 # Too big
            else: reward += 1.0 # Sweet spot

        rewards = {agent: reward for agent in self.agents}

        # Termination/Truncation : Stop if molecule is too big
        terminations = {agent: self.mol.GetNumAtoms() > 50 for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        # New Obeservations
        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    
    def observe(self, agent):
        self.mol.UpdatePropertyCache(strict=False)
        mol = self.mol.GetMol()
        
        # Init Empty Arrays (Padding with Zeros)
        x = np.zeros((self.MAX_ATOMS, self.NUM_ATOM_FEATURES), dtype=np.float32)
        mask = np.zeros((self.MAX_ATOMS,), dtype=np.int8)
        
        # Max edges = Max_Atoms * 4 (heuristic for space allocation)
        max_edges = self.MAX_ATOMS * 4 
        edge_index = np.full((2, max_edges), -1, dtype=np.int64) # Fill with -1 for empty
        edge_attr = np.zeros((max_edges, self.NUM_BOND_FEATURES), dtype=np.float32)

        # Fill Node Features
        num_atoms = mol.GetNumAtoms()
        for i, atom in enumerate(mol.GetAtoms()):
            if i >= self.MAX_ATOMS: break
            x[i] = self._get_atom_features(atom)
            mask[i] = 1 # Mark this node as real

        # Fill Edge Features (Adjacency)
        edge_count = 0
        for bond in mol.GetBonds():
            if edge_count >= max_edges: break
            
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            
            # Add edge (u, v)
            edge_index[0, edge_count] = idx1
            edge_index[1, edge_count] = idx2
            edge_attr[edge_count] = self._get_bond_features(bond)
            edge_count += 1

            # Since graphs are undirected in GNNs usually, we add (v, u) as well
            if edge_count < max_edges:
                edge_index[0, edge_count] = idx2
                edge_index[1, edge_count] = idx1
                edge_attr[edge_count] = self._get_bond_features(bond)
                edge_count += 1
        
        action_mask = self._get_action_mask() # Get the mask
        
        return {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "mask": mask,
            "action_mask": action_mask
        }
    
    def _add_atom(self, atom_symbol):
        """Add a new atom bonded to random existing atom with free valence."""
        if self.mol.GetNumAtoms() >= 50: return

        # Add the new atom
        new_idx = self.mol.AddAtom(Atom(atom_symbol))

        # Find atoms with free valence
        available_atoms = []
        for atom in self.mol.GetAtoms():
            if atom.GetIdx() == new_idx: continue
            atom.UpdatePropertyCache(strict=False)
            # Check if atom has room for bonds
            if self._get_free_valence(atom) > 0:
                available_atoms.append(atom.GetIdx())

        # Bond to one of the atoms that has room for bonds
        if available_atoms:
            target_idx = random.choice(available_atoms)
            self.mol.AddBond(new_idx, target_idx, BondType.SINGLE)
        else:
            # Remove atom that was just added if we cannot bond it
            self.mol.RemoveAtom(new_idx)

    def _add_bond(self, bond_type):
        """Adds a bond between two existing atoms that are not yet connected."""
        num_atoms = self.mol.GetNumAtoms()
        if num_atoms < 2: return

        # Determine bond order requirement
        order_map = {
            BondType.SINGLE: 1,
            BondType.DOUBLE: 2,
            BondType.TRIPLE: 3
        }
        required_valence = order_map.get(bond_type, 1)

        # Get all pairs of atoms that are not already connected
        possible_bonds = []
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                # Check if atoms are not already connected
                if self.mol.GetBondBetweenAtoms(i, j) is None:
                    # Check if both atoms have room for another bond
                    atom_i = self.mol.GetAtomWithIdx(i)
                    atom_j = self.mol.GetAtomWithIdx(j)
                    atom_i.UpdatePropertyCache(strict=False)
                    atom_j.UpdatePropertyCache(strict=False)

                    # Both must have enough room for bonds
                    if (self._get_free_valence(atom_i) >= required_valence and 
                        self._get_free_valence(atom_j) >= required_valence):
                        possible_bonds.append((i, j))

        # Add a bond to one of the possible pairs
        if possible_bonds:
            idx1, idx2 = random.choice(possible_bonds)
            self.mol.AddBond(idx1, idx2, bond_type)

    def _add_ring(self):
        """Adds a simple 6-membered aromatic ring."""
        # Add logic
        pass

    def _do_nothing(self):
        """Do nothing."""
        pass
    def _get_atom_features(self, atom):
        """
        Converts an RDKit atom into a feature vector.
        Features: [Atom Type (One-Hot), Is_Aromatic, Hybridization]
        """
        # Atom Type One-Hot (C, N, O, F, S, Cl, Other)
        possible_atoms = ['C', 'N', 'O', 'F', 'S', 'Cl']
        atom_type = [0] * (len(possible_atoms) + 1)
        symbol = atom.GetSymbol()
        if symbol in possible_atoms:
            atom_type[possible_atoms.index(symbol)] = 1
        else:
            atom_type[-1] = 1 # 'Other' category

        # Properties
        is_aromatic = [1 if atom.GetIsAromatic() else 0]
        
        # Hybridization (SP, SP2, SP3, Other)
        hybridization = [0] * 4
        hyb = str(atom.GetHybridization())
        if 'SP3' in hyb: hybridization[0] = 1
        elif 'SP2' in hyb: hybridization[1] = 1
        elif 'SP' in hyb: hybridization[2] = 1
        else: hybridization[3] = 1

        return np.array(atom_type + is_aromatic + hybridization, dtype=np.float32)

    def _get_bond_features(self, bond):
        """
        Converts an RDKit bond into a feature vector.
        Features: [Bond Type (Single, Double, Triple, Aromatic)]
        """
        bond_type = bond.GetBondType()
        features = [0] * 4
        if bond_type == BondType.SINGLE: features[0] = 1
        elif bond_type == BondType.DOUBLE: features[1] = 1
        elif bond_type == BondType.TRIPLE: features[2] = 1
        elif bond_type == BondType.AROMATIC: features[3] = 1
        return np.array(features, dtype=np.float32)
    
    def _check_unwanted_substructures(self, mol):
        """Rejects molecules with unstable/toxic motifs (e.g. N-Cl, O-O, N-N)."""

        # Reject N-Cl (Chloramines - Unstable/Explosive)
        if mol.HasSubstructMatch(MolFromSmarts('[N]-[Cl]')): return False
        if mol.HasSubstructMatch(MolFromSmarts('[N]-[F]')): return False
        
        # Reject Peroxides (O-O) - Explosive
        if mol.HasSubstructMatch(MolFromSmarts('[O]-[O]')): return False

        # Reject Long Chains (>7 Carbons in a row without branching/heteroatoms)
        # Helps prevent "greasy" nondrugs
        if mol.HasSubstructMatch(MolFromSmarts('CCCCCCCC')): return False
        
        return True

    def _get_action_mask(self):
        """
        Returns a boolean mask of valid actions [1, 0, 1, ...]
        Size = 11 (Action Space Size)
        """
        mask = np.ones(11, dtype=np.int8)
        
        # Cannot add atoms if we hit the limit
        if self.mol.GetNumAtoms() >= self.MAX_ATOMS:
            mask[0:6] = 0 # Disable Adding Atoms (Actions 0-5)
            mask[9] = 0   # Disable Adding Ring
            
        # Cannot add bonds if < 2 atoms
        if self.mol.GetNumAtoms() < 2:
            mask[6:9] = 0 # Disable Adding Bonds (Actions 6-8)

        # Check valences for specific atom types
        # If ALL current atoms are full, we can't add a bond!
        has_free_valence = False
        for atom in self.mol.GetAtoms():
            if self._get_free_valence(atom) > 0:
                has_free_valence = True
                break
        
        if not has_free_valence:
            mask[6:9] = 0 # Disable Bonds if everyone is full
            # Also effectively disables adding atoms because they need a bond site
            # But we leave it "open" so the agent learns to fail

        return mask
    
    def render(self):
        """
        Render the current state.
        Prints SMILES to console and saves an image of the molecule.
        """
        # Get current SMILES
        self.mol.UpdatePropertyCache(strict=False)
        smiles = MolToSmiles(self.mol)
        print(f"Current Molecule: {smiles}")

        # Save Image
        try:
            mol = self.mol.GetMol()
            # Draw valid molecules
            if mol.GetNumAtoms() > 0:
                img = Draw.MolToImage(mol)
                img.save("current_molecule.png")
        except:
            print("Could not draw molecule (invalid state)")

    def _get_free_valence(self, atom):
        """
        Manually calculate free valence to avoid RDKit crash.
        """
        pt = AllChem.GetPeriodicTable()
        # Get standard max valence (e.g. C=4, N=3, O=2)
        atomic_num = atom.GetAtomicNum()
        default_valence = pt.GetDefaultValence(atomic_num)
        
        # Get current bonds
        current_valence = atom.GetExplicitValence()
        
        return default_valence - current_valence


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        Native graph observation space.
        """
        return self.observation_spaces[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """
        Define the action space for a given agent.
        """
        return self.action_spaces[agent]
