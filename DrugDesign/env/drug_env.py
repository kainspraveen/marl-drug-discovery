"""
Multi-agent drug design environment using PettingZoo

"""
import functools
import random
from copy import copy
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box
import numpy as np
from rdkit.Chem import Draw, RWMol, MolToSmiles, Atom, Bond, AllChem, rdFingerprintGenerator, MolFromSmiles, Descriptors, Crippen, QED, BondType
from pettingzoo.utils.agent_selector import agent_selector

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

        # Action: 0-10 (add atoms, bonds, etc.)
        self.action_spaces = {agent: Discrete(11) for agent in self.agents}

        #Obeservation: 2048-bit fingerprint
        self.observation_spaces = {
            agent: Box(low=0, high=20, shape=(2048,), dtype=np.float32)
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

        reward = 0

        # Calculate Rewards
        if not valid_chemistry:
            reward = -10.0 #Penalize heavily for invalid molecules
        else: 
            reward += 1 #Reward for valid chemistry
        try:
            # Use continuous values instead of thresholds for faster learning
            # QED is 0.0 to 1.0 (Higher is better)
            qed_score = QED.qed(self.mol)
            reward += qed_score * 2.0  # Scale up QED importance

            # LogP: Target is usually between 0 and 5
            logp = Crippen.MolLogP(self.mol)
            if 0 < logp < 5:
                reward += 1.0
            else:
                # Penalize distance from optimal range
                reward -= 0.1 * min(abs(logp - 0), abs(logp - 5))

            # Molecular Weight: Target < 500
            mw = Descriptors.MolWt(self.mol)
            if mw < 500:
                reward += 1.0
            else:
                reward -= 0.01 * (mw - 500) # Penalize just adding Carbon infinetely

        except Exception as e:
            # If RDKit descriptors crash for any other reason
            print(f"Descriptor calculation failed: {e}")
            reward -= 1.0

        rewards = {agent: reward for agent in self.agents}

        # Termination/Truncation : Stop if molecule is too big
        terminations = {agent: self.mol.GetNumAtoms() > 50 for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        # New Obeservations
        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    
    def observe(self, agent):
        try:
            self.mol.UpdatePropertyCache(strict=False)
            mol = self.mol.GetMol()
            Chem.SanitizeMol(mol)
            fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprintAsNumPy(mol)
            return fp.astype(np.float32)
        except:
            # Fallback for invalid/empty molecules
            return np.zeros((2048,), dtype=np.float32)
    
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
