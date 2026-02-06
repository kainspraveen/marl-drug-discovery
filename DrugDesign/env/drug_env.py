"""
Multi-agent drug design environment using PettingZoo

"""
import functools
import random
from copy import copy
from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, Box
import numpy as np
from rdkit.Chem import Draw, RWMol, MolToSmiles, Atom, Bond, AllChem, rdFingerprintGenerator, MolFromSmiles
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

    def __init__(self, num_agents):
        # self.num_agents = num_agents
        # # self.agents = [f"agent_{i}" for i in range(num_agents)]
        # self.agents = ["designer_1", "designer_2", "designer_3", "editor"]
        # self.state = self.reset()
        # self.max_steps = 100
        # self.current_step = 0
        super().__init__()
        self.action_dispatch = {
            0: self._add_carbon,
            1: self._add_nitrogen,
            2: self._add_oxygen,
            3: self._add_sulfur,
            4: self._add_fluorine,
            5: self._add_chlorine,
            6: self._add_single_bond,
            7: self._add_double_bond,
            8: self._add_triple_bond,
            9: self._add_ring,
            10: self._do_nothing,
        }
        self.agents = ["affinity", "toxicity", "sa"]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, range(len(self.agents))))
        self.max_atoms = 50
        self.action_spaces = {agent: Discrete(11) for agent in self.agents}
        self.observation_spaces = {
            agent: Box(low=0, high=20, shape=(2048,), dtype=np.float32)
            for agent in self.agents
        }

    def smiles_to_fingerprint(self, smiles):
        """
        Convert a SMILES string to a molecular fingerprint.
        """
        size = 2048
        mol = MolFromSmiles(smiles)
        if mol is None:
            return np.zeros((size,), dtype=np.float32)
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp = morgan_gen.GetFingerprint(mol)
        return np.array(fp)
    def reset(self):
        """
        Reset the environment to an initial state.
        """
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self.mol = RWMol()
        # self.mol.AddAtom(Atom("C"))   Start with empty molecule

        print("=== RESET ===")
        print("Initial SMILES:", MolToSmiles(self.mol))
        return self.observe(self.agent_selection)

    def step(self, actions):

        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        truncations = {a: False for a in self.agents}


        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self.agent_selection = self._agent_selector.next()
            return

        print(f"\nAgent: {agent} | Action: {actions[agent]}")

        action = actions[agent]
        self.action_dispatch[action]()
        
        print(f"Current SMILES/State: {MolToSmiles(self.mol)}")
        observations = {a: self.observe(a) for a in self.agents}
        self.agent_selection = self._agent_selector.next()
        
        return observations, rewards, terminations, truncations, self.infos

    
    def observe(self, agent):
        """
        Observe the current state for a given agent.
        """
        mol = MolToSmiles(self.mol)
        fingerprint = self.smiles_to_fingerprint(mol)
        return fingerprint
    
    def render(self):
        """
        Render the current state of the environment.
        """
        print("Current state:", self.state)
        # Draw smiles molecules if available
        mol = Draw.MolFromSmiles(self.state)
        img = Draw.MolToImage(mol)
        img.show()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        Native graph observation space.
        """
        return self.observation_space[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """
        Define the action space for a given agent.
        """
        return self.action_spaces[agent]
