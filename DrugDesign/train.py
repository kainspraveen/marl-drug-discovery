import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch_geometric.data import Data, Batch
from rdkit import Chem
from env.drug_env import DrugDesignEnv
from gnn_agent import GNNActorCritic

# === HYPERPARAMETERS ===
LR = 0.002
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4
UPDATE_TIMESTEP = 2000
MAX_EPISODES = 5000

class PPOAgent:
    def __init__(self):
        self.gamma = GAMMA
        self.eps_clip = EPS_CLIP
        self.k_epochs = K_EPOCHS
        
        self.env = DrugDesignEnv()
        
        # 12 input features, 11 actions
        self.policy = GNNActorCritic(num_node_features=12, num_actions=11)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.policy_old = GNNActorCritic(num_node_features=12, num_actions=11)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def convert_obs_to_graph(self, obs):
        """Converts dict observation to PyG Batch"""
        x = torch.FloatTensor(obs['x'])
        edge_index = torch.LongTensor(obs['edge_index'])
        action_mask = torch.tensor(obs['action_mask'], dtype=torch.bool).unsqueeze(0)

        # Remove Padding (-1)
        mask = edge_index[0] != -1
        edge_index = edge_index[:, mask]

        data = Data(x=x, edge_index=edge_index)
        batch = Batch.from_data_list([data])
        
        return batch, action_mask

    def update(self, memory):
        # Monte Carlo estimate of rewards
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = memory.states
        old_actions = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()
        old_masks = torch.stack(memory.masks).detach().squeeze(1)

        for _ in range(self.k_epochs):
            logprobs = []
            state_values = []
            dist_entropy = []
            
            for i, state_batch in enumerate(old_states):
                logits, val = self.policy(state_batch.x, state_batch.edge_index, state_batch.batch, old_masks[i].unsqueeze(0))
                dist = torch.distributions.Categorical(logits=logits)
                
                action = old_actions[i]
                logprobs.append(dist.log_prob(action))
                state_values.append(val)
                dist_entropy.append(dist.entropy())
            
            logprobs = torch.stack(logprobs)
            state_values = torch.stack(state_values).squeeze()
            dist_entropy = torch.stack(dist_entropy)
            
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach()

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.masks = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.masks[:]

def train():
    agent = PPOAgent()
    memory = Memory()
    
    best_molecules = []
    output_file = "best_molecules.csv"
    
    print(f"Starting training for {MAX_EPISODES} episodes...")
    
    # HARDCODED ORDER to ensure strategy works as intended
    # 1. Scaffold (Builds Carbon)
    # 2. Functional (Adds N/O)
    # 3. Finetune (Adds Cl/F or Caps)
    agent_cycle = ["scaffold", "functional", "finetune"]
    
    timestep = 0
    
    for i_episode in range(1, MAX_EPISODES+1):
        observations, _ = agent.env.reset()
        ep_reward = 0
        
        for t in range(50):
            timestep += 1
            
            # 1. Select which agent acts this turn
            curr_agent = agent_cycle[t % 3] 
            
            # 2. Get that agent's specific observation (with their specific Action Mask)
            # observation dictionary is updated after every step, so this is always fresh
            state_batch, action_mask = agent.convert_obs_to_graph(observations[curr_agent])
            
            # 3. Run Policy
            action, log_prob, _, val = agent.policy_old.get_action(state_batch.x, state_batch.edge_index, state_batch.batch, action_mask)
            
            # 4. Step Env
            actions = {curr_agent: action.item()}
            obs, rewards, terminations, truncations, _ = agent.env.step(actions)
            
            # 5. Get Reward & Done for the current agent
            reward = rewards[curr_agent]
            done = terminations[curr_agent] or truncations[curr_agent]
            
            # 6. Store in memory
            memory.states.append(state_batch)
            memory.actions.append(action)
            memory.logprobs.append(log_prob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)
            memory.masks.append(action_mask)
            
            # 7. Update Loop Variables
            observations = obs # Update observations for the next turn
            ep_reward += reward
            
            # Update PPO
            if timestep % UPDATE_TIMESTEP == 0:
                print(f"   [UPDATE] Updating PPO Policy at timestep {timestep}...")
                agent.update(memory)
                memory.clear_memory()
                timestep = 0
            
            if done:
                break
        
        # Save results (Only if positive reward)
        if ep_reward > 0.0:
            smiles = Chem.MolToSmiles(agent.env.mol)
            best_molecules.append({"Episode": i_episode, "SMILES": smiles, "Reward": ep_reward})
            
            # Save strictly every 10 successful molecules
            if len(best_molecules) % 10 == 0:
                df = pd.DataFrame(best_molecules)
                df.to_csv(output_file, index=False)
                print(f"   >> Saved {len(best_molecules)} molecules to {output_file}")

        # Logging
        if i_episode % 10 == 0:
            print(f"Episode {i_episode}\t Total Reward: {ep_reward:.2f} \t Steps: {t+1}")
            
            if i_episode % 50 == 0:
                agent.env.render() 
                # Force save just in case
                if len(best_molecules) > 0:
                    df = pd.DataFrame(best_molecules)
                    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    train()