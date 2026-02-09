from pettingzoo.test import parallel_api_test
from env.drug_env import DrugDesignEnv  # Ensure this matches your folder structure
import numpy as np
import gymnasium as gym

def test_environment():
    print("1. Initializing Environment...")
    env = DrugDesignEnv()
    
    # 2. API Compliance Test (PettingZoo Standard)
    print("\n2. Running PettingZoo API Test...")
    try:
        parallel_api_test(env, num_cycles=100)
        print("   >> PASS: Environment complies with Parallel API.")
    except Exception as e:
        print(f"   >> FAIL: API Test failed: {e}")
        # Continue to debug manually
        
    # 3. Manual Deep Inspection of Graph Tensors
    print("\n3. Deep Inspection of Graph Observations...")
    observations, infos = env.reset()
    
    # Inspect the first agent's observation
    agent_id = env.agents[0]
    obs = observations[agent_id]
    
    print(f"   Agent: {agent_id}")
    print(f"   Keys found: {list(obs.keys())}")
    
    # --- Check Node Features (x) ---
    x = obs['x']
    print(f"   [x] Node Features Shape: {x.shape} (Expected: {env.MAX_ATOMS}, {env.NUM_ATOM_FEATURES})")
    
    # --- Check Edge Index ---
    edge_index = obs['edge_index']
    print(f"   [edge_index] Shape: {edge_index.shape} (Expected: 2, {env.MAX_ATOMS * 4})")
    
    # --- Check Mask ---
    mask = obs['mask']
    num_real_atoms = np.sum(mask)
    print(f"   [mask] Real Atoms: {num_real_atoms} / {env.MAX_ATOMS}")

    # --- NEW: Check Action Mask Logic ---
    action_mask = obs['action_mask']
    print(f"   [action_mask] Shape: {action_mask.shape} (Expected: 11,)")
    print(f"   [action_mask] Values: {action_mask}")
    
    # Logic Check: At step 0, we have 1 atom (Carbon). 
    # We CANNOT add bonds (indices 6, 7, 8). 
    # We SHOULD be able to add atoms (indices 0-5).
    if action_mask[6] == 0 and action_mask[7] == 0 and action_mask[0] == 1:
        print("   >> PASS: Action Mask correctly disabled Bond actions for single atom.")
    else:
        print("   >> FAIL: Action Mask logic is incorrect for start state.")

    # 4. Simulation Loop
    print("\n4. Running Simulation Loop (Random Actions)...")
    
    for i in range(5):
        print(f"\n--- Step {i+1} ---")
        
        # Pick random actions
        # masked_sample() isn't standard in Gym, so we just sample raw
        # In a real training loop, you would use the mask to zero-out logits
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        
        # Act
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Render (Save image)
        env.render()
        
        # Quick check if graph grew
        first_agent = env.agents[0]
        if first_agent in observations:
            curr_atoms = np.sum(observations[first_agent]['mask'])
            curr_reward = rewards[first_agent]
            
            # Check if we got a negative reward (likely due to invalid chemistry or filter)
            status = "GOOD" if curr_reward > 0 else "PENALIZED"
            print(f"   Atoms: {curr_atoms} | Reward: {curr_reward:.2f} ({status})")
            
            # Print the mask to see if it changes (e.g., if we hit 50 atoms)
            # print(f"   New Mask: {observations[first_agent]['action_mask']}")
        
        if any(terminations.values()) or any(truncations.values()):
            print("   >> Environment Terminated/Truncated.")
            break

    print("\n5. Success! Environment is runnable.")

if __name__ == "__main__":
    test_environment()