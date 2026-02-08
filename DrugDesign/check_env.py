from pettingzoo.test import parallel_api_test
from env.drug_env import DrugDesignEnv
import numpy as np

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
        return

    # 3. Simulation Loop
    print("\n3. Running Simulation Loop (Random Actions)...")
    observations, infos = env.reset()
    
    for i in range(10):
        print(f"\n--- Step {i+1} ---")
        
        # Pick random actions for all agents
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        print(f"Actions: {actions}")
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        env.render()
        
        print(f"Rewards: {rewards}")
        
        if any(terminations.values()) or any(truncations.values()):
            print("   >> Environment Terminated/Truncated.")
            break

    print("\n4. Success! Environment is runnable.")

if __name__ == "__main__":
    test_environment()
