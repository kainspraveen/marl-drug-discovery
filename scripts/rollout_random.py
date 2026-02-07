# scripts/rollout_random.py

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from DrugDesign.env.single_agent.drug_design_gym import DrugDesignSingleAgentEnv


def _mask_summary(mask: np.ndarray) -> Dict[str, Any]:
    mask = np.asarray(mask)
    return {
        "mask_shape": tuple(mask.shape),
        "mask_dtype": str(mask.dtype),
        "num_allowed": int(mask.sum()),
        "num_total": int(mask.size),
        "allowed_frac": float(mask.mean()) if mask.size > 0 else 0.0,
    }


def main() -> None:
    env = DrugDesignSingleAgentEnv(max_steps=30, max_atoms=30, invalid_penalty=0.1)

    obs, info = env.reset()
    print("RESET:", {k: v for k, v in info.items() if k != "action_mask"})

    total_reward = 0.0

    for t in range(200):
        # 1) Get the current joint mask (Discrete expects a 1D mask)
        mask = env.get_action_mask()

        # Safety checks to catch format mistakes early
        if not isinstance(mask, np.ndarray):
            mask = np.asarray(mask)

        if mask.ndim != 1:
            raise ValueError(f"Expected 1D Discrete mask, got shape {mask.shape}")

        # Gymnasium Discrete sampling expects an int8 mask (1 allowed, 0 disallowed)
        mask = mask.astype(np.int8)

        if mask.sum() == 0:
            raise RuntimeError(
                "Mask has zero valid actions. This means your env masked everything out. "
                "Usually happens if stop_min_atoms is too high or valence rules are too strict."
            )

        if t == 0:
            print("MASK SUMMARY:", _mask_summary(mask))

        # 2) Sample a valid action under the mask
        action = env.action_space.sample(mask=mask)  # action is an int for Discrete spaces

        # (Optional) decode for debug if your env exposes _decode (it does in your code)
        decoded = env._decode(int(action))  # DecodedAction(atom_type_id, attach_idx, bond_or_stop)

        # 3) Step
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += float(reward)

        if step_info.get("invalid"):
            print(
                f"[t={t}] INVALID: reason={step_info.get('reason')} r={reward} "
                f"action={action} decoded={decoded}"
            )
        else:
            print(
                f"[t={t}] OK: smiles={step_info.get('smiles')} atoms={step_info.get('num_atoms')} "
                f"r={reward} action={action} decoded={decoded}"
            )

        if terminated or truncated:
            print(
                "EPISODE DONE:",
                {
                    "terminated": terminated,
                    "truncated": truncated,
                    "final_smiles": step_info.get("smiles"),
                    "final_atoms": step_info.get("num_atoms"),
                    "episode_return": total_reward,
                },
            )
            obs, info = env.reset()
            print("RESET:", {k: v for k, v in info.items() if k != "action_mask"})
            total_reward = 0.0

    env.close()


if __name__ == "__main__":
    main()
