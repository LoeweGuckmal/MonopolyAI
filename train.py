import os

import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64
import ray
from gymnasium import spaces
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.framework import try_import_torch
from ray.tune.logger import UnifiedLogger

torch, nn = try_import_torch()

from monopolyEnv import MonopolyEnv

noSave = False

# Custom model that handles action masking
class MaskedActionsModel(FullyConnectedNetwork):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        #print(f"Policy observation space: {obs_space}")
        super(MaskedActionsModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation
        #print(f"Model input: {input_dict}")
        action_mask = input_dict["obs"]["action_mask"]

        # Get the model's output (action logits)
        logits, _ = super().forward(input_dict, state, seq_lens)

        # Apply the mask by setting logits for invalid actions to a large negative value
        inf_mask = torch.clamp(torch.log(action_mask), min=-1e10)
        masked_logits = logits + inf_mask

        return masked_logits, state

# Register the custom model
ModelCatalog.register_custom_model("masked_model", MaskedActionsModel)

def env_creator(env_config):
    return MonopolyEnv(config=env_config)

# Register the environment
tune.register_env("MonopolyEnv-v0", env_creator)

def main():
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Define the observation space for a single agent
    agent_obs_space = spaces.Dict({
        'action_mask': spaces.Box(low=0, high=1, shape=(123,), dtype=np.int8),
        'action': spaces.Discrete(5),
        'position': spaces.Discrete(40),
        'isPrison': spaces.Discrete(2),
        'money': spaces.Box(low=-300, high=12000, shape=(1,), dtype=np.int32),
        'owned_properties': spaces.Box(low=0, high=1, shape=(28,), dtype=np.int8),
        'rent': spaces.Box(low=0, high=2100, shape=(40,), dtype=np.float32),
        'houses': spaces.Box(low=0, high=6, shape=(28,), dtype=np.int8),
        'mortgageds': spaces.Box(low=0, high=1, shape=(28,), dtype=np.int8),
        'other_position': spaces.Box(low=0, high=40, shape=(3,), dtype=np.int8),
        'other_isPrison': spaces.Box(low=0, high=1, shape=(3,), dtype=np.int8),
        'other_money': spaces.Box(low=-300, high=12000, shape=(3,), dtype=np.int32),
        'other_owned_properties1': spaces.Box(low=0, high=1, shape=(28,), dtype=np.int8),
        'other_owned_properties2': spaces.Box(low=0, high=1, shape=(28,), dtype=np.int8),
        'other_owned_properties3': spaces.Box(low=0, high=1, shape=(28,), dtype=np.int8),
        'auction_state': spaces.Box(low=0, high=10000, shape=(4,), dtype=np.float32),
        'obogp': spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
        'trade_state': spaces.Box(low=0, high=29, shape=(2,), dtype=np.int8)
    })

    # Define Config
    config = (
        PPOConfig()
        .environment(env="MonopolyEnv-v0")
        .framework("torch")  # or "tf"
        .env_runners(num_env_runners=1, sample_timeout_s=1200, rollout_fragment_length=2000, num_envs_per_env_runner=1, batch_mode="complete_episodes")
        .training(train_batch_size=20000, lr=3.82e-5, gamma=0.99, lambda_=0.95, vf_clip_param=20.0, entropy_coeff=0.001, grad_clip=40.0, #lr=1e-4
                  lr_schedule=[[0, 3.82e-5],[1.2e6, 5e-5],[3.6e6, 1e-5],], entropy_coeff_schedule=[[0, 0.001],[3.6e5, 0.001],[1.2e6, 0.001]])
                  #lr_schedule=[[0, 1e-4],[1.2e6, 5e-5],[3.6e6, 1e-5],], entropy_coeff_schedule=[[0, 0.03],[3.6e5, 0.01],[1.2e6, 0.001]])
        .resources(num_gpus=1)
        .api_stack(enable_rl_module_and_learner=False,enable_env_runner_and_connector_v2=False)
        .multi_agent(
            policies={
                "default_policy": (
                    None,
                    agent_obs_space,
                    spaces.Discrete(123),
                    {
                        "model": {
                            "custom_model": "masked_model",
                            "fcnet_hiddens": [256, 256, 128],
                            "vf_share_layers": False,
                        }
                    },
                )
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "default_policy"
        )
    )

    # Function to create a logger
    def custom_logger_creator(config):
        logdir = os.path.expanduser("~/my_experiments/Monopoly_PPO")
        os.makedirs(logdir, exist_ok=True)
        return UnifiedLogger(config, logdir, loggers=None)

    if noSave:
        algo = config.build_algo()
    else:
        algo = config.build_algo(logger_creator=custom_logger_creator)

    # Check if checkpoint exists first
    checkpoint_path = "training/ppo_monopoly_selfplay"
    if os.path.exists(checkpoint_path):
        try:
            # Try with full path to directory
            absolute_path = os.path.abspath(checkpoint_path)
            print(f"Attempting to load checkpoint from: {absolute_path}")
            algo.load_checkpoint(absolute_path)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch.")
    else:
        print(f"Checkpoint '{checkpoint_path}' not found. Starting training from scratch.")


    import time
    import datetime
    start = round(time.time() * 1000)
    policy = algo.get_policy("default_policy")
    print("Policy model is:", type(policy.model))

    for i in range(1, 31):  # Increase training iterations for better learning, ~20 min/iter, 10 iter = 200k steps, 31=9.4h
        result = algo.train()
        print(f"({i}) Passed millis:", round(time.time() * 1000) - start)
        os.makedirs(checkpoint_path, exist_ok=True)
        algo.save_checkpoint(checkpoint_path)
        if i % 10 == 0 or i == 1:
            x = str(datetime.datetime.now()).replace(" ","_").replace(":","-").split(".")[0]
            checkpoint_dir = f"training/ppo_monopoly_selfplay_iter_{i}_{x}"
            #algo.save(checkpoint_dir)
            if not noSave:
                os.makedirs(checkpoint_dir, exist_ok=True)
                algo.save_checkpoint(checkpoint_dir)
    print("Passed millis:", round(time.time() * 1000) - start)


    # Save the trained policy
    print(f"Saving final checkpoint to: {checkpoint_path}")
    if not noSave:
        algo.save(checkpoint_path)

    # Shutdown Ray
    ray.shutdown()

if __name__ == "__main__":
    noSave = False
    main()