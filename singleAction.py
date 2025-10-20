import os
import traceback

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64
import ray
from gymnasium import spaces
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

from monopolyEnv import MonopolyEnv

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

ray.init(ignore_reinit_error=True, include_dashboard=False)

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

config = (
    PPOConfig()
    .environment(env="MonopolyEnv-v0")
    .framework("torch")  # or "tf"
    .env_runners(num_env_runners=1, sample_timeout_s=1200, rollout_fragment_length=2000, num_envs_per_env_runner=1, batch_mode="complete_episodes")
    .training(train_batch_size=20000, lr=1e-4, gamma=0.99, lambda_=0.95, vf_clip_param=20.0, entropy_coeff=0.01, grad_clip=40.0,
              lr_schedule=[[0, 1e-4],[1.2e6, 5e-5],[3.6e6, 1e-5],], entropy_coeff_schedule=[[0, 0.03],[3.6e5, 0.01],[1.2e6, 0.001]])
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
algo = config.build()

checkpoint_path = "training/ppo_monopoly_selfplay_iter_30_2025-10-17_12-31-47"
if os.path.exists(checkpoint_path):
    try:
        absolute_path = os.path.abspath(checkpoint_path)
        print(f"Attempting to load checkpoint from: {absolute_path}")
        algo.load_checkpoint(absolute_path)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Starting training from scratch.")
else:
    print(f"Checkpoint '{checkpoint_path}' not found. Starting training from scratch.")


class RLlibHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data.decode("utf-8"))
            # Expected input: { "agent_id": obs_dict }
            # obs_dict should be the observation for that agent
            processed_obs = {}
            action = 0

            # Process observations for each agent
            for agent_id in range(1, 5):  # Agents 1-4
                agent_key = f"agent_{agent_id}"
                # Get the agent's observation from the raw observation
                agent_obs = data.get(agent_key, {})
                if agent_obs == {}:
                    continue
                processed_obs[agent_key] = { #auction mask isn't bad with all buy/sell house -> writing
                    'action_mask': np.array(agent_obs.get("action_mask", [0, 0, 0, 0, 0, 1, 1]+[0]*116), dtype=np.int8),
                    'action': agent_obs.get("action", 0),
                    'position': agent_obs.get("position", 0),
                    'isPrison': agent_obs.get("isPrison", 0),
                    'money': np.array([agent_obs.get("money", [0])[0] if isinstance(agent_obs.get("money", [0]), list) else agent_obs.get("money", 0)], dtype=np.int32),
                    'owned_properties': np.array(agent_obs.get("owned_properties", [0] * 28), dtype=np.int8),
                    'rent': np.array(agent_obs.get("rent", [0.0] * 40), dtype=np.float32),
                    'houses': np.array(agent_obs.get("houses", [0] * 28), dtype=np.int8),
                    'mortgageds': np.array(agent_obs.get("mortgageds", [0] * 28), dtype=np.int8),
                    'other_position': np.array(agent_obs.get("other_position", [0] * 3), dtype=np.int8),
                    'other_isPrison': np.array(agent_obs.get("other_isPrison", [0] * 3), dtype=np.int8),
                    'other_money': np.array(agent_obs.get("other_money", [0] * 3), dtype=np.int32),
                    'other_owned_properties1': np.array(agent_obs.get("other_owned_properties1", [0] * 28), dtype=np.int8),
                    'other_owned_properties2': np.array(agent_obs.get("other_owned_properties2", [0] * 28), dtype=np.int8),
                    'other_owned_properties3': np.array(agent_obs.get("other_owned_properties3", [0] * 28), dtype=np.int8),
                    'auction_state': np.array(agent_obs.get("auction_state", [0] * 4), dtype=np.float32),
                    'obogp': np.array(agent_obs.get("obogp", [0] * 10), dtype=np.float32),
                    'trade_state': np.array(agent_obs.get("trade_state", [0] * 2), dtype=np.int8)
                }
                #print(processed_obs[agent_key])
                action = algo.compute_single_action(processed_obs[agent_key], policy_id="default_policy", explore=False)
                action_mask = agent_obs.get("action_mask", [0, 0, 0, 0, 0, 1, 1]+[0]*116)
                #print(action_mask, "->", action)

            response = json.dumps(str(action)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response)

        except Exception as e:
            traceback.print_exc()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            error_msg = {"error": str(e)}
            self.wfile.write(json.dumps(error_msg).encode("utf-8"))


def run(server_class=HTTPServer, handler_class=RLlibHandler, port=8080):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting RLlib PPO HTTP server on port {port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    run()
    ray.shutdown() #33’806’763