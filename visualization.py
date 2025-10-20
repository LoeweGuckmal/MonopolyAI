import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from gym import spaces
from gym.spaces import Box
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork

# Try importing torchviz but don't fail if it's not available
try:
    from torchviz import make_dot
    has_torchviz = True
except ImportError:
    has_torchviz = False

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
action_space = spaces.Discrete(123)

def flatten_obs_space(obs_space: spaces.Dict):
    total_dim = 0
    shapes = {}
    for key, space in obs_space.spaces.items():
        shape = space.shape if isinstance(space, Box) else (1,)
        shapes[key] = shape
        dim = int(np.prod(shape))
        total_dim += dim
        print(f"Space {key}: {shape}, flattened: {dim}")
    flat_space = Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)
    print(f"Total flattened dimension: {total_dim}")
    return flat_space, shapes

def flatten_obs(obs_dict, obs_space):
    flat_list = []
    for key, space in obs_space.spaces.items():
        val = obs_dict[key]
        if isinstance(space, spaces.Discrete):
            one_hot = np.zeros(space.n, dtype=np.float32)
            one_hot[int(val)] = 1.0
            flat_list.append(one_hot)
        elif isinstance(space, spaces.Box):
            flat_list.append(val.flatten().astype(np.float32))
        else:
            raise NotImplementedError(f"Space {key} of type {type(space)} not supported")
    return torch.tensor(np.concatenate(flat_list)).unsqueeze(0)

def correct_flatten_obs_space(obs_space: spaces.Dict):
    total_dim = 0
    shapes = {}
    for key, space in obs_space.spaces.items():
        if isinstance(space, spaces.Discrete):
            # For Discrete spaces, the flattened size is the number of possible values (one-hot encoding)
            shapes[key] = (space.n,)
            total_dim += space.n
        else:
            shape = space.shape
            shapes[key] = shape
            total_dim += int(np.prod(shape))
    flat_space = Box(low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32)
    print(f"Corrected total flattened dimension: {total_dim}")
    return flat_space, shapes

flat_obs_space, obs_shapes = correct_flatten_obs_space(agent_obs_space)

# Create a modified MaskedActionsModel class that works with flattened observations
class ModifiedMaskedActionsModel(FullyConnectedNetwork):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(ModifiedMaskedActionsModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        # Store the offset and size of the action mask in the flattened observation
        # This will help us extract it later
        self.mask_size = 123  # Size of action mask
        
        # Calculate the offset of action_mask in the flattened observation
        offset = 0
        for key, space in agent_obs_space.spaces.items():
            if key == 'action_mask':
                self.mask_offset = offset
                break
            if isinstance(space, spaces.Discrete):
                offset += space.n
            else:
                offset += int(np.prod(space.shape))
    
    def forward(self, input_dict, state, seq_lens):
        # Extract the flattened observation
        flat_obs = input_dict["obs"]
        
        # Extract the action mask from the flattened observation
        action_mask = flat_obs[:, :self.mask_size]  # Assuming action_mask is at the beginning
        
        # Pass the observation through the network
        logits, _ = super().forward(input_dict, state, seq_lens)
        
        # Apply the mask
        inf_mask = torch.clamp(torch.log(action_mask), min=-1e10)
        masked_logits = logits + inf_mask
        
        return masked_logits, state

# Load policy state
with open("training/ppo_monopoly_selfplay_iter_30_2025-10-17_12-31-47/policies/default_policy/policy_state.pkl", "rb") as f:
    policy_state = pickle.load(f)

print("Loaded policy state keys:", policy_state.keys())
state_dict = policy_state.get("state_dict", policy_state).get("weights", policy_state)
print("State dict keys:", state_dict.keys())

# Print shapes of the parameters to understand the expected dimensions
for k, v in state_dict.items():
    if isinstance(v, np.ndarray):
        state_dict[k] = torch.tensor(v, dtype=torch.float32)
    print(f"Parameter: {k}, Shape: {state_dict[k].shape}")

model_config = {"fcnet_hiddens": [256, 256, 128], "vf_share_layers": False}
model = ModifiedMaskedActionsModel(flat_obs_space, action_space, action_space.n, model_config, name="ppo_model")

print("\nModel parameters before loading:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

model.load_state_dict(state_dict)
model.eval()

print("Creating dummy input...")
# Create dummy observation dictionary correctly
dummy_obs_dict = {}
for key, space in agent_obs_space.spaces.items():
    if isinstance(space, spaces.Discrete):
        dummy_obs_dict[key] = 0  # Any valid value for the Discrete space
    elif isinstance(space, spaces.Box):
        dummy_obs_dict[key] = np.ones(space.shape, dtype=space.dtype)

# Flatten the observation with one-hot encoding for Discrete spaces
dummy_input = {"obs": flatten_obs(dummy_obs_dict, agent_obs_space)}
dummy_state = []
dummy_seq_lens = torch.tensor([1])

print("Running model inference...")
logits, _ = model(dummy_input, dummy_state, dummy_seq_lens)

# Try to generate the network diagram, but don't crash if Graphviz is missing
if has_torchviz:
    try:
        print("Generating network diagram...")
        make_dot(logits, params=dict(model.named_parameters())).render("ppo_model", format="png")
        print("Network diagram saved as 'ppo_model.png'")
    except Exception as e:
        print(f"Couldn't generate visualization diagram: {e}")
        print("To generate network diagrams, please install Graphviz from https://graphviz.org/download/")
else:
    print("torchviz not available. Install it with 'pip install torchviz' to generate network diagrams")

# Visualization of model weights - this part works without Graphviz
print("Generating weight visualizations...")
for name, param in model.named_parameters():
    if "weight" in name:
        plt.figure(figsize=(10, 8))
        plt.imshow(param.detach().cpu().numpy(), aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title(name)
        safe_name = name.replace('.', '_').replace('/', '_')
        plt.savefig(f"{safe_name}.png")
        print(f"Saved weight visualization: {safe_name}.png")
        plt.close()  # Close the plot to avoid displaying it in interactive mode

print("Model weight visualization complete.")