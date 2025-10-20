"""
Environment for Monopoly game that communicates with a separate game engine via HTTP.
This environment follows the Gymnasium interface.
"""

import socket
from typing import Dict, Tuple

import numpy as np
if not hasattr(np, "float_"):
    np.float_ = np.float64
import requests
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import time


class MonopolyEnv(MultiAgentEnv):
    """
    Monopoly environment that communicates with a game engine via HTTP.
    
    This environment implements the Gymnasium interface and sends actions to the
    game engine via HTTP POST requests and receives observations via HTTP GET requests.
    """


    metadata = {'render.modes': ['human']}
    
    def __init__(self, config=None):
        """
        Initialize the Monopoly environment.

        Args:
            config: Environment configuration dictionary
        """
        super(MonopolyEnv, self).__init__()    # Extract configuration or use defaults
        self.step_counter = 0
        if config is None:
            config = {}

        # Store the server URL and timeout
        server_url = config.get("server_url", "http://localhost:9990")
        self.timeout = config.get("timeout", 10.0)

        if server_url == "http://localhost:9990":
            # Use IP address instead of localhost for better compatibility
            IPAddr = socket.gethostbyname(socket.gethostname())
            self.server_url = "http://" + IPAddr + ":9990"
        else:
            self.server_url = server_url
        
        # Initialize game state
        self.game_id = None
        self.current_step = 0
        self.next_agent = 0  # Track which agent acts next (0-3)
        self._agent_ids = {"agent_1", "agent_2", "agent_3", "agent_4"}


        # Actions
        self.action_space = spaces.Discrete(123)
        
        # Observation space for self-play with 4 agents
        # Each agent has its own observation
        agent_obs_space = spaces.Dict({
            'action_mask': spaces.Box(low=0, high=1, shape=(123,), dtype=np.int8),
            'action': spaces.Discrete(5),
            'position': spaces.Discrete(40),
            'isPrison': spaces.Discrete(2),
            'money': spaces.Box(low=-2000, high=12000, shape=(1,), dtype=np.int32),
            'owned_properties': spaces.Box(low=0, high=1, shape=(28,), dtype=np.int8),
            'rent': spaces.Box(low=0, high=2100, shape=(40,), dtype=np.float32),
            'houses': spaces.Box(low=0, high=6, shape=(28,), dtype=np.int8),
            'mortgageds': spaces.Box(low=0, high=1, shape=(28,), dtype=np.int8),
            'other_position': spaces.Box(low=0, high=40, shape=(3,), dtype=np.int8),
            'other_isPrison': spaces.Box(low=0, high=1, shape=(3,), dtype=np.int8),
            'other_money': spaces.Box(low=-2000, high=12000, shape=(3,), dtype=np.int32),
            'other_owned_properties1': spaces.Box(low=0, high=1, shape=(28,), dtype=np.int8),
            'other_owned_properties2': spaces.Box(low=0, high=1, shape=(28,), dtype=np.int8),
            'other_owned_properties3': spaces.Box(low=0, high=1, shape=(28,), dtype=np.int8),
            'auction_state': spaces.Box(low=0, high=10000, shape=(4,), dtype=np.float32),
            'obogp': spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
            'trade_state': spaces.Box(low=0, high=29, shape=(2,), dtype=np.int8)
        })

        self.observation_space = agent_obs_space
        
        # Test connection to the server
        self._test_connection()
    
    def _test_connection(self):
        """Test the connection to the Monopoly game engine server."""
        try:
            print(f"Testing connection to {self.server_url}...")
            response = requests.get(f"{self.server_url}/ping", timeout=self.timeout)
            response.raise_for_status()
            print(f"Successfully connected to Monopoly game engine at {self.server_url}")
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not connect to Monopoly game engine: {e}")
            print("Make sure the game engine server is running")
    
    def reset(self, *, seed=None, options=None) -> Tuple[Dict, Dict]:
        """
        Reset the environment to start a new game.
        
        Args:
            seed: Random seed
            options: Additional options for resetting
            
        Returns:
            observation: The initial observation for all agents
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Send reset request to the server to start a new game
        try:
            response = requests.post(
                f"{self.server_url}/reset", 
                json={"seed": seed if seed is not None else np.random.randint(0, 1000000)},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            self.game_id = data.get("game_id")
            self.current_step = 0
            
            # Get the next agent to act (0-indexed)
            self.next_agent = int(data.get("next_agent", 1)) - 1  # Convert from 1-indexed to 0-indexed
            
            # Process observations for all agents
            observations = self._process_observation(data.get("observation", {}))
            info = data.get("info", {})
            if not info == {}:
                print("info: " + info)
            infos = {}

            return observations, infos
            
        except requests.exceptions.RequestException as e:
            print(f"Error during reset: {e}")
            # Return dummy observations if server is unreachable
            return self._get_dummy_observation(), {}
    
    def step(self, action_dict) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Take a step in the environment by sending an action to the game engine.
        
        Args:
            action_dict: Dictionary containing agent actions
            
        Returns:
            observation: The new observation for all agents after taking the action
            rewards: The rewards received by all agents
            terminated: Whether the episode is terminated
            truncated: Whether the episode is truncated
            info: Additional information
        """
        self.step_counter += 1
        if self.game_id is None:
            print("Warning: Game not initialized. Call reset() first.")
            return self._get_dummy_observation(), {"agent_1": 0.0, "agent_2": 0.0, "agent_3": 0.0, "agent_4": 0.0}, {"__all__": True}, {"__all__": False}, {}
        
        try:
            # Get the action for the current agent
            current_agent = f"agent_{self.next_agent + 1}"
            if current_agent not in action_dict:
                raise ValueError(f"Action for {current_agent} not provided in action_dict")
            action = action_dict[current_agent]


            # Send the action to the server for the current agent
            response = requests.post(
                f"{self.server_url}/step", 
                json={"game_id": self.game_id, "action": int(action), "agent": self.next_agent + 1},  # Convert to 1-indexed for server
                timeout=self.timeout
            )

            response.raise_for_status()
            data = response.json()

            # Update next agent to act
            self.next_agent = int(data.get("next_agent", 1)) - 1  # Convert from 1-indexed to 0-indexed

            # Process the response
            observations = self._process_observation(data.get("observation", {}))
            
            # Get rewards for all agents
            rewards = {
                f"agent_{i+1}": data.get("reward", {}).get(f"agent_{i+1}", 0.0)
                for i in range(4)
            }

            terminated = bool(data.get("terminated", False))
            truncated = bool(data.get("truncated", False))
            terminateds = {f"agent_{i+1}": terminated for i in range(4)}
            truncateds = {f"agent_{i+1}": truncated for i in range(4)}
            #Add __all__ keys as required by RLlib
            terminateds["__all__"] = terminated
            truncateds["__all__"] = truncated

            info = data.get("info", {})
            if not info == {}:
                print("info: " + info)
            infos = {}

            self.current_step += 1
            #print("return: ", observations, rewards, terminateds, truncateds, info)

            #print("step obs: ", observations, "-------------")

            #print("step rewards: ", rewards)
            
            return observations, rewards, terminateds, truncateds, infos
            
        except requests.exceptions.RequestException as e:
            print(f"Error during step: {e}")
            # Return a dummy result if server is unreachable
            dummy_dict = {f"agent_{i+1}": 0.0 for i in range(4)}
            return self._get_dummy_observation(), dummy_dict, {"__all__": True}, {"__all__": False}, {}

    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode: The rendering mode
            
        Returns:
            None or rendered frame depending on mode
        """
        if mode == 'human' and self.game_id is not None:
            try:
                # Request a visualization from the server
                response = requests.get(
                    f"{self.server_url}/render", 
                    params={"game_id": self.game_id},
                    timeout=self.timeout
                )
                response.raise_for_status()

                print("Game rendered on server")
                
            except requests.exceptions.RequestException as e:
                print(f"Error during render: {e}")
    
    def close(self):
        """Clean up resources and close the environment."""
        if self.game_id is not None:
            try:
                # Notify the server that we're done with this game
                requests.post(
                    f"{self.server_url}/close", 
                    json={"game_id": self.game_id},
                    timeout=self.timeout
                )
                self.game_id = None
            except requests.exceptions.RequestException as e:
                print(f"Error during close: {e}")
    
    def _process_observation(self, raw_observation: Dict) -> Dict:
        """
        Process the raw observation from the server into the format expected by the agent.
        
        Args:
            raw_observation: The raw observation from the server containing data for all agents
            
        Returns:
            The processed observation for all agents
        """
        processed_obs = {}
        
        # Process observations for each agent
        for agent_id in range(1, 5):  # Agents 1-4
            agent_key = f"agent_{agent_id}"
            if agent_id-1 == self.next_agent:
            
                # Get the agent's observation from the raw observation
                agent_obs = raw_observation.get(agent_key, {})
                #print("raw_obs: ", raw_observation, "--------------")
                #print("agent_obs: ", agent_obs, "--------------")

                #print(agent_obs.get("action_mask", [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
                # Process the agent's observation
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
        
        return processed_obs

    def _get_dummy_observation(self) -> Dict:
        """
        Generate a dummy observation for cases when the server is unreachable.

        Returns:
            A dummy observation for all agents that matches the observation space
        """
        # Create a dummy observation for a single agent
        agent_dummy_obs = {
            'action': 0,
            'position': 0,
            'isPrison': 0,
            'money': np.array([0], dtype=np.int32),
            'owned_properties': np.array([0] * 28, dtype=np.int8),
            'rent': np.array([0.0] * 40, dtype=np.float32),
            'houses': np.array([0] * 28, dtype=np.int8),
            'mortgageds': np.array([0] * 28, dtype=np.int8),
            'other_position': np.array([0] * 3, dtype=np.int8),
            'other_isPrison': np.array([0] * 3, dtype=np.int8),
            'other_money': np.array([0] * 3, dtype=np.int32),
            'other_owned_properties1': np.array([0] * 28, dtype=np.int8),
            'other_owned_properties2': np.array([0] * 28, dtype=np.int8),
            'other_owned_properties3': np.array([0] * 28, dtype=np.int8),
            'auction_state': np.array([0] * 4, dtype=np.float32),
            'obogp': np.array([0] * 10, dtype=np.float32),
            'trade_state': np.array([0] * 2, dtype=np.int8),
            'action_mask': np.array([1]*2+[0]*121, dtype=np.int8)
        }
        #raise RuntimeError("Server unreachable, cannot generate valid observation")

        # Return dummy observations for all agents
        return {
            'agent_1': agent_dummy_obs.copy(),
            'agent_2': agent_dummy_obs.copy(),
            'agent_3': agent_dummy_obs.copy(),
            'agent_4': agent_dummy_obs.copy()
            #'next_agent': 0  # Default to agent 1 (0-indexed)
        }

    def get_agent_id_for_next_step(self) -> str:
        """Override this method from MultiAgentEnv to implement turn-based logic.
        Returns the ID of the agent that should act next.
        """
        return f"agent_{self.next_agent + 1}"
    def get_agent_ids(self):
        """Return the list of agent ids in the environment"""
        return [f"agent_{i+1}" for i in range(4)]


# Helper function to create the environment
def make_monopoly_env(server_url: str = "http://localhost:9990") -> MonopolyEnv:
    """
    Create a Monopoly environment instance.
    
    Args:
        server_url: The URL of the Monopoly game engine server
        
    Returns:
        A MonopolyEnv instance
    """
    return MonopolyEnv()
