import gymnasium as gym
from snakeGameCheese import snakeGame
from typing import Optional
import numpy as np

class snakeRLEnvironment(gym.Env):
    def __init__(self, length_of_grid_x = 30, length_of_grid_y = 24):
            self.game = snakeGame()
            self.length_of_grid_x = length_of_grid_x
            self.length_of_grid_y = length_of_grid_y
    
            # Initialize positions - will be set randomly in reset()
            # Using -1,-1 as "uninitialized" state
            self._agent_location = np.array([-1, -1], dtype=np.int32)
            self._target_location = np.array([-1, -1], dtype=np.int32) # the apple
    
            # Define what the agent can observe
            # Dict space gives us structured, human-readable observations
            self.observation_space = gym.spaces.Dict(
                {
                    "agent": gym.spaces.Box(0, self.length_of_grid_x - 1, shape=(2,), dtype=int),   # [x, y] coordinates
                    "target": gym.spaces.Box(0, self.length_of_grid_x - 1, shape=(2,), dtype=int),  # [x, y] coordinates
                }
            )
    
            self.action_space = gym.spaces.Discrete(4)
    
            # Map action numbers to actual movements on the grid
            # This makes the code more readable than using raw numbers
            self._action_to_direction = {
                0: "UP",   # move up
                1: "DOWN",  # move down
                2: "LEFT",  # move left
                3: "RIGHT" # move right
            }

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            dict: Observation with agent and target positions
        """
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            dict: Info with distance between agent and target
        """
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        agent_location_x = self.np_random.integers(0, self.length_of_grid_x, size=1, dtype=int)
        agent_location_y = self.np_random.integers(0, self.length_of_grid_y, size=1, dtype=int)
        self._agent_location = np.array([agent_location_x, agent_location_y])

        # Randomly place target, ensuring it's different from agent position
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            target_location_x = self.np_random.integers(0, self.length_of_grid_x, size=1, dtype=int)
            target_location_y = self.np_random.integers(0, self.length_of_grid_y, size=1, dtype=int)
            self._target_location = np.array([target_location_x, target_location_y])

        self.game = snakeGame(self._target_location, self._agent_location)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-2 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-2) to a movement direction
        direction = self._action_to_direction[action]

        self.game.step_function(direction)
        self._agent_location = self.game.snake_position
        self._target_location = self.game.fruit_position

        # Check if agent reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency
        reward = 1 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info