import gymnasium as gym
from snakeGameCheese import snakeGameCheese
from typing import Optional
import numpy as np
import pygame
import math

# NOTE FOR FUTURE: gymnasium prefers float32?

class snakeRLEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, length_of_grid_x = 30, length_of_grid_y = 24, render_mode = None):
        self.length_of_grid_x = length_of_grid_x
        self.length_of_grid_y = length_of_grid_y

        self.metadata = {"render_modes": []}
        self.render_mode = render_mode
        self.window = None
        self.clock = None
    
        self.game = snakeGameCheese(
            grid_size=np.array([self.length_of_grid_x, self.length_of_grid_y]),
            debug=False,
            draw=False
        )
        self.prev_direction = self.game.direction
    
        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32) # the apple
    
        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self.length_of_grid_x - 1, self.length_of_grid_y - 1]),
                    dtype=np.int32
                ),
                "target": gym.spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self.length_of_grid_x - 1, self.length_of_grid_y - 1]),
                    dtype=np.int32
                ),
                "danger": gym.spaces.Box(
                    low = 0,
                    high = 1,
                    shape=(4,),
                    dtype=np.int32
                ),
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
        dangers = self._get_dangers()
        danger_arr = np.array([
            dangers["UP"],
            dangers["DOWN"],
            dangers["LEFT"],
            dangers["RIGHT"]
        ], dtype=np.int32)
        
        return {"agent": self._agent_location, "target": self._target_location, "danger": danger_arr}

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
    def _get_dangers(self):
        # should detect if next move results in collision with itself or a wall, returns dictionary of where dangers are based on direction
        head_x, head_y = self.game.snake_position
        step = self.game.cell_size

        potential_positions = {"UP": (head_x, head_y - step), 
                               "DOWN": (head_x, head_y + step), 
                               "LEFT": (head_x - step, head_y), 
                               "RIGHT": (head_x + step, head_y),}
        dangers = {}

        for direction, (nx, ny) in potential_positions.items():
            danger = 0
            # boundary/wall check
            if nx < 0 or nx >= self.game.size_x or ny < 0 or ny >= self.game.size_y:
                danger = 1
            # body collision check
            for segment in self.game.snake_body[1:]:
                if segment[2] == self.game.active_body_key:
                    if nx == segment[0] and ny == segment[1]:
                        danger = 1
                        break
                        
            dangers[direction] = danger

        return dangers
        
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

        while True:
            try:
                # Randomly place the agent anywhere on grid
                self._agent_location = np.array([
                    self.np_random.integers(0, self.length_of_grid_x),
                    self.np_random.integers(0, self.length_of_grid_y),
                ], dtype=np.int32)
                # Randomly place target in a different location than agent
                while True:
                    self._target_location = np.array([
                        self.np_random.integers(0, self.length_of_grid_x),
                        self.np_random.integers(0, self.length_of_grid_y),
                    ], dtype=np.int32)
                    if not np.array_equal(self._target_location, self._agent_location):
                        break
                self.game = snakeGameCheese(
                    grid_size=np.array([self.length_of_grid_x, self.length_of_grid_y]),
                    fruit_position=self._target_location,
                    snake_position=self._agent_location,
                    debug=False,
                    draw=False
                )
                break  # success, exit loop
        
            except ValueError:
                continue  # retry random positions

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
        prev_score = self.game.score
        prev_distance = 0
        try:
            x_calc = float(self.game.fruit_position[0]) - float(self.game.snake_position[0])
            y_calc = float(self.game.fruit_position[1]) - float(self.game.snake_position[1])
            prev_distance = math.sqrt((x_calc) ** 2 + (y_calc) ** 2) # Euclidean Distance
        except ValueError:
            prev_distance = 0
        
        self.game.step_function(direction)

        # Convert pixel positions back to grid positions
        self._agent_location = self.game.snake_position // self.game.cell_size
        self._target_location = self.game.fruit_position // self.game.cell_size

        # Check if agent reached the target or died
        terminated = self.game.wall_dead or self.game.body_dead # Game terminates if either snake runs into itself or a wall

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        current_distance = 0
        try:
            x_calc = float(self.game.fruit_position[0]) - float(self.game.snake_position[0])
            y_calc = float(self.game.fruit_position[1]) - float(self.game.snake_position[1])
            current_distance = math.sqrt((x_calc) ** 2 + (y_calc) ** 2) # Euclidean Distance
        except ValueError:
            current_distance = 0

        reward = 0

        # Simple reward structure: +1 for reaching target, 0 otherwise
        # Alternative: could give small negative rewards for each step to encourage efficiency (NOTE FOR FUTURE: negative reward if dies?)
        if self.game.score > prev_score:
            reward += 5
        elif terminated: 
            reward -= 5
        else:
            reward += 0.001 # for staying alive
            distance_diff = prev_distance - current_distance
            reward = reward + ((distance_diff / max(self.length_of_grid_x, self.length_of_grid_y))) / 2
            if direction != self.prev_direction:
                reward = reward - 0.01
        self.prev_direction = direction

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def score(self):
        return self.game.score

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None:
            self.game.fps = pygame.time.Clock()
            pygame.init()
            pygame.display.set_caption('Snake Game with Cheese Variation')
            self.window = pygame.display.set_mode((self.game.size_x, self.game.size_y))

        self.window.fill('green')
        # draw head regardless of its assigned visibility
        head = self.game.snake_body[0]
        pygame.draw.rect(self.window, 'cyan', pygame.Rect(head[0], head[1], self.game.cell_size, self.game.cell_size))
        # draw body parts alternating by body key
        for pos in self.game.snake_body[1:]:
            if pos[2] == self.game.active_body_key: # actual body, not skipped part
                pygame.draw.rect(self.window, 'blue',
                                pygame.Rect(pos[0], pos[1], self.game.cell_size, self.game.cell_size))
        
        pygame.draw.rect(self.window, 'red', pygame.Rect(
            self.game.fruit_position[0], self.game.fruit_position[1], self.game.cell_size, self.game.cell_size))

        # displaying score continuously
        color = 'white'
        font = 'times new roman'
        size = 30
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render('Score : ' + str(self.game.score), True, color)
        score_rect = score_surface.get_rect()
        self.window.blit(score_surface, score_rect)
    
        # Refresh game screen
        pygame.display.update()
        

        return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))

    

