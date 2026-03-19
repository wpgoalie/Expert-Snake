import gymnasium as gym
from snakeGameCheese import snakeGameCheese
from fruit import EnemyFruit, DecayFruit
from typing import Optional
import numpy as np
import pygame
import math

class snakeRLEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, length_of_grid_x = 30, length_of_grid_y = 24, render_mode = None):
        self.length_of_grid_x = length_of_grid_x
        self.length_of_grid_y = length_of_grid_y

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.steps_survived = 0
    
        self.game = snakeGameCheese(
            grid_size=np.array([self.length_of_grid_x, self.length_of_grid_y]),
            debug=False,
            draw=False
        )
        self._prev_direction = self.game.direction
    
        # Initialize positions - will be set randomly in reset()
        # Using -1,-1 as "uninitialized" state
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._num_fruits = len(self.game.fruits)
    
        # Define what the agent can observe
        # Dict space gives us structured, human-readable observations
        obs_size = 2 + self._num_fruits * 6 + 4
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        self.grid_max = np.array(
            [self.length_of_grid_x - 1, self.length_of_grid_y - 1],
            dtype=np.float32
        )
        
        self.action_space = gym.spaces.Discrete(4)
        
        # Map action numbers to actual movements on the grid
        self._action_to_direction = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT"
        }

    def _get_obs(self):
        dangers = self._get_dangers()
        danger_arr = np.array([
            dangers["UP"],
            dangers["DOWN"],
            dangers["LEFT"],
            dangers["RIGHT"]
        ], dtype=np.float32)
    
        agent_norm = self._agent_location / self.grid_max
    
        fruit_features = []
    
        for fruit in self.game.fruits:
    
            # normalized position
            pos_norm = fruit.position / self.grid_max
    
            # relative direction
            delta = (fruit.position - self._agent_location) / self.grid_max
    
            # euclidean distance
            dist = np.linalg.norm(fruit.position - self._agent_location)
            dist_norm = dist / np.linalg.norm(self.grid_max)
    
            value_norm = fruit.value / fruit.max_value
    
            fruit_features.extend([
                pos_norm[0],
                pos_norm[1],
                value_norm,
                delta[0],
                delta[1],
                dist_norm
            ])

        obs = np.concatenate([
            agent_norm,
            fruit_features,
            danger_arr
        ])
    
        return obs.astype(np.float32)

    def _get_info(self):
        info = {}
        snake_grid = self._agent_location
        for i, fruit in enumerate(self.game.fruits):
            fruit_grid = fruit.position
            dist = snake_grid - fruit_grid
            info[f"distance_fruit_{i}"] = dist
        return info
        
    def _get_dangers(self):
        # should detect if next move results in collision with itself or a wall, returns dictionary of where dangers are based on direction
        head_x, head_y = self.game.snake_position

        potential_positions = {"UP": (head_x, head_y - 1), 
                               "DOWN": (head_x, head_y + 1), 
                               "LEFT": (head_x - 1, head_y), 
                               "RIGHT": (head_x + 1, head_y),}
        dangers = {}

        for direction, (nx, ny) in potential_positions.items():
            danger = 0
            # boundary/wall check
            if nx < 0 or nx >= self.length_of_grid_x  or ny < 0 or ny >= self.length_of_grid_y:
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
                # game already takes care of this part
                self.game = snakeGameCheese(
                    grid_size=np.array([self.length_of_grid_x, self.length_of_grid_y]),
                    snake_position=self._agent_location,
                    debug=False,
                    draw=False
                )
                break  # success, exit loop
        
            except ValueError:
                continue  # retry random positions

        observation = self._get_obs()
        info = self._get_info()
        self._prev_direction = self.game.direction

        return observation, info

    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-2 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action (0-2) to a movement direction
        action = int(action)
        direction = self._action_to_direction[action]
        prev_score = self.game.score

        prev_distances = [np.linalg.norm(self._agent_location - fruit.position) for fruit in self.game.fruits]
        
        self.game.step_function(direction)

        # Convert pixel positions back to grid positions
        self._agent_location = self.game.snake_position
        self._target_locations = [fruit.position for fruit in self.game.fruits]

        # Check if agent reached the target or died
        terminated = self.game.wall_dead or self.game.body_dead or self.game.fruit_dead # Game terminates if either snake runs into itself or a wall

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        current_distances = [np.linalg.norm(self._agent_location - fruit.position) for fruit in self.game.fruits]

        # reward = 0

        # # score change (handles +fruit and -fruit automatically)
        # score_delta = self.game.score - prev_score
        # reward += 20 * score_delta
        
        # # death penalty
        # if terminated:
        #     reward -= 25
        # else:
        #     # small step penalty (prevents infinite loops)
        #     reward -= 0.01
        
        #     # distance shaping toward closest fruit
        #     closest_idx = np.argmin(prev_distances)
        
        #     prev_d = prev_distances[closest_idx]
        #     curr_d = current_distances[closest_idx]
        
        #     distance_diff = prev_d - curr_d
        
        #     # normalize distance
        #     max_grid = np.sqrt(self.length_of_grid_x**2 + self.length_of_grid_y**2)
        #     shaped = distance_diff / max_grid
        
        #     fruit = self.game.fruits[closest_idx]
        
        #     if isinstance(fruit, EnemyFruit):
        #         reward -= 0.15 * shaped
        #     else:
        #         reward += 0.15 * shaped

        reward = 0

        score_diff = self.game.score - prev_score
        reward += 30 * score_diff
        
        if terminated:
            reward -= 25
        
        reward -= 0.05


        self._prev_direction = direction
        
        observation = self._get_obs()
        info = self._get_info()
                
        self._prev_direction = direction

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def score(self):
        return self.game.score

    def fruit_stats(self):
        return self.game.fruit_cnts.copy()

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

        if len(self.game.snake_body) == 0:
            # do not render if snake has no body left
            pygame.display.update()
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))
            
        # draw head
        head = self.game.snake_body[0]
        head_pixel = self.game.grid_to_pixel(head[:2], self.game.cell_size)
        pygame.draw.rect(self.window, 'cyan', pygame.Rect(head_pixel[0], head_pixel[1], self.game.cell_size, self.game.cell_size))
        
        # draw body
        for pos in self.game.snake_body[1:]:
            if pos[2] == self.game.active_body_key:
                pixel_pos = self.game.grid_to_pixel(pos[:2], self.game.cell_size)
                pygame.draw.rect(self.window, 'blue', pygame.Rect(pixel_pos[0], pixel_pos[1], self.game.cell_size, self.game.cell_size))
        
        # draw all fruits
        for fruit in self.game.fruits:
            fruit_pixel = self.game.grid_to_pixel(fruit.position, self.game.cell_size)
            pygame.draw.rect(self.window, fruit.color, pygame.Rect(fruit_pixel[0], fruit_pixel[1], self.game.cell_size, self.game.cell_size))

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

    

