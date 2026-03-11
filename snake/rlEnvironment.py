import gymnasium as gym
from snakeGameCheese import snakeGameCheese
from typing import Optional
import numpy as np
import pygame
import math
from fruit import Fruit, EnemyFruit, DecayFruit

class snakeRLEnvironment(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, length_of_grid_x = 30, length_of_grid_y = 24, render_mode = None):
        self.length_of_grid_x = length_of_grid_x
        self.length_of_grid_y = length_of_grid_y

        self.metadata = {"render_modes": []}
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
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([1, 1]),
                    # high=np.array([self.length_of_grid_x - 1, self.length_of_grid_y - 1]),
                    dtype=np.float32
                ),
                "fruits": gym.spaces.Box(
                    low=0, high=1, shape=(self._num_fruits, 3), dtype=np.float32
                    # each fruit: [x, y, value]
                    # fruit[0] = normal, fruit[1] = decay/cheese
                ),
                "danger": gym.spaces.Box(
                    low = 0,
                    high = 1,
                    shape=(4,),
                    dtype=np.float32
                ),
                "enemy_danger":gym.spaces.Box(
                    low = 0,
                    high = 1,
                    shape=(4,),
                    dtype=np.float32
                ),
            }
        )
        # for normalization in _get_obs(self)
        self.grid_max = np.array([self.length_of_grid_x - 1, self.length_of_grid_y - 1], dtype=np.float32)
    
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
        all_dangers = self._get_dangers()
        dangers = all_dangers[0]
        danger_arr = np.array([
            dangers["UP"],
            dangers["DOWN"],
            dangers["LEFT"],
            dangers["RIGHT"]
        ], dtype=np.float32)
        enemy_dangers = all_dangers[1]
        enemy_danger_arr = np.array([
            enemy_dangers["UP"],
            enemy_dangers["DOWN"],
            enemy_dangers["LEFT"],
            enemy_dangers["RIGHT"]
        ], dtype=np.float32)

        # initialize structure [x,y,value] for each fruit
        fruits_arr = np.zeros((self._num_fruits, 3), dtype=np.float32)
        # fill in fruit list
        for i, fruit in enumerate(self.game.fruits):
            # normalize position
            pos = fruit.position / self.grid_max
            x_norm, y_norm = pos
            # normalize value
            value_norm = fruit.value / fruit.max_value
            fruits_arr[i] = [x_norm, y_norm, value_norm]

        agent_norm = self._agent_location / self.grid_max
        
        return {"agent": agent_norm, "fruits": fruits_arr, "danger": danger_arr, "enemy_danger": enemy_danger_arr}

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
        enemy_x, enemy_y = self.game.fruits[2].position[0], self.game.fruits[2].position[1]

        potential_positions = {"UP": (head_x, head_y - 1), 
                               "DOWN": (head_x, head_y + 1), 
                               "LEFT": (head_x - 1, head_y), 
                               "RIGHT": (head_x + 1, head_y),}

        potential_positions_enemy = {3: (enemy_x, enemy_y - 1), 
                               1: (enemy_x, enemy_y + 1), 
                               2: (enemy_x - 1, enemy_y), 
                               0: (enemy_x + 1, enemy_y),}
        
        dangers = {}
        enemy_dangers = {}

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

        danger = 0
        current_enemy_direction = self.game.fruits[2].cur_side
        
        # enemy fruit check
        for index, segment in enumerate(self.game.snake_body):
            if segment[2] == self.game.active_body_key:
                if  potential_positions_enemy[current_enemy_direction][0] == segment[0] and potential_positions_enemy[current_enemy_direction][1] == segment[1]:
                    danger = 1
                    break

        if current_enemy_direction == 0: # moving right
            enemy_dangers["RIGHT"] = danger
            enemy_dangers["UP"] = 0
            enemy_dangers["DOWN"] = 0
            enemy_dangers["LEFT"] = 0
        elif current_enemy_direction == 1: # moving down
            enemy_dangers["RIGHT"] = 0
            enemy_dangers["UP"] = 0
            enemy_dangers["DOWN"] = danger
            enemy_dangers["LEFT"] = 0
        elif current_enemy_direction == 2: # moving left
            enemy_dangers["RIGHT"] = 0
            enemy_dangers["UP"] = 0
            enemy_dangers["DOWN"] = 0
            enemy_dangers["LEFT"] = danger
        else:
            enemy_dangers["RIGHT"] = 0
            enemy_dangers["UP"] = danger
            enemy_dangers["DOWN"] = 0
            enemy_dangers["LEFT"] = 0

        return dangers, enemy_dangers
        
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
                # # Randomly place target in a different location than agent
                # while True:
                #     self._target_location = np.array([
                #         self.np_random.integers(0, self.length_of_grid_x),
                #         self.np_random.integers(0, self.length_of_grid_y),
                #     ], dtype=np.int32)
                #     if not np.array_equal(self._target_location, self._agent_location):
                #         break
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
        direction = self._action_to_direction[action]
        prev_score = self.game.score
        # prev_distance = 0
        # try:
        #     x_calc = float(self.game.fruit_position[0]) - float(self.game.snake_position[0])
        #     y_calc = float(self.game.fruit_position[1]) - float(self.game.snake_position[1])
        #     prev_distance = math.sqrt((x_calc) ** 2 + (y_calc) ** 2) # Euclidean Distance
        # except ValueError:
        #     prev_distance = 0

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

        # current_distance = 0
        # try:
        #     x_calc = float(self.game.fruit_position[0]) - float(self.game.snake_position[0])
        #     y_calc = float(self.game.fruit_position[1]) - float(self.game.snake_position[1])
        #     current_distance = math.sqrt((x_calc) ** 2 + (y_calc) ** 2) # Euclidean Distance
        # except ValueError:
        #     current_distance = 0
        current_distances = [np.linalg.norm(self._agent_location - fruit.position) for fruit in self.game.fruits]

        reward = 0

        if self.game.score > prev_score:
            # strong fruit reward that grows slightly with score
            reward += 20 * (self.game.score - prev_score) # + (1 + 0.01 * self.game.score)
        elif self.game.score < prev_score:
            reward -= 15 # new
        elif self.game.wall_dead:
            reward -= 15
        elif self.game.body_dead:
            reward -= 8
        elif self.game.fruit_dead:
            reward -= 5

        
        else:
            self.steps_survived += 1
        
            # small survival pressure (prevents infinite loops)
            reward -= 0.01
        
            max_grid = np.sqrt(self.length_of_grid_x**2 + self.length_of_grid_y**2) #max(self.length_of_grid_x, self.length_of_grid_y)
        
            # focus on closest fruit to reduce noise
            closest_idx = np.argmin(prev_distances)
        
            prev_d = prev_distances[closest_idx]
            curr_d = current_distances[closest_idx]
        
            distance_diff = prev_d - curr_d
        
            # normalize
            shaped = 0.5 * (distance_diff / max_grid)

            fruit = self.game.fruits[closest_idx]
            # enemy fruits should be avoided
            if isinstance(fruit, EnemyFruit):
                reward -= shaped * 2
            elif (isinstance(fruit, DecayFruit)):
                value_ratio = fruit.value / fruit.max_value # current value div by max value
                reward += shaped * (1 + value_ratio * 2)
            else:
                reward += shaped
        
            # discourage moving away from fruit
            if distance_diff <= 0:
                reward -= 0.05
        
            # reduce zig-zagging
            if direction != self._prev_direction:
                reward -= 0.003

        
        # reward = 0
        # if self.game.score > prev_score:
        #     reward += 10 * (self.game.score - prev_score) + (1 + 0.01 * self.game.score) # fruit reward that increases with more apples
        # elif terminated:
        #     reward -= 15  # make death clearly worse than apple
        # else:
        #     self.steps_survived += 1
        #     # small survival reward 
        #     reward += 0.001
        #     # normalized distance shaping
        #     max_grid = max(self.length_of_grid_x, self.length_of_grid_y)
        #     for prev_d, curr_d in zip(prev_distances, current_distances):
        #         distance_diff = prev_d - curr_d
        #         reward += 0.5 * (distance_diff / max_grid) # comment out if uncommented DECAYFRUIT
        #         # weighted sum of fruits based on inverse distance (closer fruits = more reward)
        #         # reward += 0.2 * ((prev_d - curr_d) / max_grid) * (1 / (prev_d + 1e-5)) # DECAYFRUIT
        #         # penalize moving away or staying same distance from fruit to prevent circling
        #         if distance_diff <= 0:
        #             reward -= 0.02
        #     # stronger turn penalty to reduce zigzag
        #     if direction != self._prev_direction:
        #         reward -= 0.02
                
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

    

