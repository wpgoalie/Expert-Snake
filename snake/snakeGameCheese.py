# Source: https://www.geeksforgeeks.org/python/snake-game-in-python-using-pygame-module/, but edited for cheese variation
import pygame
import numpy as np
import random
from fruit import Fruit, DecayFruit, EnemyFruit

# for debug to see death 
import time

class snakeGameCheese():
    def __init__(self, grid_size = np.array([80, 80], dtype=np.int32), 
                 fruit_positions = None, 
                 snake_position = None, 
                 debug = False, draw = False):

        self.grid_size = grid_size
        self.cell_size = 10
        self.size_x = self.grid_size[0] * self.cell_size
        self.size_y = self.grid_size[1] * self.cell_size
        self.DEBUG = debug
        self.DRAW = draw
        
        if self.DEBUG:
            self.log = "log.txt"
            # clear log file so we only have current run
            with open(self.log, "w") as f:
                pass

        # game state instead of quitting immediately
        self.wall_dead = False
        self.body_dead = False
        self.fruit_dead = False
        # determine which body segment is visible
        self.active_body_key = 1
        # keeps track of how many turns to grow for
        self.grow_tail = 0
        
        self.score = 0
        self.direction = "RIGHT"
        self.direction_switch = self.direction
        
        if self.DRAW:
            self.fps = pygame.time.Clock()

        # set up snake position
        if snake_position is None:
            mid_x = self.grid_size[0] // 2
            mid_y = self.grid_size[1] // 2
            self.snake_position = np.array([mid_x, mid_y], dtype=np.int32)
        else:
            self.snake_position = np.array(snake_position, dtype=np.int32)
        # generate snake body
        self.snake_body = np.array([
            [self.snake_position[0] - i, self.snake_position[1], 
             1 if i % 2 == 0 else 0] for i in range(8)
        ], dtype=np.int32)
        # out of bounds check
        if (np.any(self.snake_body[:, 0] < 0) or 
            np.any(self.snake_body[:, 0] >= self.grid_size[0]) or 
            np.any(self.snake_body[:, 1] < 0) or 
            np.any(self.snake_body[:, 1] >= self.grid_size[1])
           ):
            raise ValueError("Snake body goes off the grid, adjust snake_position")

        # generate fruits and place in positions
        self.fruits = []
        if fruit_positions is not None:
            self.fruits.append(Fruit(fruit_positions[0]))
            self.fruits.append(DecayFruit(fruit_positions[1])) # DECAYFRUIT
            self.fruits.append(EnemyFruit(fruit_positions[2])) # ENEMYFRUIT
        else:
            self.fruits.append(Fruit(np.array([0,0])))
            self.fruits.append(DecayFruit(np.array([0,0]))) # DECAYFRUIT
            self.fruits.append(EnemyFruit(np.array([0,0]))) # ENEMYFRUIT
            # respawn each at random positions
            for i in range(len(self.fruits)):
                self.spawn_fruit(i)

        if self.DRAW:
            pygame.init()
            pygame.display.set_caption('Snake Game with Cheese Variation')
            self.game_screen = pygame.display.set_mode((self.size_x, self.size_y))

        # keep track of each type of fruit
        self.fruit_cnts = {type(fruit).__name__: 0 for fruit in self.fruits}
    
    def grid_to_pixel(self, pos, cell_size):
        return np.array(pos) * cell_size

    def spawn_fruit(self, index, eaten=False):
        # increase fruit count
        fruit_class = type(self.fruits[index])  # get fruit type
        if eaten:
            self.fruit_cnts[fruit_class.__name__] += 1
        while True:
            x = random.randrange(0, self.grid_size[0])
            y = random.randrange(0, self.grid_size[1])
            candidate = np.array([x, y], dtype=np.int32)
            border_check = True

            # Checking if the square path will fit inside the screen or not
            if fruit_class == EnemyFruit:
                length_check = candidate[0] + self.fruits[index].path_length < self.grid_size[0]
                height_check = candidate[1] + self.fruits[index].path_length < self.grid_size[1]
                if not length_check or not height_check:
                    border_check = False

            # check collisions
            collision = any((seg[0] == x and seg[1] == y) for seg in self.snake_body)
            fruit_collision = any(
                (f.position[0] == x and f.position[1] == y) for i,f in enumerate(self.fruits) if i != index
            )
    
            if not collision and not fruit_collision and border_check:
                # respawn the fruit at this index
                if fruit_class == DecayFruit:
                    self.fruits[index] = DecayFruit(candidate)
                else:
                    self.fruits[index] = fruit_class(candidate)
                break

    def score_display(self, color, font, size):
        # creating font object score_font
        score_font = pygame.font.SysFont(font, size)
        
        # create the display surface object 
        # score_surface
        score_surface = score_font.render('Score : ' + str(self.score), True, color)
        
        # create a rectangular object for the text
        # surface object
        score_rect = score_surface.get_rect()
        
        # displaying text
        self.game_screen.blit(score_surface, score_rect)

    def fruit_eaten(self, idx, fruit):
        self.score += fruit.on_eat()
        # grow tail for two turns each fruit
        self.grow_tail += fruit.value * 2
        if self.DEBUG:
            with open(self.log, "a") as f:
                print(f'****************EAT FRUIT: {fruit.color.upper()}, {fruit.value}*****************', file=f)
        # respawn this fruit at same index
        self.spawn_fruit(idx, eaten=True)

    def step_function(self, action):
        # prevent snake turning 180 degrees
        self.direction_switch = action
        if self.direction_switch == 'UP' and self.direction != 'DOWN':
            self.direction = 'UP'
        if self.direction_switch == 'DOWN' and self.direction != 'UP':
            self.direction = 'DOWN'
        if self.direction_switch == 'LEFT' and self.direction != 'RIGHT':
            self.direction = 'LEFT'
        if self.direction_switch == 'RIGHT' and self.direction != 'LEFT':
            self.direction = 'RIGHT'
    
        # Moving the snake
        if self.direction == 'UP':
            self.snake_position[1] -= 1
        if self.direction == 'DOWN':
            self.snake_position[1] += 1
        if self.direction == 'LEFT':
            self.snake_position[0] -= 1
        if self.direction == 'RIGHT':
            self.snake_position[0] += 1

        # Snake body growing mechanism
        # determine if this is visible or inviisble segment
        body_key = int(not self.snake_body[0][2])
        # add new head
        new_head = np.append(self.snake_position, body_key)
        self.snake_body = np.insert(self.snake_body, 0, new_head, axis=0)
        
        if self.DEBUG:
            with open(self.log, "a") as f:
                print('===================================================', file=f)
                print(self.snake_body, file=f)
                print("LENGTH: ", np.sum(self.snake_body[:, 2] == self.active_body_key), file=f)
                print('===================================================', file=f)
                print("LEFTOVER GROWTH: ", self.grow_tail, file=f)
                
        for i, fruit in enumerate(self.fruits):
            if isinstance(fruit, EnemyFruit):
                for snake_part in self.snake_body:
                    if snake_part[2] == 1 and np.array_equal(fruit.position, snake_part[:-1]):
                        self.fruit_eaten(i, fruit)
                        break
            elif np.array_equal(fruit.position, self.snake_position):
                self.fruit_eaten(i, fruit)

        if self.grow_tail < 0:
            if abs(self.grow_tail) > len(self.snake_body):
                if self.DEBUG:
                    with open(self.log, "a") as f:
                        print("FRUIT DEATH, DEDUCTION LONGER THAN CURRENT BODY", file=f)
                    if self.DRAW: 
                        time.sleep(3)
                self.fruit_dead = True
                return
            else:
                # negative means we have to remove part of the snake off
                self.snake_body = self.snake_body[:self.grow_tail]
                self.grow_tail = 0
            
        # if not eaten, update fruit values
        for i, fruit in enumerate(self.fruits):
            fruit.update()
            if not fruit.active: # disappears/no longer active
                self.spawn_fruit(i, eaten=False)

        if self.grow_tail > 0:
            self.grow_tail -= 1
        else:
            self.snake_body = self.snake_body[:-1]

        if self.DRAW:
            self.game_screen.fill('green')
            # draw head regardless of its assigned visibility
            head = self.snake_body[0]
            head_pixel = self.grid_to_pixel(head[:2], self.cell_size)
            pygame.draw.rect(self.game_screen, 'cyan', pygame.Rect(head_pixel[0], head_pixel[1], self.cell_size, self.cell_size))
            # draw body parts alternating by body key
            for pos in self.snake_body[1:]:
                if pos[2] == self.active_body_key: # actual body, not skipped part
                    pixel_pos = self.grid_to_pixel(pos[:2], self.cell_size)
                    pygame.draw.rect(self.game_screen, 'blue',
                                    pygame.Rect(pixel_pos[0], pixel_pos[1], self.cell_size, self.cell_size))
            
            for fruit in self.fruits:
                fruit_pixel = self.grid_to_pixel(fruit.position, self.cell_size)
                pygame.draw.rect(self.game_screen, fruit.color, 
                                    pygame.Rect(fruit_pixel[0], fruit_pixel[1], self.cell_size, self.cell_size))
    
        # Game Over Condition: hit walls
        if self.snake_position[0] < 0 or self.snake_position[0] >= self.grid_size[0]:
            if self.DEBUG:
                with open(self.log, "a") as f:
                    print("HIT WALL IN X-DIRECTION", file=f)
                if self.DRAW: 
                    time.sleep(3)
            self.wall_dead = True
            return
        if self.snake_position[1] < 0 or self.snake_position[1] >= self.grid_size[1]:
            if self.DEBUG:
                with open(self.log, "a") as f:
                    print("HIT WALL IN Y-DIRECTION", file=f)
                if self.DRAW: 
                   time.sleep(3)
            self.wall_dead = True
            return
    
        # Game Over Condition: touching snake body
        for pos in self.snake_body[1:]:
            if pos[2] == self.active_body_key and self.snake_position[0] == pos[0] and self.snake_position[1] == pos[1]:
                if self.DEBUG:
                    with open(self.log, "a") as f:
                        print("DIED BECAUSE OF ", pos, file=f)
                        print("BODY KEY:", self.active_body_key, file=f)
                    if self.DRAW: 
                        time.sleep(5)
                self.body_dead = True
                return
                
        if self.DRAW:
            # displaying score continuously
            self.score_display('white', 'times new roman', 30)
        
            # Refresh game screen
            pygame.display.update()
        
            # Frame Per Second /Refresh Rate
            self.fps.tick(15)

    def main_game(self):
        # Main Function
        while True:  
            # time delay for user to react
            if self.DRAW:
                time.sleep(0.05)

            # handling key events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.direction_switch = 'UP'
                    if event.key == pygame.K_DOWN:
                        self.direction_switch = 'DOWN'
                    if event.key == pygame.K_LEFT:
                        self.direction_switch = 'LEFT'
                    if event.key == pygame.K_RIGHT:
                        self.direction_switch = 'RIGHT'

                elif self.DRAW and event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            self.step_function(self.direction_switch)

            # stop running game
            if self.wall_dead or self.body_dead or self.fruit_dead:
                break
        

if __name__ == "__main__":
    newGame = snakeGameCheese(debug=True, draw=True, grid_size = np.array([30, 24], dtype=np.int32))
    newGame.main_game()
