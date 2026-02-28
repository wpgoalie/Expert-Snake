# Source: https://www.geeksforgeeks.org/python/snake-game-in-python-using-pygame-module/, but edited for cheese variation
import pygame
import numpy as np
import random

# for debug to see death 
import time

# NOTE FOR FUTURE: normalizing improves convergence and stability (coordinates can be normalized)

class snakeGameCheese():
    def __init__(self, grid_size = np.array([80, 80], dtype=np.int32), 
                 fruit_position = None, 
                 snake_position = None, 
                 debug = False, draw = False):
        
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
        # determine which body segment is visible
        self.active_body_key = 1
        # grow tail for two turns each fruit
        self.grow_tail = False
        
        self.score = 0
        self.direction = "RIGHT"
        self.direction_switch = self.direction
        self.fruit_spawn = True
        
        if self.DRAW:
            self.fps = pygame.time.Clock()

        # determine initial positions
        self.cell_size = 10
        self.size_x = grid_size[0] * self.cell_size
        self.size_y = grid_size[1] * self.cell_size

        # set up snake position
        if snake_position is None:
            mid_x = (self.size_x // 2) // self.cell_size * self.cell_size
            mid_y = (self.size_y // 2) // self.cell_size * self.cell_size
            self.snake_position = np.array([mid_x, mid_y], dtype=np.int32)
        else:
            self.snake_position = np.array(snake_position, dtype=np.int32) * self.cell_size
        # generate snake body
        self.snake_body = np.array([
            [self.snake_position[0] - i * self.cell_size, self.snake_position[1], 
             1 if i % 2 == 0 else 0] for i in range(8)
        ], dtype=np.int32)
        # out of bounds check
        if (np.any(self.snake_body[:, 0] < 0) or 
            np.any(self.snake_body[:, 0] >= self.size_x) or 
            np.any(self.snake_body[:, 1] < 0) or 
            np.any(self.snake_body[:, 1] >= self.size_y)
           ):
            raise ValueError("Snake body goes off the grid, adjust snake_position")

        # fruit position
        if fruit_position is None:
            self.spawn_fruit()
        else:
            self.fruit_position = fruit_position * self.cell_size

        if self.DRAW:
            pygame.init()
            pygame.display.set_caption('Snake Game with Cheese Variation')
            self.game_screen = pygame.display.set_mode((self.size_x, self.size_y))

    def spawn_fruit(self):
        while True:
            grid_x = random.randrange(0, self.size_x // self.cell_size)
            grid_y = random.randrange(0, self.size_y // self.cell_size)
            candidate = np.array([grid_x, grid_y]) * self.cell_size

            # check collision with snake body
            collision = any(
                (segment[0] == candidate[0] and segment[1] == candidate[1])
                for segment in self.snake_body
            )
            if not collision:
                self.fruit_position = candidate
                self.fruit_spawn = True
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
            self.snake_position[1] -= self.cell_size
        if self.direction == 'DOWN':
            self.snake_position[1] += self.cell_size
        if self.direction == 'LEFT':
            self.snake_position[0] -= self.cell_size
        if self.direction == 'RIGHT':
            self.snake_position[0] += self.cell_size

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

        if self.snake_position[0] == self.fruit_position[0] and self.snake_position[1] == self.fruit_position[1]:
            self.score += 1
            self.fruit_spawn = False
            self.grow_tail = True
            if self.DEBUG:
                with open(self.log, "a") as f:
                    print('****************APPLE*****************', file=f)
        elif self.grow_tail:
            self.grow_tail = False
        else:
            self.snake_body = self.snake_body[:-1]

        # keep spawning at random coordinate until no collision with body occurs
        if not self.fruit_spawn:
            self.spawn_fruit()

        if self.DRAW:
            self.game_screen.fill('green')
            # draw head regardless of its assigned visibility
            head = self.snake_body[0]
            pygame.draw.rect(self.game_screen, 'cyan', pygame.Rect(head[0], head[1], self.cell_size, self.cell_size))
            # draw body parts alternating by body key
            for pos in self.snake_body[1:]:
                if pos[2] == self.active_body_key: # actual body, not skipped part
                    pygame.draw.rect(self.game_screen, 'blue',
                                    pygame.Rect(pos[0], pos[1], self.cell_size, self.cell_size))
            
            pygame.draw.rect(self.game_screen, 'red', pygame.Rect(
                self.fruit_position[0], self.fruit_position[1], self.cell_size, self.cell_size))
    
        # Game Over Condition: hit walls
        if self.snake_position[0] < 0 or self.snake_position[0] > self.size_x-self.cell_size:
            if self.DEBUG:
                with open(self.log, "a") as f:
                    print("HIT WALL IN X-DIRECTION", file=f)
                if self.DRAW: 
                    time.sleep(5)
            self.wall_dead = True
            return
        if self.snake_position[1] < 0 or self.snake_position[1] > self.size_y-self.cell_size:
            if self.DEBUG:
                with open(self.log, "a") as f:
                    print("HIT WALL IN Y-DIRECTION", file=f)
                if self.DRAW: 
                   time.sleep(5)
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
                time.sleep(0.01)

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
        

if __name__ == "__main__":
    newGame = snakeGameCheese(debug=True, draw=True)
    newGame.main_game()
