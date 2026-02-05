# Source: https://www.geeksforgeeks.org/python/snake-game-in-python-using-pygame-module/
import pygame
import numpy as np
import random

# for debug to see death 
import time

class snakeGame():
    def __init__(self, fruit_position = np.array([500, 500], dtype=np.int32), snake_position = np.array([350, 250], dtype=np.int32)):
        self.DEBUG = True
        self.log = "log.txt"
        # clear log file so we only have current run
        with open(self.log, "w") as f:
            pass

        # cheese-specific instructions
        # determine which body segment is visible
        self.active_body_key = 1
        # grow tail for two turns each fruit
        self.grow_tail = False

        self.size_x = 800
        self.size_y = 800
        self.score = 0
        self.fps = pygame.time.Clock()
        # fruit position
        self.fruit_position = [random.randrange(1, (self.size_x//10)) * 10, 
                          random.randrange(1, (self.size_y//10)) * 10]
        self.fruit_spawn = True
        self.snake_position = np.array([350, 250], dtype=np.int32)
        # defining first 4 blocks of snake body
        self.snake_body = np.array([
                      [350, 250, 1],
                      [340, 250, 0],
                      [330, 250, 1],
                      [320, 250, 0],
                      [310, 250, 1],
                      [300, 250, 0],
                      [290, 250, 1],
                      [280, 250, 0],
                      ], dtype=np.int32)
        self.direction = "RIGHT"
        self.direction_switch = self.direction
        
        pygame.init()
        pygame.display.set_caption('Snake Game with Cheese Variation')
        self.game_screen = pygame.display.set_mode((self.size_x, self.size_y))

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
            self.snake_position[1] -= 10
        if self.direction == 'DOWN':
            self.snake_position[1] += 10
        if self.direction == 'LEFT':
            self.snake_position[0] -= 10
        if self.direction == 'RIGHT':
            self.snake_position[0] += 10

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
            
        if not self.fruit_spawn:
            self.fruit_position = [random.randrange(1, (self.size_x//10)) * 10, 
                              random.randrange(1, (self.size_y//10)) * 10]
            
        self.fruit_spawn = True
        self.game_screen.fill('green')
        
        # draw head regardless of its assigned visibility
        head = self.snake_body[0]
        pygame.draw.rect(self.game_screen, 'cyan', pygame.Rect(head[0], head[1], 10, 10))
        # draw body parts alternating by body key
        for pos in self.snake_body[1:]:
            if pos[2] == self.active_body_key: # actual body, not skipped part
                pygame.draw.rect(self.game_screen, 'blue',
                                pygame.Rect(pos[0], pos[1], 10, 10))
        
        pygame.draw.rect(self.game_screen, 'red', pygame.Rect(
            self.fruit_position[0], self.fruit_position[1], 10, 10))
    
        # Game Over Condition: hit walls
        if self.snake_position[0] < 0 or self.snake_position[0] > self.size_x-10:
            if self.DEBUG:
                with open(self.log, "a") as f:
                    print("HIT WALL IN X-DIRECTION", file=f)
                time.sleep(5)
            quit()
        if self.snake_position[1] < 0 or self.snake_position[1] > self.size_y-10:
            if self.DEBUG:
                with open(self.log, "a") as f:
                    print("HIT WALL IN Y-DIRECTION", file=f)
                time.sleep(5)
            quit()
    
        # Game Over Condition: touching snake body
        for pos in self.snake_body[1:]:
            if pos[2] == self.active_body_key and self.snake_position[0] == pos[0] and self.snake_position[1] == pos[1]:
                if self.DEBUG:
                    with open(self.log, "a") as f:
                        print("DIED BECAUSE OF ", pos, file=f)
                        print("BODY KEY:", self.active_body_key, file=f)
                    time.sleep(5)
                quit()
    
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
            if self.DEBUG:
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

                elif event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            self.step_function(self.direction_switch)
        
            
        

        

if __name__ == "__main__":
    newGame = snakeGame()
    newGame.main_game()
