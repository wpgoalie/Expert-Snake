# Source: https://www.geeksforgeeks.org/python/snake-game-in-python-using-pygame-module/
import pygame
import numpy as np
import random

class snakeGame():
    def __init__(self):
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
        self.snake_body = np.array([[350, 250],
                      [340, 250],
                      [330, 250],
                      [320, 250],
                      [310, 250],
                      [300, 250],
                      [290, 250],
                      [280, 250],
                      [270, 250],
                      [260, 250],
                      [250, 250],
                      [240, 250],
                      [230, 250],
                      [220, 250]
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


    def main_game(self):
        # Main Function
        while True:
            
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
        
            # If two keys pressed simultaneously
            # we don't want snake to move into two 
            # directions simultaneously
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
            # if fruits and snakes collide then scores
            # will be incremented by 10
            self.snake_body = np.insert(self.snake_body, 0, self.snake_position, axis=0)
            print(self.snake_body)
            if self.snake_position[0] == self.fruit_position[0] and self.snake_position[1] == self.fruit_position[1]:
                self.score += 1
                self.fruit_spawn = False
            else:
                self.snake_body = self.snake_body[:-1]
                
            if not self.fruit_spawn:
                self.fruit_position = [random.randrange(1, (self.size_x//10)) * 10, 
                                  random.randrange(1, (self.size_y//10)) * 10]
                
            self.fruit_spawn = True
            self.game_screen.fill('black')
            
            for key, pos in enumerate(self.snake_body):
                if key % 2 == 0:
                    pygame.draw.rect(self.game_screen, 'orange',
                                    pygame.Rect(pos[0], pos[1], 10, 10))
                else:
                    pygame.draw.rect(self.game_screen, 'white',
                                    pygame.Rect(pos[0], pos[1], 10, 10))
            
            pygame.draw.rect(self.game_screen, 'yellow', pygame.Rect(
                self.fruit_position[0], self.fruit_position[1], 10, 10))
        
            # Game Over conditions
            if self.snake_position[0] < 0 or self.snake_position[0] > self.size_x-10:
                quit()
            if self.snake_position[1] < 0 or self.snake_position[1] > self.size_y-10:
                quit()
        
            # Touching the snake body
            for key, block in enumerate(self.snake_body[1:]):
                if self.snake_position[0] == block[0] and self.snake_position[1] == block[1] and key % 2 == 1:
                    # quit the program
                    quit()
        
            # displaying score continuously
            self.score_display('white', 'times new roman', 30)
        
            # Refresh game screen
            pygame.display.update()
        
            # Frame Per Second /Refresh Rate
            self.fps.tick(15)
        

        

if __name__ == "__main__":
    newGame = snakeGame()
    newGame.main_game()
