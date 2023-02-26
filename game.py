"""
This code is a Python implementation of the classic game Snake, which is played on a rectangular board.
In this game, the player controls a snake that moves around the board and grows longer as it eats food.
The objective is to avoid running into the walls or the snake's own body, while trying to eat as much food as
possible to gain points.

The game is implemented using the Pygame library, which provides functions for creating a graphical display,
detecting user input, and updating the screen.
The game state is represented by several variables, including the current direction of the snake, its position on
the board, the locations of the food and the snake's body, the current score, and the frame iteration
(i.e. the number of times the game has been updated since it started).

The game loop consists of several steps:
Collect user input: Pygame's event handling system is used to detect when the user has closed the window,
and the game is quit if this happens.

Move: The snake's position is updated based on the current direction, and its body is shifted accordingly.
The new head position is determined by adding or subtracting a fixed block size (20 pixels in this case) in the
x or y direction, depending on the direction of movement.

Check if game over: The game is considered over if the snake collides with the walls or its own body.
In this case, the game_over variable is set to True and the reward is set to -10.

Place new food or just move: If the snake's head is at the same position as the food, the food is consumed and a 
new one is placed randomly on the board. If not, the snake's tail is removed to simulate movement.

Update UI and clock: The Pygame display is updated to reflect the new game state, including the positions of the
snake and the food, the score, and any other relevant information. The clock is also ticked to control the game's 
speed.

Return game over and score: The function returns the current reward, game_over and score to the caller.

The game also includes a SnakeGameAI class, which provides an interface for an artificial intelligence (AI)
to interact with the game. The AI is expected to call the play_step method repeatedly, passing in its chosen
action (a one-hot encoded vector representing the desired direction of movement) and receiving the resulting
reward, game_over and score.

The SnakeGameAI class is a simple implementation of the classic Snake game, with an AI player instead of a 
human player. The game is played on a 2D grid, with the snake represented as a series of connected squares.
The objective of the game is to move the snake around the grid and collect food, while avoiding collisions with
the boundaries of the grid and the snake's own body.

The game is implemented using the Pygame library, which provides a simple way to create graphical games in Python.
The pygame.init() function is called at the beginning of the script to initialize the library.

The Direction class is an enumeration of the four possible directions that the snake can move: right, left, up,
and down. The Point class is a simple named tuple that represents a point on the grid, with x and y coordinates.

The RGB color constants WHITE, RED, BLUE1, BLUE2, and BLACK are defined to be used later for drawing the game
elements.

The BLOCK_SIZE constant determines the size of each square in the grid, and SPEED determines the speed of the game.

The SnakeGameAI class has several methods to initialize, reset, and play the game. The __init__() method sets
up the game window, with a specified width and height, and sets the game clock.

The reset() method initializes the game state, setting the starting direction, the head of the snake, and the
initial snake body. It also sets the score to zero, places the food on the grid, and sets the frame iteration to zero.

The _place_food() method places the food randomly on the grid, avoiding any locations where the snake currently is.
If the food location is the same as the snake, it recursively tries again until a valid location is found.

The play_step() method is called for each step of the game, and takes an action as an argument. The action
is a one-hot encoded vector representing the desired direction of movement for the snake. The method first
collects any user input events, such as quitting the game, then moves the snake according to the action.
If the snake collides with a boundary or its own body, or the frame iteration exceeds a certain threshold,
the game is over. If the snake eats the food, the score is increased, the food is placed in a new location, 
and the snake grows. The method also updates the game window, ticks the clock, and returns the reward, game
over status, and score.

The is_collision() method checks whether a given point is a collision with the boundaries of the grid or the
snake's own body. If no point is provided, it checks the head of the snake by default.

The _update_ui() method clears the game window, draws the snake and food, and displays the score.

The _move() method updates the direction of the snake based on the action, then moves the head of the snake in
the new direction.

"""

import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (0, 0, 0)
RED = (238,238,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (245,255,250)

BLOCK_SIZE = 20
SPEED = 40

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h),pygame.RESIZABLE)
        pygame.display.set_caption('Snake AI By : Kuldeep')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # [straight[1,0,0], right[0,1,0], left[0,0,1]]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> up
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)