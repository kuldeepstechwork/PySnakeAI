"""
This Python code implements an agent for playing the game of Snake using reinforcement learning.
The agent uses a Q-learning algorithm to learn how to play the game and improve its score over time. 
The code is composed of several classes and functions that work together to train the agent and play the game.

The Agent class is the main class that represents the reinforcement learning agent. It initializes with several
parameters, including the discount rate (gamma), the maximum memory size (MAX_MEMORY), and the learning rate (LR).
It also initializes a neural network model (Linear_QNet) and a trainer (QTrainer) to train the model.

The get_state method of the Agent class takes the current game state as input and returns a numpy array
representing the state of the game. The state is a vector of 11 elements, representing information about the
position of the snake's head, the direction it is moving, and the position of the food.

The remember method of the Agent class stores the current state, action, reward, next state, and done flag in
a deque memory. The deque is used to implement a replay buffer, which allows the agent to learn from past
experiences.

The train_long_memory method of the Agent class is called when the memory buffer has reached the maximum size.
It retrieves a random sample of experiences from the memory buffer and uses them to train the neural network model.
The training is performed by calling the train_step method of the trainer object.

The train_short_memory method of the Agent class is called after each game step to train the model on the current
experience. This method calls the train_step method of the trainer object with the current experience.

The get_action method of the Agent class takes the current state as input and returns the action to take.
The action is chosen randomly with a probability proportional to the agent's exploration rate (epsilon).
Otherwise, the agent selects the action with the highest Q-value according to its neural network model.

The train function is the main function that initializes the agent and plays the game. It starts by creating
an instance of the Agent class and the SnakeGameAI class, which represents the game environment. The function then enters a loop where it performs the following steps:

Gets the current state of the game using the get_state method of the agent object.
Chooses the action to take using the get_action method of the agent object.
Performs the action on the game environment using the play_step method of the game object and gets the reward,
done flag, and score.
Trains the agent on the current experience using the train_short_memory and remember methods of the agent object.
If the game is over, resets the game environment and trains the agent on the long-term memory using the 
train_long_memory method of the agent object.
Updates the score and plots the score and mean score over time using the plot function from the helper module.
Overall, this code provides a simple implementation of a reinforcement learning agent for playing the game 
of Snake. It demonstrates the use of Q-learning and replay buffers to train an agent, as well as the 
importance of balancing exploration and exploitation when selecting actions.
"""

import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()