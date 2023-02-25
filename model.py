"""
This code defines a simple Deep Q-Network (DQN) for training an agent to play games. The model consists of a
single hidden layer with a Rectified Linear Unit (ReLU) activation function and a linear output layer.
The QTrainer class handles training and implements the Q-learning algorithm with experience replay.

The Linear_QNet class defines the neural network model, which takes the input state of the game and predicts
the Q-values for each possible action. The model is a simple fully connected neural network with one hidden 
layer and a linear output layer. The forward method takes the input state as an argument, applies the linear
transformation and the activation function, and returns the output of the linear layer.

The QTrainer class initializes the optimizer and criterion and defines the train_step method, which implements
the Q-learning algorithm. The train_step method takes the current state, action, reward, next state, and done 
flag as inputs and computes the target Q-value using the Q-learning equation. The predicted Q-value is computed
by passing the current state through the neural network model. The target Q-value is computed by setting the
Q-value of the selected action to the sum of the reward and the discounted maximum Q-value of the next state.
The loss is computed as the mean squared error between the predicted and target Q-values and the optimizer is 
used to update the model parameters.

Finally, the Linear_QNet class also implements a save method that saves the model parameters to a file for later
use. The method creates a model folder if it doesn't exist and saves the parameters to a file in that folder.

Overall, this code provides a simple framework for training a DQN agent to play games. The network architecture
and training algorithm can be modified to suit different game environments and agent behaviors.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


