import torch.nn as nn
import torch
from basic_game import Direction


class Neuron(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, input_tensor):
        activated = self.relu(self.linear(input_tensor))
        return activated


class DNN(nn.Module):
    def __init__(self, input_dim: int, *args: [int]):
        super().__init__()
        if len(args) < 1:
            raise ValueError("A dnn should have at least one layer")
        self.input_dim = input_dim
        self.neuron = nn.ModuleList()

        network_in = input_dim

        for new_dim in args:
            self.neuron.append(Neuron(network_in, new_dim))
            network_in = new_dim

    def forward(self, input_tensor):
        for neuron in self.neuron:
            input_tensor = neuron(input_tensor)
        return input_tensor


class DQN(nn.Module):
    def __init__(self, width, height, hidden_size=128, num_layers=1):
        super().__init__()
        # noinspection PyTypeChecker
        output_dim = len(Direction)
        self.lstm = nn.LSTM(2, hidden_size, num_layers, batch_first=True)
        self.width = width
        self.height = height
        self.hidden = self.init_hidden()
        self.linear = nn.Linear(hidden_size+2, output_dim)

    def init_hidden(self, batch=1):
        hidden_size = self.lstm.num_layers, batch, self.lstm.hidden_size
        return torch.randn(hidden_size), torch.randn(hidden_size)

    def forward(self, x):
        if self.hidden[0].size(1) != len(x):
            self.hidden = self.init_hidden(batch=len(x))
        # x: batch sequence input
        food = x[:, 0]
        snake = x[:, 1:]
        memory, self.hidden = self.lstm(snake, self.hidden)
        memory = memory[:, -1]
        combined = torch.cat((food, memory), dim=1)
        output = self.linear(combined)
        return output
