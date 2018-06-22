import torch.nn as nn

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
    def __init__(self, width, height):
        super().__init__()
        # noinspection PyTypeChecker
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        # noinspection PyTypeChecker
        self.head = nn.Linear(width * height * 32, len(Direction))
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
