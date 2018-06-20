import torch.nn as nn

from basic_game import MapStatus, Direction


class Neuron(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, input_tensor):
        return self.relu(self.linear(input_tensor))  # activated output


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
        input_dim = width * height * len(MapStatus)
        # noinspection PyTypeChecker
        output_dim = len(Direction)
        hidden_dim = width * height * 2
        self.dnn = DNN(input_dim, *([hidden_dim] * 2), output_dim)
        # self.linear = nn.Linear(hidden_dim, output_dim)
        # self.softmax = nn.Softmax(1)

    def forward(self, input_tensor):
        dnn_result = self.dnn(input_tensor)
        return dnn_result
