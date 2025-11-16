import torch.nn.functional as F
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()
        # TODO: Define las capas así como la cantidad de variables de entrada y salida
        self.fc1 = nn.Linear(input_dims, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, output_dims)

    def forward(self, x):
        # TODO: Define el forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
