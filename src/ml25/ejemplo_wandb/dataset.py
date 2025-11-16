import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


class HousingDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data.astype("float32")
        self.labels = labels.astype("float32")

        # TODO: calcula la cantidad de variables de entrada y salida
        self.input_dims = data.shape[1]  # N, D
        self.output_dims = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(self.data.shape, type(self.data), idx)
        # print(self.labels.shape, type(self.labels), idx)
        datapoint = self.data[idx]
        label = self.labels[idx]
        label = np.expand_dims(label, 0)  # Transformarlo a vector de 1x1
        return datapoint, label
