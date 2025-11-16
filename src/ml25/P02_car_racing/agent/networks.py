import torch.nn as nn
import torch
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, history_length=0, n_classes=3):
        super(CNN, self).__init__()
        # TODO : define layers of a convolutional neural network
        # ¿ how many images to use as input? (this is your input channels)
        # How many actions can your agent execute? (this is your output neurons)

    def forward(self, x):
        batch_size = x.shape[0]
        # TODO: implement forward pass

        return x

    def calc_out_size(self, w, h, kernel_size, padding, stride):
        # TODO: Implement the function to get the output size for a cnn given the parameters
        return width, height
