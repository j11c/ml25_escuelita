import numpy as np
import torch
from torch.autograd import Variable

action_to_id = {
    "LEFT": 1,
    "RIGHT": 2,
    "STRAIGHT": 0,
    "ACCELERATE": 3,
    "BRAKE": 4,
}


def tt(ndarray):
    return Variable(torch.from_numpy(ndarray).float().cuda(), requires_grad=False)


def rgb2gray(rgb):
    """
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[..., :3], [0.2125, 0.7154, 0.0721])
    return gray.astype("float32")


class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """

    def __init__(self):
        self.episode_reward = 0
        self.actions_ids = []

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)

    def get_action_usage(self, action_name):
        action_id = action_to_id[action_name]
        ids = np.array(self.actions_ids)
        return len(ids[ids == action_id]) / len(ids)
