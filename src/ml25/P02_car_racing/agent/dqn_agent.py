import numpy as np
import torch
import torch.optim as optim
from ml_clases.proyectos.P02_car_racing.agent.networks import CNN
from ml_clases.proyectos.P02_car_racing.agent.replay_buffer import ReplayBuffer
from ml_clases.proyectos.P02_car_racing.utils import tt
import json


class DQNAgent:

    def __init__(
        self,
        num_actions,
        model_cfg={
            "batch_size": 64,
            "gamma": 0.99,
            "epsilon": 0.1,
            "tau": 0.01,
            "lr": 1e-4,
        },
        img_cfg={
            "history_length": 3,
            "skip_frames": 2,
        },
        device="cpu",
    ):
        """
        Q-Learning agent for off-policy TD control using Function Approximation.
        Finds the optimal greedy policy while following an epsilon-greedy policy.

        Args:
           Q: Action-Value function estimator (Neural Network)
           Q_target: Slowly updated target network to calculate the targets.
           num_actions: Number of actions of the environment.
           gamma: discount factor of future rewards.
           batch_size: Number of samples per batch.
           tau: indicates the speed of adjustment of the slowly updated target network.
           epsilon: Chance to sample a random action. Float betwen 0 and 1.
           lr: learning rate of the optimizer
        """
        self.model_cfg = model_cfg
        self.img_cfg = img_cfg
        # setup networks
        history_length = img_cfg.get("history_length")
        self.Q = CNN(history_length, num_actions).to(device)
        self.Q_target = CNN(history_length, num_actions).to(device)

        self.Q_target.load_state_dict(self.Q.state_dict())

        # define replay buffer
        self.replay_buffer = ReplayBuffer()

        # parameters
        self.batch_size = model_cfg.get("batch_size")
        self.gamma = model_cfg.get("gamma")
        self.tau = model_cfg.get("tau")
        self.epsilon = model_cfg.get("epsilon")

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.Q.parameters(), lr=model_cfg.get("lr"))

        self.num_actions = num_actions
        self.device = device

    def train(self, state, action, next_state, reward, terminal):
        """
        This method stores a transition to the replay buffer and updates the Q networks.
        """
        # TODO:
        # 1. add current transition to replay buffer
        ...

        # TODO: 2. sample next BATCH and perform batch update:
        (
            batch_states,
            batch_actions,
            batch_next_states,
            batch_rewards,
            batch_terminal_flags,
        ) = ...

        # TODO: use tt() function to transform arrays to tensors
        (
            batch_states,
            batch_actions,
            batch_next_states,
            batch_rewards,
            batch_terminal_flags,
        ) = (
            tt(batch_states),
            tt(batch_actions),
            tt(batch_next_states),
            tt(batch_rewards),
            tt(batch_terminal_flags),
        )

        # TODO: 2.1 compute td targets and loss
        #  td_target =  reward + discount * max_a Q_target(next_state_batch, a)

        td_target = ...
        current_prediction = ...
        loss = ...

        #  TODO: 2.2 update the Q network
        self.optimizer.zero_grad()
        ...

        # Call soft update for updating target network
        self.soft_update()

    def act(self, state, deterministic):
        """
        This method creates an epsilon-greedy policy based on the Q-function approximator and epsilon (probability to select a random action)
        Args:
            state: current state input
            deterministic:  if True, the agent should execute the argmax action (False in training, True in evaluation)
        Returns:
            action id
        """
        r = np.random.uniform()
        if deterministic or r > self.epsilon:
            # TODO: take greedy action (argmax)
            # Consider that the state needs to be converted to a torch tensor
            # return the action id as an integer
            action_id = ...
        else:

            # TODO: sample random action
            # Hint for the exploration in CarRacing: sampling the action from a uniform distribution will probably not work.
            # You can sample the agents actions with different probabilities (need to sum up to 1) so that
            # the agent will prefer to accelerate or going straight.
            # To see how the agent explores, turn the rendering in the training on and look what the agent is doing.
            action_id = ...

        return action_id

    def save(self, file_name):
        torch.save(self.Q.state_dict(), file_name + ".pt")
        with open(f"{file_name}_cfg.json", "w") as f:
            json.dump(
                {"models_cfg": self.model_cfg, "img_cfg": self.img_cfg}, f, indent=4
            )

    def load(self, file_name):
        # Load model weights
        state_dict = torch.load(file_name, map_location="cpu")
        self.Q.load_state_dict(state_dict)
        self.Q_target.load_state_dict(state_dict)

        # Load config
        cfg_path = file_name.replace(".pt", "_cfg.json")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

        # Restore configuration attributes
        self.model_cfg = cfg.get("models_cfg", {})
        self.img_cfg = cfg.get("img_cfg", {})

    def soft_update(self):
        target = self.Q_target
        source = self.Q
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
