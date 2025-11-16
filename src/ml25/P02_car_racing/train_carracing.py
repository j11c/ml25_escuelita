import os
import datetime
import numpy as np
import gymnasium as gym
import torch
from datetime import datetime, timezone
from tqdm import tqdm
import wandb

from ml_clases.proyectos.P02_car_racing.utils import (
    EpisodeStats,
    rgb2gray,
)
from ml_clases.proyectos.P02_car_racing.agent.dqn_agent import DQNAgent

this_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(this_dir, "./models_carracing")
if not os.path.exists(MODELS_DIR):
    os.mkdir(MODELS_DIR)

print(f"MODELS WILL BE STORED AT {MODELS_DIR}")


def run_eval(
    run, env, agent, eval_cfg, img_cfg, curr_episode, max_episodes, best_return
):
    num_eval_episodes = eval_cfg.get("n_episodes")
    eval_every_n_ep = eval_cfg.get("every_n_ep")

    mean_reward = 0
    if curr_episode % eval_every_n_ep == 0:
        for j in range(num_eval_episodes):
            stats = run_episode(
                env,
                agent,
                img_cgf=img_cfg,
                deterministic=True,
                rendering=False,
                max_timesteps=1000,
                do_training=False,
            )
            mean_reward += stats.episode_reward
        mean_reward /= num_eval_episodes
        run.log({"val/mean_return": mean_reward, "episode": curr_episode})
        print(
            f"[EVAL] Episode {curr_episode}: Mean Reward: {mean_reward} over {num_eval_episodes} episodes"
        )

    # Save model every eval_every_n_ep episodes
    eval_cond = curr_episode % eval_every_n_ep == 0 or (
        curr_episode >= max_episodes - 1
    )
    best_reward_cond = mean_reward > best_return
    if eval_cond:
        model_name = (
            "dqn_agent_best_eval" if best_reward_cond else "dqn_agent_{curr_episode}"
        )
        agent.save(os.path.join(MODELS_DIR, model_name))


def run_episode(
    env,
    agent,
    deterministic,
    img_cgf={},
    do_training=True,
    rendering=False,
    max_timesteps=1000,
):
    """
    This methods runs one episode for a gym environment.
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    step = 0
    state = env.reset()[0]

    # Append image history to first state
    image_hist = []
    history_length = img_cgf.get("history_length", 0)
    skip_frames = img_cgf.get("skip_frames", 0)
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1)

    # we use while true since the agent can finish before max_timesteps (terminal or max timesteps)
    while True:
        state_cnn = np.expand_dims(np.transpose(state, (2, 0, 1)), 0)
        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly.
        # change state to match torch cnn dimensions (batch, channels,w,h)
        action = agent.act(state_cnn, deterministic)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, truncated, info = env.step(action)
            reward += r
            if rendering:
                env.render()

            if terminal:
                break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).reshape(96, 96, history_length + 1)

        if do_training:
            # changed to match torch dims (channels, w,h)
            state_switch_channels = np.transpose(state, (2, 0, 1))
            next_switch_channels = np.transpose(next_state, (2, 0, 1))
            agent.train(
                state_switch_channels, action, next_switch_channels, reward, terminal
            )

        stats.step(reward, action)

        state = next_state

        if terminal or (step * (skip_frames + 1)) > max_timesteps:
            break
        step += 1
    # return episode statistics
    return stats


def train_online(run, env, agent, num_episodes, img_cfg={}, eval_cfg={}):
    print("TRAINING AGENT STARTED...")

    best_return = -float("inf")
    max_timesteps = 200
    for ep in tqdm(range(num_episodes)):
        # After 300 episodes, allow longer episodes
        if ep > 300:
            max_timesteps = 1000

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)
        stats = run_episode(
            env,
            agent,
            deterministic=False,
            img_cgf=img_cfg,
            do_training=True,
            max_timesteps=max_timesteps,
        )

        run.log(
            {
                "episode": ep,
                "train/ep_return": stats.episode_reward,
                "train/straight": stats.get_action_usage("STRAIGHT"),
                "train/left": stats.get_action_usage("LEFT"),
                "train/right": stats.get_action_usage("RIGHT"),
                "train/accel": stats.get_action_usage("ACCELERATE"),
                "train/brake": stats.get_action_usage("BRAKE"),
            }
        )

        # ----- Evaluation ------ #
        run_eval(run, env, agent, eval_cfg, img_cfg, ep, num_episodes, best_return)
        # visualize learning every 100 episodes
        if ep % 100 == 0:
            if max_timesteps < 1000:
                max_timesteps += 150
            else:
                max_timesteps = 1000
            # Run one episode with rendering
            run_episode(
                env,
                agent,
                deterministic=True,
                img_cgf=img_cfg,
                do_training=False,
                rendering=True,
                max_timesteps=300,
            )


def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0


def init_wandb(cfg):
    # Initialize wandb
    now_utc = datetime.now(timezone.utc)
    timestamp = now_utc.strftime("%Y-%m-%d_%H-%M-%S-%f")

    run = wandb.init(
        project="CarRacing-DQN",
        config=cfg,
        name=f"DQN-CarRacing_{timestamp}_utc",
    )
    return run


if __name__ == "__main__":
    # https://gymnasium.farama.org/environments/box2d/car_racing/
    # pip install Box2D gymnasium
    env = gym.make(
        "CarRacing-v3", continuous=False, render_mode="human"
    )  # We load the environment as discrete to have a discrete action space, state space is the image
    # Hyperparams
    cfg = {
        "evaluation": {
            "n_episodes": 5,
            "every_n_ep": 20,  # run evaluation every n episodes
        },
        "training": {"n_episodes": 1000},
        "model": {
            "batch_size": 64,
            "gamma": 0.99,
            "epsilon": 0.1,
            "tau": 0.01,
            "lr": 1e-4,
        },
        "image_preprocessing": {
            "history_length": 3,
            "skip_frames": 2,
        },
    }
    run = init_wandb(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_actions = env.action_space.n  # 5
    model_cfg = cfg.get("model")
    agent = DQNAgent(
        n_actions,
        model_cfg,
        img_cfg=cfg.get("image_preprocessing"),
        device=device,
    )

    train_cfg = cfg.get("training")
    train_online(
        run,
        env,
        agent,
        train_cfg.get("n_episodes"),
        img_cfg=cfg.get("image_preprocessing"),
        eval_cfg=cfg.get("evaluation"),
    )
