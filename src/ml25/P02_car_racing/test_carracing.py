import os
import gym
from ml_clases.proyectos.P02_car_racing.agent.dqn_agent import DQNAgent
from ml_clases.proyectos.P02_car_racing.train_carracing import run_episode
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm
from datetime import datetime, timezone
import torch

np.random.seed(0)
this_dir = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(this_dir, "./test_results"))
MODELS_DIR = os.path.join(this_dir, "./models_carracing")

if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)


if __name__ == "__main__":
    env = gym.make("CarRacing-v3", continuous=False, render_mode="human")
    history_length = 0

    # TODO: Replace with desired model WITHOUT the extension
    model_name = "dqn_agent_780"
    agent_path = os.path.abspath(f"{MODELS_DIR}/{model_name}")

    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = DQNAgent(
        n_actions,
        device=device,
    )

    # Will load model weights and configuration
    agent.load(agent_path)
    n_test_episodes = 15

    episode_rewards = []
    for i in tqdm(range(n_test_episodes)):
        stats = run_episode(
            env,
            agent,
            deterministic=True,
            img_cgf=cfg.get("image_preprocessing"),
            do_training=False,
            max_timesteps=1000,
            rendering=True,
        )
        episode_rewards.append(stats.episode_reward)
        print("episode ", i, "reward ", stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["model_name"] = model_name
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
    print("mean reward over " + str(n_test_episodes) + " episodes:", results["mean"])

    now_utc = datetime.now(timezone.utc)
    timestamp = now_utc.strftime("%Y-%m-%d_%H-%M-%S-%f")

    fname = f"{RESULTS_DIR}/carracing_results_dqn_{now_utc}.json"
    fh = open(fname, "w")
    json.dump(results, fh)

    env.close()
    print("... finished")
