"""
generate random states for maze environment
"""

from absl import app
from active_imitation_learning.utils import make_env
from ml_collections import ConfigDict
import pickle
import numpy as np
import tqdm


def main(_):
    # TODO: randomize the velocity
    config = ConfigDict(
        dict(
            env="MAZE",
            env_id="PointMaze_UMazeDense-v3",
            seed=0,
            max_episode_steps=1000,
            num_states=100000,
            num_envs=100,
        )
    )
    output_file = f"/scr/aliang80/active_imitation_learning/active_imitation_learning/datasets/sac_maze_random_states_{config.num_states}.pkl"
    env = make_env(
        config.env,
        config.env_id,
        config.seed,
        num_envs=config.num_envs,
        maze_map="all_goals",
        max_episode_steps=config.max_episode_steps,
        freeze_rand_vec=False,
    )

    # reset a bunch of times and store the states
    states = []

    for _ in tqdm.tqdm(range(config.num_states // config.num_envs)):
        state, _ = env.reset()
        state = np.concatenate(
            (state["observation"], state["desired_goal"], state["meta"]), axis=-1
        )
        states.append(state)

    states = np.concatenate(states)

    # change velocity to be sampled uniformly between (-5, 5)
    states[:, 2] = np.random.uniform(-1, 1, size=(config.num_states,))

    # save to file
    with open(output_file, "wb") as f:
        pickle.dump(states, f)


if __name__ == "__main__":
    app.run(main)
