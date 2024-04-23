import metaworld
import gym as old_gym
import gymnasium as gym
import numpy as np

MAZE_MAPS = {
    "standard": [
        [1, 1, 1, 1, 1, 1, 1],
        [1, "g", "g", "g", "g", "g", 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, "r", "r", "r", "r", "r", 1],
        [1, 1, 1, 1, 1, 1, 1],
    ],
    "all_goals": [
        [1, 1, 1, 1, 1, 1, 1],
        [1, "c", "c", "c", "c", "c", 1],
        [1, "c", "c", "c", "c", "c", 1],
        [1, "c", "c", "c", "c", "c", 1],
        [1, "c", "c", "c", "c", "c", 1],
        [1, "c", "c", "c", "c", "c", 1],
        [1, 1, 1, 1, 1, 1, 1],
    ],
}


def make_env(
    env_name,
    env_id,
    seed,
    num_envs: int = 1,
    maze_map: str = "standard",
    freeze_rand_vec: bool = False,
    goal_observable: bool = True,
    max_episode_steps: int = 1000,
):
    def thunk():
        if env_name == "MW":
            # env_name = env_name.replace("metaworld-", "")
            # env_name = "drawer-open-v2"
            if goal_observable:
                env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[
                    env_id + "-goal-observable"
                ](seed=seed, render_mode="rgb_array")
            else:
                env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[
                    env_id + "-goal-hidden"
                ](seed=seed)

            env.camera_name = "corner"  # corner2, corner3, behindGripper
            # to randomize it change the freeze rand vec
            # make sure seeded env has the same goal
            env._freeze_rand_vec = freeze_rand_vec
            env.seed(seed)
            env = gym.wrappers.RecordEpisodeStatistics(env)
        elif env_name == "MAZE":
            env = gym.make(
                "PointMaze_UMazeDense-v3",
                maze_map=MAZE_MAPS[maze_map],
                continuing_task=False,
                reset_target=False,
                render_mode="rgb_array",
                max_episode_steps=max_episode_steps,
                reward_type="dense",
                camera_id=0,
            )
        elif env_name == "D4RL":
            # uses an old version of gym
            env = old_gym.make(env_id, render_mode="rgb_array").unwrapped
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.EnvCompatibility(old_env=env, render_mode="rgb_array")
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)

            # import ipdb

            # ipdb.set_trace()
            # convert spaces so i can use env batching
            env.observation_space = gym.spaces.Box(
                shape=env.observation_space.shape,
                dtype=np.float32,
                low=-np.inf,
                high=np.inf,
            )
            env.action_space = gym.spaces.Box(
                shape=env.action_space.shape, dtype=np.float32, low=-1.0, high=1.0
            )

        # print(env)
        return env

    if num_envs == 1:
        env = thunk()
    else:
        env = gym.vector.SyncVectorEnv([lambda: thunk() for _ in range(num_envs)])

        # env = gym.vector.AsyncVectorEnv([lambda: thunk() for _ in range(num_envs)])

    return env
