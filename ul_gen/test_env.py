import gym
import numpy as np

# env = ProcgenEnv(num_envs=1, env_name="coinrun")
save_path = '/home/karam/Downloads/procgen.npy'
n_resets = 2
traj = 1000
all_data = []

env = gym.make("procgen:procgen-coinrun-v0", start_level=0, num_levels=500, distribution_mode="hard")
obs = env.reset()
for _ in range(10000):
    ac = env.action_space.sample()
    obs, reward, done, info = env.step(ac)
    env.render()
    if done:
        obs = env.reset()

