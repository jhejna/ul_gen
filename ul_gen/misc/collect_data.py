from procgen import ProcgenEnv
import gym
import numpy as np
import os

# env = ProcgenEnv(num_envs=1, env_name="coinrun")
save_path = None
save_dir = 'data/'
n_resets = 10
traj = 100
all_data = []

for i in range(n_resets):
	env = gym.make("procgen:procgen-coinrun-v0")
	obs = env.reset()
	traj_folder = os.path.join(save_dir, f'traj_{i}')
	if not os.path.exists(traj_folder):
		os.makedirs(traj_folder)

	for j in range(traj):
		ac = env.action_space.sample()
		obs, reward, done, info = env.step(ac)
		label = {'reward': reward, 'done': done, 'info': info}
		all_data.append((obs.transpose(2,0,1), label))
		# env.render()

		obs_file = os.path.join(traj_folder, f"obs_{j}.png")
		plt.imsave(obs_file, obs)

		if done:
			break

np.save(save_path, all_data)
