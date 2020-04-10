from procgen import ProcgenEnv
import gym
import numpy as np

# env = ProcgenEnv(num_envs=1, env_name="coinrun")
save_path = '/home/karam/Downloads/procgen.npy'
n_resets = 200
traj = 1000
all_data = []

for _ in range(n_resets):
	env = gym.make("procgen:procgen-coinrun-v0")
	obs = env.reset()

	for _ in range(traj):
		ac = env.action_space.sample()
		obs, reward, done, info = env.step(ac)
		label = {'reward': reward, 'done': done, 'info': info}
		all_data.append((obs.transpose(2,0,1), label))
		# env.render()
		if done:
			break

np.save(save_path, all_data)
