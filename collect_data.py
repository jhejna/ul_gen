from procgen import ProcgenEnv
import gym

# env = ProcgenEnv(num_envs=1, env_name="coinrun")
save_path = '/home/karam/Downloads/procgen.npy'
data = []
env = gym.make("procgen:procgen-coinrun-v0")
obs = env.reset()
for _ in range(1000):
	env.render()
	ac = env.action_space.sample()
	obs, reward, done, info = env.step(ac)
	print(type(obs))
	data.append([obs, [reward, done, info]])
	

np.save(save_path, data)
