from procgen import ProcgenEnv
import gym

# env = ProcgenEnv(num_envs=1, env_name="coinrun")
env = gym.make("procgen:procgen-coinrun-v0")
obs = env.reset()
for _ in range(1000):
	env.render()
	ac = env.action_space.sample()
	# print(ac)
	obs, reward, done, info = env.step(ac)
	print(obs.shape)
	print(type(obs))
	break
	if done:
		obs = env.reset()
