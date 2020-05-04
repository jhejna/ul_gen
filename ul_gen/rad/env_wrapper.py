import gym
from gym import ObservationWrapper
import numpy as np
from rlpyt.envs.gym import GymEnvWrapper


class PytImgWrapper(ObservationWrapper):
    def __init__(self, env):
        super(PytImgWrapper, self).__init__(env)

    def observation(self, observation):
        return np.rollaxis(observation, 2, 0)  

def make(*args, **kwargs):
    # Make the RLPYT Environment after wrapping the gym environment
    env = gym.make(*args, **kwargs)
    env = PytImgWrapper(env)
    env = GymEnvWrapper(env)
    return env


if __name__ == "__main__":
    env = make("procgen:procgen-coinrun-v0")

    obs = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(obs.shape)
        env.render()
        if done:
            obs = env.reset()
