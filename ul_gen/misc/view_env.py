from procgen import ProcgenEnv
import gym
from gym import wrappers
import numpy as np
from IPython import display
import matplotlib.pyplot as plt
import os
import cv2

name = "procgen:procgen-coinrun-v0"
env = gym.make(name)
env = wrappers.Monitor(env, f'/home/ashwin/ul_gen/ul_gen/data/{name}/', force=True)
obs = env.reset()

img = plt.imshow(obs) # only call this once

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    cv2.imshow("Observation", obs)
    # img.set_data(obs) # just update the data
    # display.display(plt.gcf())
    # display.clear_output(wait=True)

plt.show()
env.close()