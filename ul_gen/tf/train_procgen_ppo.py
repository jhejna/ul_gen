import os
from datetime import datetime
import json
import gym
import numpy as np
import tensorflow as tf
from stable_baselines import logger
from stable_baselines.bench import Monitor

from stable_baselines import PPO2
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import FeedForwardPolicy, CnnPolicy
from stable_baselines.common.callbacks import BaseCallback

from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv

DATA_DIR = 'data'

def make_env(env_name, save_path, rank, seed=None, **kwargs):
        env_key = "procgen:procgen-{0}-v0".format(env_name)
        mon_path = save_path + "/" + str(rank) 
        def _init():
            assert isinstance(mon_path, str)
            env = gym.make(env_key, **kwargs)
            if seed:
                env.seed(seed + rank)
            env = Monitor(env, mon_path)
            return env
        if seed:
            set_global_seeds(seed)
        return _init

def impala_cnn(unscaled_images, depths=[16,32,32], **kwargs):
    """
    Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
    Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
    """
    layer_num = 0

    def get_layer_num_str():
        nonlocal layer_num
        num_str = str(layer_num)
        layer_num += 1
        return num_str

    def conv_layer(out, depth):
        return tf.layers.conv2d(out, depth, 3, padding='same', name='layer_' + get_layer_num_str())

    def residual_block(inputs):
        depth = inputs.get_shape()[-1].value
        out = tf.nn.relu(inputs)
        out = conv_layer(out, depth)
        out = tf.nn.relu(out)
        out = conv_layer(out, depth)
        return out + inputs

    def conv_sequence(inputs, depth):
        out = conv_layer(inputs, depth)
        out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
        out = residual_block(out)
        out = residual_block(out)
        return out

    out = tf.cast(unscaled_images, tf.float32) / 255.

    for depth in depths:
        out = conv_sequence(out, depth)

    out = tf.layers.flatten(out)
    out = tf.nn.relu(out)
    out = tf.layers.dense(out, 256, activation=tf.nn.relu, name='layer_' + get_layer_num_str())

    return out


class ImpalaPolicy(FeedForwardPolicy):
    # Citation: https://arxiv.org/abs/1802.01561
    def __init__(self, *args, **kwargs):
        super(ImpalaPolicy, self).__init__(*args, **kwargs,
                                           feature_extraction="cnn",
                                           cnn_extractor=impala_cnn,
                                           )

class TrainCallback(BaseCallback):
    """
    Saves model on best 100 ep reward
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(TrainCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)
        return True


def main():
    # PPO Params
    params = {
        "env" : 'coinrun',
        "seed" : None,
        "num_proc": 8,
        "normalize" : True,
        "timesteps": 100000,
        "log_interval": 20,
        "eval_freq" : 10000,
        "alg_args" : {
            "learning_rate" : 5e-4,
            "gamma" : 0.999,
            "lam": 0.95,
            "ent_coef": 0.01,
            "nminibatches": 8,
            "noptepochs": 3,
            "cliprange": 0.2,
            "cliprange_vf": 0.2,
        },
        "policy" : "impala",
        "policy_args" : {
            "depths" : [16,32,32],
            "scale": False
        },
        "env_args" : {
            "num_levels" : 500,
            "start_level": 0,
            "distribution_mode" : "easy"
        },
    }
    # Get the save path
    save_path = os.path.join(DATA_DIR, datetime.now().strftime("%m_%d_%y-%H:%M:%S"))
    os.makedirs(save_path, exist_ok=True)
    # Save the model parameters
    with open(os.path.join(save_path, "params.json"), 'w') as fp:
        json.dump(params, fp, indent=4)

    # Create MultiProcessing Env
    env = SubprocVecEnv([make_env(params["env"], save_path, i, seed=params['seed'], **params['env_args']) for i in range(params['num_proc'])])
    if params['normalize']:
        env = VecNormalize(env)

    # Determine the policy
    policy = {
        "cnn" : CnnPolicy,
        "impala" : ImpalaPolicy
    }[params['policy']]

    model = PPO2(policy, env, verbose=1, policy_kwargs=params['policy_args'], **params['alg_args'])
    callback = TrainCallback(params['eval_freq'], save_path)
    model.learn(total_timesteps=params['timesteps'], log_interval=params['log_interval'], callback=callback)

    model.save(save_path + "/final_model")
    if params['normalize']:
        # Save normalization params for inference
        env.save("env")

if __name__ == '__main__':
    main()

