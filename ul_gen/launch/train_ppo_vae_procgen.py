
import sys

from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector
from rlpyt.envs import gym
from procgen import ProcgenEnv
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config

from ul_gen.models.vae import VaePolicy
from ul_gen.agents.vae_agent import CategoricalPgVaeAgent
from ul_gen.algs.vae_ppo import PPO_VAE

from ul_gen.configs.ppo_vae_procgen_config import configs


def build_and_train(slot_affinity_code, log_dir, run_ID, config_key):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)

    sampler = GpuSampler(
        EnvCls=gym.make,
        env_kwargs=config["env"],
        CollectorCls=GpuWaitResetCollector,
        eval_env_kwargs=config["env"],
        **config["sampler"]
    )
    
    algo = PPO_VAE(optim_kwargs=config["optim"], **config["algo"])
    agent = CategoricalPgVaeAgent(ModelCls=VaePolicy, model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )
    name = config["env"]["id"]
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
