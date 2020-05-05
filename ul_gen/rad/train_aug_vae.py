import sys
import torch
from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector, GpuResetCollector
from rlpyt.envs import gym
from procgen import ProcgenEnv
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config

from ul_gen.rad.ppo_aug_vae import PPO_AUG_VAE
from ul_gen.rad.ppo_aug_vae_config import configs
from ul_gen.rad.aug_vae import RadVaePolicy 
from ul_gen.rad.rad_agent import RADPgVaeAgent
from ul_gen.rad.env_wrapper import make

def build_and_train(slot_affinity_code, log_dir, run_ID, config_key):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)

    eval_env_config = config["env"].copy()
    eval_env_config["start_level"] = config["env"]["num_levels"] + 100
    eval_env_config["num_levels"] = 100
    sampler = GpuSampler(
        EnvCls=make,
        env_kwargs=config["env"],
        CollectorCls=GpuResetCollector,
        eval_env_kwargs=eval_env_config,
        **config["sampler"]
    )

    if config["checkpoint"]:
        model_state_dict = torch.load(config["checkpoint"])
        print("Loaded.")
    else:
        model_state_dict = None

    algo = PPO_AUG_VAE(optim_kwargs=config["optim"], **config["algo"])
    agent = RADPgVaeAgent(ModelCls=RadVaePolicy, model_kwargs=config["model"], 
                            initial_model_state_dict=model_state_dict, **config["agent"])
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )
    name = config["env"]["id"]

    with logger_context(log_dir, run_ID, name, config, snapshot_mode='last'):
        runner.train()

if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
