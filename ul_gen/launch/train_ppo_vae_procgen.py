import json
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
from rlpyt.utils.logging import logger

from ul_gen.configs.ppo_vae_procgen_config import configs
from ul_gen.algos.ppo_vae import PPO_VAE
from ul_gen.models.vae import VaePolicy
from ul_gen.agents.vae_agent import CategoricalPgVaeAgent


class MinibatchRlEvalVAE(MinibatchRlEval):

    def log_diagnostics(self, itr, eval_traj_infos, eval_time, prefix='Diagnostics/'):
        logger.log("INFO: Saving VAE Samples.")
        self.agent.model.log_samples(logger.get_snapshot_dir(), itr)
        super().log_diagnostics(itr, eval_traj_infos, eval_time, prefix=prefix)


def build_and_train(slot_affinity_code, log_dir, run_ID, config_key):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)

    sampler = GpuSampler(
        EnvCls=gym.make,
        env_kwargs=config["env"],
        CollectorCls=GpuResetCollector,
        eval_env_kwargs=config["eval_env"],
        **config["sampler"]
    )
    if config["checkpoint"]:
        model_state_dict = torch.load(config["checkpoint"])
        print("Loaded.")

    else:
        model_state_dict = None

    algo = PPO_VAE(optim_kwargs=config["optim"], **config["algo"])
    agent = CategoricalPgVaeAgent(
        ModelCls=VaePolicy,
        model_kwargs=config["model"], 
        initial_model_state_dict=model_state_dict,
        **config["agent"]
    )
    runner = MinibatchRlEvalVAE(
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
