
import sys

from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector, GpuResetCollector
from rlpyt.envs import gym
from procgen import ProcgenEnv
from rlpyt.agents.pg.categorical import CategoricalPgAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config

from ul_gen.configs.ppo_alae_config import configs
from ul_gen.models.alae import ALAEPolicy
from ul_gen.algos.ppo_alae import PPO_ALAE


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
    else:
        model_state_dict = None

    algo = PPO_ALAE(**config["algo"])
    agent = CategoricalPgAgent(
        ModelCls=ALAEPolicy,
        model_kwargs=config["model"],
        initial_model_state_dict=model_state_dict,
        **config["agent"]
    )
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
