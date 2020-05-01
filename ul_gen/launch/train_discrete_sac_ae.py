
import sys

from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector, GpuResetCollector

from rlpyt.envs import gym
from procgen import ProcgenEnv
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config

from ul_gen.algos.discrete_sac_ae import DiscreteSACAE
from ul_gen.configs.discrete_sac_ae_config import configs
from ul_gen.agents.discrete_sac_ae_agent import DiscreteSacAEAgent

def build_and_train(slot_affinity_code, log_dir, run_ID, config_key):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)
    
    sampler = GpuSampler(
        EnvCls=gym.make,
        env_kwargs=config["env"],
        eval_env_kwargs=config["eval_env"],
        **config["sampler"]
    )
    algo = DiscreteSACAE(optim_kwargs=config["optim"], ae_optim_kwargs=config["ae_optim"], **config["algo"])
    agent = DiscreteSacAEAgent(**config["agent"], encoder_kwargs=config["encoder"], model_kwargs=config["actor"], critic_kwargs=config["critic"])
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
