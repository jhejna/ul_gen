import torch
import os
from torch import optim
from rlpyt.utils.launching.affinity import encode_affinity, affinity_from_code, prepend_run_slot
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector, GpuResetCollector
from rlpyt.agents.base import BaseAgent, AgentStep, AgentInputs
from rlpyt.envs import gym
from rlpyt.utils.seed import set_seed, make_seed
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

from ul_gen.algos.discrete_sac_ae import DiscreteSACAE
from ul_gen.configs.discrete_sac_ae_config import configs
from ul_gen.agents.discrete_sac_ae_agent import DiscreteSacAEAgent

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--savepath",type=str,default="./ae_data/")

args = parser.parse_args()

os.makedirs(args.savepath, exist_ok=True)

EmptyAgentInfo = namedarraytuple("EmptyAgentInfo", [])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

affinity_code = encode_affinity(
    n_cpu_core=4,
    n_gpu=1,
    n_socket=1,
)

affinity = affinity_from_code(prepend_run_slot(0, affinity_code))
# Get Params
config = configs["discrete_sac_ae"]

# Setup the data collection pipeline
# Edit the sampler kwargs to get a larger batch size
config["sampler"]["batch_T"] = 24
config["sampler"]["batch_B"] = 16

sampler = GpuSampler(
        EnvCls=gym.make,
        env_kwargs=config["env"],
        CollectorCls=GpuResetCollector,
        eval_env_kwargs=config["env"],
        **config["sampler"]
    )
agent = DiscreteSacAEAgent(**config["agent"], encoder_kwargs=config["encoder"], model_kwargs=config["actor"], critic_kwargs=config["critic"],
                            random_actions_for_pretraining=True)
seed = make_seed()
set_seed(seed)
sampler.initialize(agent=agent, affinity=affinity, seed=seed+1,rank=0)
agent.to(affinity.get("cuda_idx", None))

encoder_optimizer = optim.Adam(agent.encoder_parameters(), lr=config["algo"]["ae_learning_rate"], **config["ae_optim"])
decoder_optimizer = optim.Adam(agent.decoder_parameters(), lr=config["algo"]["ae_learning_rate"], **config["ae_optim"])

# Train
for itr in range(config["pretrain_steps"]):
    samples, _ = sampler.obtain_samples(itr)
    agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
    agent_inputs = buffer_to(agent_inputs, device=device)
    
    z, mu, log_sd = agent.z(*agent_inputs)
    x_hat = agent.decode(z)
    x = agent_inputs.observation.permute(0, 1 , 4, 2, 3).float() / 255.

    batch_size = x.shape[0] * x.shape[1]
    recon_loss = torch.sum( (x_hat - x).pow(2) ) / batch_size
    if agent.rae:
        latent_loss = (0.5 * z.pow(2)).sum() / batch_size
    else:
        log_var = 2*log_sd
        latent_loss = torch.sum(-0.5*(1 + log_var - mu.pow(2) - log_var.exp())) / bs
    
    loss = recon_loss + config["algo"]["ae_beta"] * latent_loss

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    if (itr + 1) % config["runner"]["log_interval_steps"] == 0:
        print("Iteration", itr+1, "- Reconstruction:", recon_loss.item(), "Latent:", latent.item())
        # Save reconstructions
        reconstructions = torch.cat((x[:8], x_hat[:8]),dim=0) 
        reconstructions = (recon + 1)/2
        save_image(reconstructions.detach().cpu(), os.path.join(args.savepath, 'recon_' + str(itr) +'.png'), nrow=8)
        
torch.save(model.state_dict(), '%s/ae-final' % (args.savepath))
print("Training complete.")
sampler.shutdown()
