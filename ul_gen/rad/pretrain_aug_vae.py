import torch
import os
from torch import optim
from rlpyt.utils.launching.affinity import encode_affinity, affinity_from_code, prepend_run_slot
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector, GpuResetCollector
from rlpyt.agents.base import BaseAgent, AgentStep, AgentInputs
from rlpyt.utils.seed import set_seed, make_seed
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

from torchvision.utils import save_image

from ul_gen.rad.ppo_aug_vae_config import configs
from ul_gen.rad.rad_agent import RADPgVaeAgent
from ul_gen.rad.aug_vae import RadVaePolicy
from ul_gen.rad.env_wrapper import make

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--savepath",type=str,default="./raug_vae_data/")

args = parser.parse_args()

os.makedirs(args.savepath, exist_ok=True)

EmptyAgentInfo = namedarraytuple("EmptyAgentInfo", [])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

affinity_code = encode_affinity(
    n_cpu_core=4,
    n_gpu=0,
    n_socket=1,
)

affinity = affinity_from_code(prepend_run_slot(0, affinity_code))
# Get Params
config = configs["ppo_vae"]

# Setup the data collection pipeline
# Edit the sampler kwargs to get a larger batch size
config["sampler"]["batch_T"] = 24
config["sampler"]["batch_B"] = 16

config["algo"]["learning_rate"] = 1e-4

sampler = GpuSampler(
        EnvCls=make,
        env_kwargs=config["env"],
        CollectorCls=GpuResetCollector,
        eval_env_kwargs=config["env"],
        **config["sampler"]
    )
agent = RADPgVaeAgent(ModelCls=RadVaePolicy, model_kwargs=config["model"], **config["agent"])
seed = make_seed()
set_seed(seed)
sampler.initialize(agent=agent, affinity=affinity, seed=seed+1,rank=0)
agent.to_device(affinity.get("cuda_idx", None))

optimizer = optim.Adam(agent.model.parameters(), lr=config["algo"]["learning_rate"], **config["optim"])

# Train
for itr in range(10000):
    samples, _ = sampler.obtain_samples(itr)
    agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation.view(-1, 3, 64, 64),
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
    agent_inputs = buffer_to(agent_inputs, device=device)
    
    optimizer.zero_grad()
    _, _, _, _, loss = agent(*agent_inputs)
    loss.backward()
    optimizer.step()

    if (itr + 1) % 100 == 0:
        print("Iteration", itr+1, "Loss", loss.item())

    if (itr + 1) % 1000 == 0:
        print("Saving.")
        # Save reconstructions

        x_hat = agent.reconstructions(*agent_inputs)
        x = agent_inputs.observation.detach().cpu().float() / 255.

        reconstructions = torch.cat((x[:8], x_hat[:8]),dim=0) 
        save_image(reconstructions, os.path.join(args.savepath, 'recon_' + str(itr+1) +'.png'), nrow=8)
        torch.save(agent.model.state_dict(), '%s/ae-aug_vae_data-%d' % (args.savepath, int(itr+1)))

print("Training complete.")
sampler.shutdown()
