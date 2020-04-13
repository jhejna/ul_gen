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

from ul_gen.configs.ppo_vae_procgen_config import configs
from ul_gen.models.vae import VaePolicy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--savepath",type=str,default="./vae_data/")
args = parser.parse_args()

os.makedirs(args.savepath, exist_ok=True)

EmptyAgentInfo = namedarraytuple("EmptyAgentInfo", [])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

affinity_code = encode_affinity(
    n_cpu_core=4,
    n_gpu=1,
    # hyperthread_offset=20,
    n_socket=2
    # cpu_per_run=2,
)
affinity = affinity_from_code(prepend_run_slot(0, affinity_code))
# Get Params
config = configs["pretrain"]

# Setup Data Collection
# The below are util classes

class RandomAgent(BaseAgent):

    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        action = self.model(*model_inputs)
        agent_info = EmptyAgentInfo()
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

class RandomDiscreteModel(torch.nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
    
    def forward(self, observation, prev_action, prev_reward):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        action =  torch.randint(low=0, high=self.num_actions, size=(T*B,))
        action = restore_leading_dims((action), lead_dim, T, B)
        return action

# Setup the data collection pipeline
sampler = GpuSampler(
        EnvCls=gym.make,
        env_kwargs=config["env"],
        CollectorCls=GpuResetCollector,
        eval_env_kwargs=config["env"],
        **config["sampler"]
    )
agent = RandomAgent(ModelCls=RandomDiscreteModel, model_kwargs={"num_actions": 15})
seed = make_seed()
set_seed(seed)
sampler.initialize(agent=agent, affinity=affinity, seed=seed+1,rank=0)

# Create the model
model = VaePolicy(**config["model"]).to(device)
# Setup the optimizers
opt = optim.Adam(model.parameters(), **config["optim"])

# Train
for itr in range(config["train_steps"]):
    samples, _ = sampler.obtain_samples(itr)
    inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
    inputs = buffer_to(inputs, device=device)[:,:]
    
    opt.zero_grad()

    recon_loss, kl_loss = model.loss(config["algo"]["loss"],inputs)
    loss = recon_loss + config["algo"]["vae_beta"] * kl_loss

    loss.backward()
    opt.step()

    if (itr + 1) % config["log_freq"] == 0:
        print("Iteration", itr+1, "- Reconstruction:", recon_loss.item(), "KL:", kl_loss.item()) 
    if (itr + 1) % config["eval_freq"] == 0:
        print("Iteration", itr+1, "Evaluating.")
        model.log_images(inputs.observation, args.savepath, itr) 

torch.save(model.state_dict(), '%s/vae-final' % (args.savepath))
print("Training complete.")
sampler.shutdown()
