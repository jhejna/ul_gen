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

from ul_gen.configs.ppo_bigan_procgen_config import configs
from ul_gen.models.bigan import BiGAN
import json
import argparse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--savepath",type=str,default="./vae_data/experiment/")
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
with open('%s/params.json' % args.savepath, 'w') as fp:
    json.dump(config, fp, indent=4, sort_keys=True)
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
steps = config["train_steps"]

# Create the model
model = BiGAN(**config["model"])
if config["load_path"]:
    model.load_state_dict(torch.load(config["load_path"]))
# Setup the optimizers
lr=config["optim"]["lr"]
d_optimizer = torch.optim.Adam(model.d.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=2.5e-5)
g_optimizer = torch.optim.Adam(list(model.e.parameters()) + list(model.g.parameters()), lr=lr, betas=(0.5, 0.999), weight_decay=2.5e-5)
g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer,lambda epoch: (steps - epoch) / steps,
                                                     last_epoch=-1)
d_optimizer = torch.optim.Adam(model.d.parameters(), lr=lr, betas=(0., 0.9), weight_decay=2.5e-5)
d_scheduler = torch.optim.lr_scheduler.LambdaLR(d_optimizer,lambda epoch: (steps - epoch) / steps,
                                                             last_epoch=-1)
# Train
for itr in range(steps):
    samples, _ = sampler.obtain_samples(itr)
    inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
    inputs = buffer_to(inputs, device=device)[:,:]
    
    # discriminator only on even
    if itr % 2 == 0:
        d_optimizer.zero_grad()
        d_loss = model.loss(inputs.observation, d_loss=True)
        d_loss.backward()
        d_optimizer.step()

    # generator and encoder update
    g_optimizer.zero_grad()
    g_loss = model.loss(inputs.observation, d_loss=False)
    g_loss.backward()
    g_optimizer.step()

    g_scheduler.step()
    d_scheduler.step()

    if (itr + 1) % config["log_freq"] == 0:
        print("Iteration", itr+1, "D_loss:", d_loss.item(), "G_loss:", g_loss.item()) 
    if (itr + 1) % config["eval_freq"] == 0:
        print("Iteration", itr+1, "Evaluating.")
        model.save_images(args.savepath, inputs.observation, itr) 
        model.save_models(args.savepath,itr)

model.save_models('%s/vae-final'%args.savepath)
print("Training complete.")
sampler.shutdown()
