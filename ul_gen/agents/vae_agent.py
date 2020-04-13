import torch

from rlpyt.agents.base import AgentStep, BaseAgent
from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method
from rlpyt.utils.collections import namedarraytuple

AgentInfoVae = namedarraytuple("AgentInfoVae", ["dist_info", "value", "latent", "reconstruction"])

class CategoricalPgVaeAgent(BaseAgent):
    """
    Agent for policy gradient algorithm using categorical action distribution.
    Same as ``GausssianPgAgent`` and related classes, except uses
    ``Categorical`` distribution, and has a different interface to the model
    (model here outputs discrete probabilities in place of means and log_stds,
    while both output the value estimate).
    """

    def __call__(self, observation, prev_action, prev_reward):

        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        pi, value, latent, reconstruction = self.model(*model_inputs)
        return buffer_to((DistInfo(prob=pi), value, latent, reconstruction), device="cpu")

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        self.distribution = Categorical(dim=env_spaces.action.n)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        pi, value, latent, reconstruction = self.model(*model_inputs)
        # print(pi)
        # print(latent)
        # print()
        dist_info = DistInfo(prob=pi)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfoVae(dist_info=dist_info, value=value, latent=latent, reconstruction=reconstruction)

        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _pi, value, _latent, _reconstruction = self.model(*model_inputs)
        return value.to("cpu")

    @torch.no_grad()
    def reconstruction(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _pi, _value, _latent, reconstruction = self.model(*model_inputs)
        return reconstruction.to("cpu")

    @torch.no_grad()
    def latent(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _pi, _value, latent, _reconstruction = self.model(*model_inputs)
        return latent.to("cpu")
