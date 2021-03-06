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
    def __init__(self,ModelCls,override,
        model_kwargs, initial_model_state_dict, **agent_config):
        super().__init__(ModelCls,model_kwargs,initial_model_state_dict,**agent_config)
        self.override=override

    def __call__(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        pi, value, latent, reconstruction, idx = self.model(*model_inputs)
        return buffer_to((DistInfo(prob=pi), value, latent, reconstruction), device="cpu"), idx

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        if self.override['override_policy_value']:
            policy_layers=self.override["policy_layers"]
            value_layers=self.override["value_layers"]
            self.model.override_policy_value(policy_layers=policy_layers,
                value_layers=value_layers)
        self.distribution = Categorical(dim=env_spaces.action.n)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        pi, value, latent, reconstruction, idx = self.model(*model_inputs)
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
        _pi, value, _latent, _reconstruction, _idx = self.model(*model_inputs)
        return value.to("cpu")

    @torch.no_grad()
    def reconstruction(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _pi, _value, _latent, reconstruction, _idx = self.model(*model_inputs)
        return reconstruction.to("cpu")

    @torch.no_grad()
    def latent(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _pi, _value, latent, _reconstruction, _idx = self.model(*model_inputs)
        return latent.to("cpu")

    @torch.no_grad()
    def idx(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _pi, _value, latent, _reconstruction, idx = self.model(*model_inputs)
        return idx
