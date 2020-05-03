import torch

from rlpyt.agents.base import (AgentStep, BaseAgent, RecurrentAgentMixin,
    AlternatingRecurrentAgentMixin)
from rlpyt.agents.pg.base import AgentInfo, AgentInfoRnn
from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.utils.buffer import buffer_to, buffer_func, buffer_method

from collections import OrderedDict
from ul_gen.rad import data_aug as rad

class RADPgAgent(BaseAgent):
    """
    Agent for policy gradient algorithm using categorical action distribution.
    Same as ``GausssianPgAgent`` and related classes, except uses
    ``Categorical`` distribution, and has a different interface to the model
    (model here outputs discrete probabilities in place of means and log_stds,
    while both output the value estimate).
    """
    def __init__(self, ModelCls=None, model_kwargs=None, initial_model_state_dict=None, data_augs=""):
        super().__init__(ModelCls=ModelCls, model_kwargs=model_kwargs, initial_model_state_dict=initial_model_state_dict)
        self.data_augs = data_augs

    def aug_obs(self, observation):
        # Apply initial augmentations
        for aug, func in self.augs_funcs.items():
            if 'cutout' in aug:
                observation = func(observation)
        
        observation = observation.type(torch.float)  # Expect torch.uint8 inputs
        observation = observation.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        for aug, func in self.augs_funcs.items():
            if 'cutout' in aug:
                continue
            else:
                observation = func(observation)
        return observation

    def __call__(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)

        # This is what needs to modified to apply the augmentation from the data.
        assert len(observation.shape) == 4, "Observation shape was not length 4"
        observation = self.aug_obs(observation)

        observation, prev_action, prev_reward = buffer_to((observation, prev_action, prev_reward),
            device=self.device)

        # For visualizing the observations
        # import matplotlib.pyplot as plt
        # from torchvision.utils import make_grid
        # def show_imgs(x,max_display=16):
        #     grid = make_grid(x[:max_display],4).permute(1,2,0).cpu().numpy()
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.imshow(grid)
        #     plt.show()
        # show_imgs(orig_observation)
        # show_imgs(observation)

        pi, value = self.model(observation, prev_action, prev_reward)
        return buffer_to((DistInfo(prob=pi), value), device="cpu")

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        self.distribution = Categorical(dim=env_spaces.action.n)

        self.augs_funcs = OrderedDict()
        aug_to_func = {
                'crop':rad.random_crop,
                'crop_horiz': rad.random_crop_horizontile,
                'grayscale':rad.random_grayscale,
                'cutout':rad.random_cutout,
                'cutout_color':rad.random_cutout_color,
                'flip':rad.random_flip,
                'rotate':rad.random_rotation,
                'rand_conv':rad.random_convolution,
                'color_jitter':rad.random_color_jitter,
                'no_aug': rad.no_aug,
            }

        if self.data_augs == "":
            aug_names = []
        else:
            aug_names = self.data_augs.split('-')
        for aug_name in aug_names:
            assert aug_name in aug_to_func, 'invalid data aug string'
            self.augs_funcs[aug_name] = aug_to_func[aug_name]

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        observation = observation.type(torch.float)  # Expect torch.uint8 inputs
        observation = observation.mul_(1. / 255)  # From [0-255] to [0-1], in place.
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        pi, value = self.model(*model_inputs)
        dist_info = DistInfo(prob=pi)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfo(dist_info=dist_info, value=value)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        observation = observation.type(torch.float)  # Expect torch.uint8 inputs
        observation = observation.mul_(1. / 255)  # From [0-255] to [0-1], in place.
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _pi, value = self.model(*model_inputs)
        return value.to("cpu")
