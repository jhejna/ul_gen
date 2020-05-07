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
    def __init__(self, ModelCls=None, model_kwargs=None, initial_model_state_dict=None, data_augs="", both_actions=False):
        super().__init__(ModelCls=ModelCls, model_kwargs=model_kwargs, initial_model_state_dict=initial_model_state_dict)
        self.data_augs = data_augs
        self.both_actions = both_actions

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
        augmented = self.aug_obs(observation)

        observation = observation.type(torch.float)  # Expect torch.uint8 inputs
        observation = observation.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        augmented, observation, prev_action, prev_reward = buffer_to((augmented, observation, prev_action, prev_reward),
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
        aug_pi, aug_value = self.model(augmented, prev_action, prev_reward)
        if self.both_actions:
            pi, value = self.model(observation, prev_action, prev_reward)
            return buffer_to((DistInfo(prob=aug_pi), DistInfo(prob=pi), aug_value, value), device="cpu")
        return buffer_to((DistInfo(prob=aug_pi), aug_value), device="cpu")

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

from rlpyt.utils.collections import namedarraytuple
AgentInfoVae = namedarraytuple("AgentInfoVae", ["dist_info", "value", "latent", "reconstruction"])

class RADPgVaeAgent(BaseAgent):
    """
    Agent for policy gradient algorithm using categorical action distribution.
    Same as ``GausssianPgAgent`` and related classes, except uses
    ``Categorical`` distribution, and has a different interface to the model
    (model here outputs discrete probabilities in place of means and log_stds,
    while both output the value estimate).
    """
    def __init__(self, ModelCls=None, model_kwargs=None, initial_model_state_dict=None, data_augs="",
                vae_loss_type="l2",
                vae_beta=1.0,
                sim_loss_coef=0.1,
                k_dim=24
                ):
        super().__init__(ModelCls=ModelCls, model_kwargs=model_kwargs, initial_model_state_dict=initial_model_state_dict)
        self.data_augs = data_augs
        self.vae_loss_type = vae_loss_type
        self.vae_beta = vae_beta
        self.sim_loss_coef = sim_loss_coef
        self.k_dim = k_dim

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
        observation_one = self.aug_obs(observation)
        observation_two = self.aug_obs(observation.detach().clone())

        observation_one, observation_two, prev_action, prev_reward = buffer_to((observation_one, observation_two, prev_action, prev_reward),
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

        pi_one, value_one, latent_one, reconstruction_one = self.model(observation_one, prev_action, prev_reward)
        pi_two, value_two, latent_two, reconstruction_two = self.model(observation_two, prev_action, prev_reward)
        bs = 2*len(observation)

        if self.vae_loss_type == "l2":
            recon_loss = (torch.sum((observation_one - reconstruction_one).pow(2)) +
                            torch.sum((observation_two - reconstruction_two).pow(2))) / bs
        elif self.vae_loss_type == "bce":
            recon_loss = (torch.nn.functional.binary_cross_entropy(reconstruction, obs) +
                        torch.nn.functional.binary_cross_entropy(reconstruction, obs)) / 2
        
        # Calculate the similarity loss
        mu_one, logsd_one = torch.chunk(latent_one, 2, dim=1)
        mu_two, logsd_two = torch.chunk(latent_two, 2, dim=1)

        latent_loss_one = torch.sum(-0.5*(1 + (2*logsd_one) - mu_one.pow(2) - (2*logsd_one).exp()))
        latent_loss_two = torch.sum(-0.5*(1 + (2*logsd_two) - mu_two.pow(2) - (2*logsd_two).exp()))
        latent_loss = (latent_loss_one + latent_loss_two) / bs

        mu_one, mu_two = mu_one[:, :self.k_dim], mu_two[:, :self.k_dim]
        logvar_one, logvar_two = 2*logsd_one[:, :self.k_dim], 2*logsd_two[:, :self.k_dim]
        # KL divergence between original and augmented.
        sim_loss = torch.sum(logvar_two - logvar_one + 0.5*(logvar_one.exp() + (mu_one - mu_two).pow(2))/logvar_two.exp() - 0.5)/ (bs // 2)
        
        vae_loss = recon_loss + self.vae_beta * latent_loss + self.sim_loss_coef * sim_loss

        # Average estimates for better power
        value_est = (value_one + value_two) / 2
        pi_est = (pi_one + pi_two) / 2

        return buffer_to((DistInfo(prob=pi_est), value_est, vae_loss), device="cpu")

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
        pi, value, latent, reconstruction = self.model(*model_inputs)
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
        _pi, value, _latent, _reconstruction = self.model(*model_inputs)
        return value.to("cpu")

    @torch.no_grad()
    def reconstructions(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        observation = observation.type(torch.float)
        observation = observation.mul_(1. / 255)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _pi, _value, _latent, reconstruction = self.model(*model_inputs)
        return reconstruction.to("cpu")
