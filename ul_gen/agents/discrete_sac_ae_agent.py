
import numpy as np
import torch
from collections import namedtuple
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallelCPU as DDPC

from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.utils.quick_args import save__init__args
# from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.distributions.categorical import Categorical, DistInfo, EPS

from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.collections import namedarraytuple

from ul_gen.models.discrete_sac_ae_models import PixelEncoder, PixelDecoder, SACAEActor, SACAECritic

# MIN_LOG_STD = -20
# MAX_LOG_STD = 2

AgentInfo = namedarraytuple("AgentInfo", ["dist_info"])
Models = namedtuple("Models", ["pi", "q1", "q2", "v"])

class DiscreteSacAEAgent(BaseAgent):
    """Agent for SAC algorithm, including action-squashing, using twin Q-values."""

    def __init__(
            self,
            ModelCls=SACAEActor,  # Pi model.
            CriticCls=SACAECritic,
            EncoderCls=PixelEncoder,
            DecoderCls=PixelDecoder,
            model_kwargs={},  # Pi model.
            critic_kwargs={},
            encoder_kwargs={},
            initial_model_state_dict=None,  # All models.
            pretrain_std=0.75,  # With squash 0.75 is near uniform.
            random_actions_for_pretraining=False
            ):
        model_kwargs["EncoderCls"] = EncoderCls
        model_kwargs["encoder_kwargs"] = encoder_kwargs
        """Saves input arguments; network defaults stored within."""
        super().__init__(ModelCls=ModelCls, model_kwargs=model_kwargs,
            initial_model_state_dict=initial_model_state_dict)
        save__init__args(locals())
        self.min_itr_learn = 0  # Get from algo.

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        _initial_model_state_dict = self.initial_model_state_dict
        self.initial_model_state_dict = None  # Don't let base agent try to load.
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        self.initial_model_state_dict = _initial_model_state_dict

        self.critic = self.CriticCls(**self.env_model_kwargs, EncoderCls=self.EncoderCls, encoder_kwargs=self.encoder_kwargs, **self.critic_kwargs)
        self.target_model = self.CriticCls(**self.env_model_kwargs, EncoderCls=self.EncoderCls, encoder_kwargs=self.encoder_kwargs, **self.critic_kwargs)
        self.decoder = self.DecoderCls(**self.encoder_kwargs)
        # self.q1_model = self.QModelCls(**self.env_model_kwargs, **self.q_model_kwargs)
        # self.q2_model = self.QModelCls(**self.env_model_kwargs, **self.q_model_kwargs)
        # self.target_q1_model = self.QModelCls(**self.env_model_kwargs,
        #     **self.q_model_kwargs)
        # self.target_q2_model = self.QModelCls(**self.env_model_kwargs,
        #     **self.q_model_kwargs)
        self.target_model.load_state_dict(self.critic.state_dict())
        # Tie the Encoder of the actor to that of the critic
        self.model.encoder.copy_weights_from(self.critic.encoder)

        # self.target_q1_model.load_state_dict(self.q1_model.state_dict())
        # self.target_q2_model.load_state_dict(self.q2_model.state_dict())
        if self.initial_model_state_dict is not None:
            self.load_state_dict(self.initial_model_state_dict)
        # assert len(env_spaces.action.shape) == 1
        # self.distribution = Gaussian(
        #     dim=env_spaces.action.shape[0],
        #     squash=self.action_squash,
        #     min_std=np.exp(MIN_LOG_STD),
        #     max_std=np.exp(MAX_LOG_STD),
        # )
        self.distribution = Categorical(dim=env_spaces.action.n)

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)
        # self.q1_model.to(self.device)
        # self.q2_model.to(self.device)
        # self.target_q1_model.to(self.device)
        # self.target_q2_model.to(self.device)
        self.critic.to(self.device)
        self.target_model.to(self.device)
        self.decoder.to(self.device)

    def data_parallel(self):
        super().data_parallel
        DDP_WRAP = DDPC if self.device.type == "cpu" else DDP
        # self.q1_model = DDP_WRAP(self.q1_model)
        # self.q2_model = DDP_WRAP(self.q2_model)
        self.critic = DDP_WRAP(self.critic)

    def give_min_itr_learn(self, min_itr_learn):
        self.min_itr_learn = min_itr_learn  # From algo.

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(
            observation_shape=env_spaces.observation.shape,
            action_size=env_spaces.action.n,
        )

    def q(self, observation, prev_action, prev_reward, detach_encoder=False):
        """Compute twin Q-values for state/observation and input action 
        (with grad)."""
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        # q1 = self.q1_model(*model_inputs)
        # q2 = self.q2_model(*model_inputs)
        q1, q2, _, _ = self.critic(*model_inputs, detach_encoder=detach_encoder)
        # print("critic device", self.critic.q1[0].weight.device)
        return q1.cpu(), q2.cpu()

    def target_q(self, observation, prev_action, prev_reward, detach_encoder=False):
        """Compute twin target Q-values for state/observation and input
        action.""" 
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        # target_q1 =self.target_q1_model(*model_inputs)
        # target_q2 = self.target_q2_model(*model_inputs)
        target_q1, target_q2, _, _ = self.target_model(*model_inputs, detach_encoder=detach_encoder)
        return target_q1.cpu(), target_q2.cpu()

    def pi(self, observation, prev_action, prev_reward, detach_encoder=False):
        """Compute action log-probabilities for state/observation, and
        sample new action (with grad).  Uses special ``sample_loglikelihood()``
        method of Gaussian distriution, which handles action squashing
        through this process."""
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        # mean, log_std = self.model(*model_inputs)
        # dist_info = DistInfoStd(mean=mean, log_std=log_std)
        # action, log_pi = self.distribution.sample_loglikelihood(dist_info)
        pi, _, _ = self.model(*model_inputs)
        dist_info = DistInfo(prob=pi)
        action = self.distribution.sample(dist_info)
        log_pi = torch.log(pi)
        
        # TODO: potentially use argmax to determine the action instead of sampling from the distribution.
        # TODO: Figure out what to do for log_pi

        # action = self.distribution.sample(dist_info)
        # log_pi = self.distribution.log_likelihood(action, dist_info)
        log_pi, dist_info = buffer_to((log_pi, dist_info), device="cpu")
        return action, log_pi, dist_info  # Action stays on device for q models.

    def z(self, observation, prev_action, prev_reward, detach_encoder=False):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _, _, mu, log_sd = self.critic(*model_inputs, detach_encoder=detach_encoder)
        if not self.critic.rae:
            # Reparameterize
            sd = torch.exp(log_sd)
            eps = torch.randn_like(sd)
            z = mu + eps*sd
        else:
            z = mu
        return z, mu, log_sd

    def decode(self, z):
        z, = buffer_to((z,), device=self.device)
        return self.decoder(z)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        # mean, log_std = self.model(*model_inputs)
        # dist_info = DistInfoStd(mean=mean, log_std=log_std)
        # action = self.distribution.sample(dist_info)
        if self.random_actions_for_pretraining:
            action = torch.randint_like(prev_action, 15)
            action = buffer_to(action, device="cpu")
            return AgentStep(action=action, agent_info=AgentInfo(dist_info=None))

        pi, _, _ = self.model(*model_inputs)
        dist_info = DistInfo(prob=pi)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfo(dist_info=dist_info)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def update_target(self, tau=1):
        update_state_dict(self.target_model, self.critic.state_dict(), tau)

    def rae(self):
        return self.critic.rae

    @property
    def models(self):
        return Models(model=self.model, critic=self.critic)

    def actor_parameters(self):
        return self.model.parameters()

    def critic_parameters(self):
        return self.critic.parameters()

    def decoder_parameters(self):
        return self.decoder.parameters()

    def encoder_parameters(self):
        return self.critic.encoder.parameters()

    def train_mode(self, itr):
        super().train_mode(itr)
        self.critic.train()

    def sample_mode(self, itr):
        super().sample_mode(itr)
        self.critic.eval()
        if itr == 0:
            logger.log(f"Agent at itr {itr}, sample std: {self.pretrain_std}")
        if itr == self.min_itr_learn:
            logger.log(f"Agent at itr {itr}, sample std: learned.")
        # std = None if itr >= self.min_itr_learn else self.pretrain_std
        # self.distribution.set_std(std)  # If None: std from policy dist_info.

    def eval_mode(self, itr):
        super().eval_mode(itr)
        self.critic.eval()
        # self.distribution.set_std(0.)  # Deterministic (dist_info std ignored).

    def state_dict(self):
        return dict(
            model=self.model.state_dict(),  # Pi model.
            critic=self.critic.state_dict(),
            target_model=self.target_model.state_dict(),
            decoder=self.decoder.state_dict()
        )

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_model.load_state_dict(state_dict["target_model"])
        self.decoder.load_state_dict(state_dict["decoder"])
      
