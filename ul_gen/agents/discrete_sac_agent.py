
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

from ul_gen.models.discrete_sac_models import DiscreteMlp, DiscreteQModel

# MIN_LOG_STD = -20
# MAX_LOG_STD = 2

AgentInfo = namedarraytuple("AgentInfo", ["dist_info"])
Models = namedtuple("Models", ["pi", "q1", "q2", "v"])

class DiscreteSacAgent(BaseAgent):
    """Agent for SAC algorithm, including action-squashing, using twin Q-values."""

    def __init__(
            self,
            ModelCls=DiscreteMlp,  # Pi model.
            QModelCls=DiscreteQModel,
            model_kwargs=None,  # Pi model.
            q_model_kwargs=None,
            v_model_kwargs=None,
            initial_model_state_dict=None,  # All models.
            pretrain_std=0.75,  # With squash 0.75 is near uniform.
            ):
        """Saves input arguments; network defaults stored within."""
        if model_kwargs is None:
            model_kwargs = dict(hidden_sizes=[256, 256])
        if q_model_kwargs is None:
            q_model_kwargs = dict(hidden_sizes=[256, 256])
        if v_model_kwargs is None:
            v_model_kwargs = dict(hidden_sizes=[256, 256])
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
        self.q1_model = self.QModelCls(**self.env_model_kwargs, **self.q_model_kwargs)
        self.q2_model = self.QModelCls(**self.env_model_kwargs, **self.q_model_kwargs)
        self.target_q1_model = self.QModelCls(**self.env_model_kwargs,
            **self.q_model_kwargs)
        self.target_q2_model = self.QModelCls(**self.env_model_kwargs,
            **self.q_model_kwargs)
        self.target_q1_model.load_state_dict(self.q1_model.state_dict())
        self.target_q2_model.load_state_dict(self.q2_model.state_dict())
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
        self.q1_model.to(self.device)
        self.q2_model.to(self.device)
        self.target_q1_model.to(self.device)
        self.target_q2_model.to(self.device)

    def data_parallel(self):
        super().data_parallel
        DDP_WRAP = DDPC if self.device.type == "cpu" else DDP
        self.q1_model = DDP_WRAP(self.q1_model)
        self.q2_model = DDP_WRAP(self.q2_model)

    def give_min_itr_learn(self, min_itr_learn):
        self.min_itr_learn = min_itr_learn  # From algo.

    def make_env_to_model_kwargs(self, env_spaces):
        return dict(
            observation_shape=env_spaces.observation.shape,
            action_size=env_spaces.action.n,
        )

    def q(self, observation, prev_action, prev_reward):
        """Compute twin Q-values for state/observation and input action 
        (with grad)."""
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        q1 = self.q1_model(*model_inputs)
        q2 = self.q2_model(*model_inputs)
        return q1.cpu(), q2.cpu()

    def target_q(self, observation, prev_action, prev_reward):
        """Compute twin target Q-values for state/observation and input
        action.""" 
        model_inputs = buffer_to((observation, prev_action, prev_reward), device=self.device)
        target_q1 =self.target_q1_model(*model_inputs)
        target_q2 = self.target_q2_model(*model_inputs)
        return target_q1.cpu(), target_q2.cpu()

    def pi(self, observation, prev_action, prev_reward):
        """Compute action log-probabilities for state/observation, and
        sample new action (with grad).  Uses special ``sample_loglikelihood()``
        method of Gaussian distriution, which handles action squashing
        through this process."""
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        # mean, log_std = self.model(*model_inputs)
        # dist_info = DistInfoStd(mean=mean, log_std=log_std)
        # action, log_pi = self.distribution.sample_loglikelihood(dist_info)
        pi = self.model(*model_inputs)
        dist_info = DistInfo(prob=pi)
        action = self.distribution.sample(dist_info)
        log_pi = torch.log(pi)
        
        # TODO: potentially use argmax to determine the action instead of sampling from the distribution.
        # TODO: Figure out what to do for log_pi

        # action = self.distribution.sample(dist_info)
        # log_pi = self.distribution.log_likelihood(action, dist_info)
        log_pi, dist_info = buffer_to((log_pi, dist_info), device="cpu")
        return action, log_pi, dist_info  # Action stays on device for q models.

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        # mean, log_std = self.model(*model_inputs)
        # dist_info = DistInfoStd(mean=mean, log_std=log_std)
        # action = self.distribution.sample(dist_info)
        pi = self.model(*model_inputs)
        dist_info = DistInfo(prob=pi)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfo(dist_info=dist_info)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def update_target(self, tau=1):
        update_state_dict(self.target_q1_model, self.q1_model.state_dict(), tau)
        update_state_dict(self.target_q2_model, self.q2_model.state_dict(), tau)

    @property
    def models(self):
        return Models(pi=self.model, q1=self.q1_model, q2=self.q2_model)

    def pi_parameters(self):
        return self.model.parameters()

    def q1_parameters(self):
        return self.q1_model.parameters()

    def q2_parameters(self):
        return self.q2_model.parameters()

    def train_mode(self, itr):
        super().train_mode(itr)
        self.q1_model.train()
        self.q2_model.train()

    def sample_mode(self, itr):
        super().sample_mode(itr)
        self.q1_model.eval()
        self.q2_model.eval()
        if itr == 0:
            logger.log(f"Agent at itr {itr}, sample std: {self.pretrain_std}")
        if itr == self.min_itr_learn:
            logger.log(f"Agent at itr {itr}, sample std: learned.")
        # std = None if itr >= self.min_itr_learn else self.pretrain_std
        # self.distribution.set_std(std)  # If None: std from policy dist_info.

    def eval_mode(self, itr):
        super().eval_mode(itr)
        self.q1_model.eval()
        self.q2_model.eval()
        # self.distribution.set_std(0.)  # Deterministic (dist_info std ignored).

    def state_dict(self):
        return dict(
            model=self.model.state_dict(),  # Pi model.
            q1_model=self.q1_model.state_dict(),
            q2_model=self.q2_model.state_dict(),
            target_q1_model=self.target_q1_model.state_dict(),
            target_q2_model=self.target_q2_model.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.q1_model.load_state_dict(state_dict["q1_model"])
        self.q2_model.load_state_dict(state_dict["q2_model"])
        self.target_q1_model.load_state_dict(state_dict["target_q1_model"])
        self.target_q2_model.load_state_dict(state_dict["target_q2_model"])