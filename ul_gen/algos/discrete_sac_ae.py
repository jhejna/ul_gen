
import numpy as np
import torch
from collections import namedtuple

from rlpyt.algos.base import RlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.replays.non_sequence.uniform import (UniformReplayBuffer,
    AsyncUniformReplayBuffer)
from rlpyt.replays.non_sequence.time_limit import (TlUniformReplayBuffer,
    AsyncTlUniformReplayBuffer)
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.buffer import buffer_to
# from rlpyt.distributions.gaussian import Gaussian
# from rlpyt.distributions.gaussian import DistInfo as GaussianDistInfo
from rlpyt.distributions.categorical import Categorical, DistInfo

from rlpyt.utils.tensor import valid_mean
from rlpyt.algos.utils import valid_from_done


OptInfo = namedtuple("OptInfo",
    ["qLoss", "piLoss",
    "qGradNorm", "piGradNorm",
    "q1", "q2", "piLog", "qMeanDiff", "alpha", "aeLoss", "reconLoss", "zLoss"])
SamplesToBuffer = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done"])
SamplesToBufferTl = namedarraytuple("SamplesToBufferTl",
    SamplesToBuffer._fields + ("timeout",))

class DiscreteSACAE(RlAlgorithm):
    """Soft actor critic algorithm, training from a replay buffer."""

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            discount=0.99,
            batch_size=256,
            min_steps_learn=int(1e4),
            replay_size=int(1e6),
            replay_ratio=256,  # data_consumption / data_generation
            target_update_tau=0.005,  # tau=1 for hard update.
            target_update_interval=1,  # 1000 for hard update, 1 for soft.
            learning_rate=3e-4,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            initial_optim_state_dict=None,  # for all of them.
            action_prior="uniform",  # or "gaussian"
            reward_scale=1,
            target_entropy="auto",  # "auto", float, or None
            clip_grad_norm=1e9,
            # policy_output_regularization=0.001,
            n_step_return=1,
            updates_per_sync=1,  # For async mode only.
            bootstrap_timelimit=True,
            detach_critic=False,
            ae_update_interval=1,
            ae_learning_rate=3e-4,
            ae_optim_kwargs={},
            ae_beta=1,
            ae_pretraining=False,
            ):
        """Save input arguments."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        assert action_prior in ["uniform", "gaussian"]
        self._batch_size = batch_size
        del batch_size  # Property.
        save__init__args(locals())

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples,
            world_size=1, rank=0):
        """Stores input arguments and initializes replay buffer and optimizer.
        Use in non-async runners.  Computes number of gradient updates per
        optimization iteration as `(replay_ratio * sampler-batch-size /
        training-batch_size)`."""
        self.agent = agent
        self.n_itr = n_itr
        self.mid_batch_reset = mid_batch_reset
        self.sampler_bs = sampler_bs = batch_spec.size
        self.updates_per_optimize = int(self.replay_ratio * sampler_bs /
            self.batch_size)
        logger.log(f"From sampler batch size {sampler_bs}, training "
            f"batch size {self.batch_size}, and replay ratio "
            f"{self.replay_ratio}, computed {self.updates_per_optimize} "
            f"updates per iteration.")
        self.min_itr_learn = self.min_steps_learn // sampler_bs
        agent.give_min_itr_learn(self.min_itr_learn)
        self.initialize_replay_buffer(examples, batch_spec)
        self.optim_initialize(rank)

    def async_initialize(self, agent, sampler_n_itr, batch_spec, mid_batch_reset,
            examples, world_size=1):
        """Used in async runner only; returns replay buffer allocated in shared
        memory, does not instantiate optimizer. """
        self.agent = agent
        self.n_itr = sampler_n_itr
        self.initialize_replay_buffer(examples, batch_spec, async_=True)
        self.mid_batch_reset = mid_batch_reset
        self.sampler_bs = sampler_bs = batch_spec.size
        self.updates_per_optimize = self.updates_per_sync
        self.min_itr_learn = int(self.min_steps_learn // sampler_bs)
        agent.give_min_itr_learn(self.min_itr_learn)
        return self.replay_buffer

    def optim_initialize(self, rank=0):
        """Called in initilize or by async runner after forking sampler."""
        self.rank = rank
        self.actor_optimizer = self.OptimCls(self.agent.actor_parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        self.critic_optimizer = self.OptimCls(self.agent.critic_parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        self._log_alpha = torch.zeros(1, requires_grad=True)
        self._alpha = torch.exp(self._log_alpha.detach())
        self.alpha_optimizer = self.OptimCls((self._log_alpha,),
            lr=self.learning_rate, **self.optim_kwargs)
        if self.target_entropy == "auto":
            self.target_entropy = -np.log((1.0 / self.agent.env_spaces.action.n)) * 0.98 #-np.prod(self.agent.env_spaces.action.shape)
        if self.initial_optim_state_dict is not None:
            self.load_optim_state_dict(self.initial_optim_state_dict)
        # if self.action_prior == "gaussian":
        #     self.action_prior_distribution = Gaussian(
        #         dim=np.prod(self.agent.env_spaces.action.shape), std=1.)
        # self.decoder_optimizer = self.OptimCls(self.)
        self.encoder_optimizer = self.OptimCls(self.agent.encoder_parameters(), lr=self.ae_learning_rate, **self.ae_optim_kwargs)
        self.decoder_optimizer = self.OptimCls(self.agent.decoder_parameters(), lr=self.ae_learning_rate, **self.ae_optim_kwargs)

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        """
        Allocates replay buffer using examples and with the fields in `SamplesToBuffer`
        namedarraytuple.
        """
        example_to_buffer = SamplesToBuffer(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
        )
        if not self.bootstrap_timelimit:
            ReplayCls = AsyncUniformReplayBuffer if async_ else UniformReplayBuffer
        else:
            example_to_buffer = SamplesToBufferTl(*example_to_buffer,
                timeout=examples["env_info"].timeout)
            ReplayCls = AsyncTlUniformReplayBuffer if async_ else TlUniformReplayBuffer
        replay_kwargs = dict(
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            n_step_return=self.n_step_return,
        )
        self.replay_buffer = ReplayCls(**replay_kwargs)

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        """
        Extracts the needed fields from input samples and stores them in the 
        replay buffer.  Then samples from the replay buffer to train the agent
        by gradient updates (with the number of updates determined by replay
        ratio, sampler batch size, and training batch size).
        """
        itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr.
        if samples is not None:
            samples_to_buffer = self.samples_to_buffer(samples)
            self.replay_buffer.append_samples(samples_to_buffer)
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        if itr < self.min_itr_learn:
            return opt_info
        # print("###############################")
        # print("Updates per opt", self.updates_per_optimize)
        for _ in range(self.updates_per_optimize):
            samples_from_replay = self.replay_buffer.sample_batch(self.batch_size)

            if not self.ae_pretraining:
                losses, values = self.loss(samples_from_replay)
                critic_loss, actor_loss, alpha_loss = losses

                if alpha_loss is not None:
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    self._alpha = torch.exp(self._log_alpha.detach())

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.actor_parameters(),
                    self.clip_grad_norm)
                self.actor_optimizer.step()

                # Step Q's last because pi_loss.backward() uses them?
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.critic_parameters(),
                    self.clip_grad_norm)
                self.critic_optimizer.step()

                grad_norms = (critic_grad_norm, actor_grad_norm)

            else:
                losses, values, grad_norms = None, None, None

            # Add VAE Loss update:
            if self.update_counter % self.ae_update_interval == 0:
                ae_loss, ae_values = self.ae_loss(samples_from_replay)
                # Update the Auto-Encoder
                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                ae_loss.backward()
                # TODO: Perhaps clip grad norm here as well?
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()
            else:
                ae_loss, ae_values = None, (None, None)

            self.append_opt_info_(opt_info, losses, grad_norms, values, ae_loss, ae_values)
            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target(self.target_update_tau)

        return opt_info

    def samples_to_buffer(self, samples):
        """Defines how to add data from sampler into the replay buffer. Called
        in optimize_agent() if samples are provided to that method."""
        samples_to_buffer = SamplesToBuffer(
            observation=samples.env.observation,
            action=samples.agent.action,
            reward=samples.env.reward,
            done=samples.env.done,
        )
        if self.bootstrap_timelimit:
            samples_to_buffer = SamplesToBufferTl(*samples_to_buffer,
                timeout=samples.env.env_info.timeout)
        return samples_to_buffer

    def ae_loss(self, samples):
        agent_inputs, target_inputs, action = buffer_to(
            (samples.agent_inputs, samples.target_inputs, samples.action), self.agent.device)

        z, mu, log_sd = self.agent.z(*agent_inputs)
        x_hat = self.agent.decode(z)
       
        x = agent_inputs.observation.permute(0, 3, 1, 2).float() / 255.

        batch_size = x.shape[0]

        recon_loss = torch.sum( (x_hat - x).pow(2) ) / batch_size
        if self.agent.rae:
            latent_loss = (0.5 * z.pow(2)).sum() / batch_size
        else:
            log_var = 2*log_sd
            latent_loss = torch.sum(-0.5*(1 + log_var - mu.pow(2) - log_var.exp())) / bs
        
        loss = recon_loss + self.ae_beta * latent_loss
        values = tuple(val.detach() for val in (recon_loss, latent_loss))
        return loss, values

    def loss(self, samples):
        """
        Computes losses for twin Q-values against the min of twin target Q-values
        and an entropy term.  Computes reparameterized policy loss, and loss for
        tuning entropy weighting, alpha.  
        
        Input samples have leading batch dimension [B,..] (but not time).
        """
        agent_inputs, target_inputs, action = buffer_to(
            (samples.agent_inputs, samples.target_inputs, samples.action))

        if self.mid_batch_reset and not self.agent.recurrent:
            valid = torch.ones_like(samples.done, dtype=torch.float)  # or None
        else:
            valid = valid_from_done(samples.done)
        if self.bootstrap_timelimit:
            # To avoid non-use of bootstrap when environment is 'done' due to
            # time-limit, turn off training on these samples.
            valid *= (1 - samples.timeout_n.float())

        q1_logits, q2_logits = self.agent.q(*agent_inputs, detach_encoder=self.detach_critic)
        
        # print(action, action.requires_grad)
        q1 = q1_logits.gather(1, action.long().unsqueeze(-1))
        q2 = q2_logits.gather(1, action.long().unsqueeze(-1))

        with torch.no_grad():
            target_action, target_log_pi, target_dist_info = self.agent.pi(*target_inputs) # TODO Get act prob, and correct target log pi
            target_q1, target_q2 = self.agent.target_q(*target_inputs) # Note: remove action input
            min_target_q = torch.min(target_q1, target_q2)
            target_value = torch.sum(target_dist_info.prob * (min_target_q - self._alpha * target_log_pi), dim=1, keepdims=True) # TODO: Verify if this should be sum or mean.
        disc = self.discount ** self.n_step_return
        y = (self.reward_scale * samples.return_.unsqueeze(-1) +
            (1 - samples.done_n.float().unsqueeze(-1)) * disc * target_value)

        q1_loss = 0.5 * valid_mean((y - q1) ** 2, valid)
        q2_loss = 0.5 * valid_mean((y - q2) ** 2, valid)
        critic_loss = q1_loss + q2_loss
        
        new_action, log_pi, dist_info = self.agent.pi(*agent_inputs, detach_encoder=True)

        log_target1, log_target2 = self.agent.q(*agent_inputs, detach_encoder=True)
        min_log_target = torch.min(log_target1, log_target2)
        # prior_log_pi = self.get_action_prior(new_action.cpu())

        pi_losses = torch.sum(dist_info.prob * (self._alpha * log_pi - min_log_target), dim=1, keepdims=True)
        # print("Losses Shape", pi_losses.shape)
        # if self.policy_output_regularization > 0:
        #     pi_losses += self.policy_output_regularization * torch.mean(
        #         0.5 * pi_mean ** 2 + 0.5 * pi_log_std ** 2, dim=-1)

        pi_loss = valid_mean(pi_losses, valid)

        if self.target_entropy is not None:
            pi_entropy = torch.sum(dist_info.prob * log_pi, dim=1)
            alpha_losses = - self._log_alpha * (pi_entropy.detach() + self.target_entropy)
            alpha_loss = valid_mean(alpha_losses, valid)

            # Alternate Alpha Loss from explicit entropy? # TODO: Investigate
            # alpha_losses = - self.log_alpha * ()
        else:
            alpha_loss = None

        losses = (critic_loss, pi_loss, alpha_loss)
        values = tuple(val.detach() for val in (q1, q2, log_pi))
        return losses, values

    def get_action_prior(self, action):
        if self.action_prior == "uniform":
            prior_log_pi = 0.0
        # elif self.action_prior == "gaussian":
        #     prior_log_pi = self.action_prior_distribution.log_likelihood(
        #         action, GaussianDistInfo(mean=torch.zeros_like(action)))
        return prior_log_pi

    def append_opt_info_(self, opt_info, losses, grad_norms, values, ae_loss, ae_values):
        """In-place."""
        if not self.ae_pretraining:
            critic_loss, pi_loss, alpha_loss = losses
            critic_grad_norm, actor_grad_norm = grad_norms
            q1, q2, log_pi = values
            opt_info.qLoss.append(critic_loss.item())
            opt_info.piLoss.append(pi_loss.item())
            opt_info.qGradNorm.append(critic_grad_norm)
            opt_info.piGradNorm.append(actor_grad_norm)
            opt_info.q1.extend(q1[::10].numpy())  # Downsample for stats.
            opt_info.q2.extend(q2[::10].numpy())
            opt_info.piLog.extend(log_pi[::10].numpy())
            opt_info.qMeanDiff.append(torch.mean(abs(q1 - q2)).item())
            opt_info.alpha.append(self._alpha.item())
        if not ae_loss is None:
            recon_loss, latent_loss = ae_values
            opt_info.aeLoss.append(ae_loss.item())
            opt_info.reconLoss.append(recon_loss.item())
            opt_info.zLoss.append(latent_loss.item())

    def optim_state_dict(self):
        return dict(
            actor_optimizer=self.actor_optimizer.state_dict(),
            critic_optimizer=self.critic_optimizer.state_dict(),
            alpha_optimizer=self.alpha_optimizer.state_dict(),
            log_alpha=self._log_alpha.detach().item(),
        )

    def load_optim_state_dict(self, state_dict):
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])
        with torch.no_grad():
            self._log_alpha[:] = state_dict["log_alpha"]
            self._alpha = torch.exp(self._log_alpha.detach())
