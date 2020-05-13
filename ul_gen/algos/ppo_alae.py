
import torch
from torchvision.utils import save_image
from rlpyt.algos.pg.base import PolicyGradientAlgo, OptInfo
from rlpyt.agents.base import AgentInputs, AgentInputsRnn
from rlpyt.utils.tensor import valid_mean, infer_leading_dims
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs

import itertools
import logging
from ul_gen.alae.checkpointer import Checkpointer
from ul_gen.alae.scheduler import ComboMultiStepLR
from ul_gen.alae.custom_adam import LREQAdam
from ul_gen.alae.tracker import LossTracker

LossInputs = namedarraytuple("LossInputs",
    ["agent_inputs", "action", "return_", "advantage", "valid", "old_dist_info"])


class PPO_ALAE(PolicyGradientAlgo):
    """
    Proximal Policy Optimization algorithm.  Trains the agent by taking
    multiple epochs of gradient steps on minibatches of the training data at
    each iteration, with advantages computed by generalized advantage
    estimation.  Uses clipped likelihood ratios in the policy loss.
    """
    
    def __init__(
            self,
            discount=0.99,
            learning_rate=0.001,
            alae_learning_rate=0.002,
            value_loss_coeff=1.,
            entropy_loss_coeff=0.01,
            OptimCls=torch.optim.Adam,
            clip_grad_norm=1.,
            initial_optim_state_dict=None,
            gae_lambda=1,
            minibatches=4,
            epochs=4,
            ratio_clip=0.1,
            linear_lr_schedule=True,
            normalize_advantage=False,
            normalize_rewards=False,
            adam_betas=(0.0, 0.99),
            ):
        """Saves input settings."""
        save__init__args(locals())

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset=False,
            examples=None, world_size=1, rank=0):
        """
        Extends base ``initialize()`` to initialize learning rate schedule, if
        applicable.
        """
        self.agent = agent
        self.alae = self.agent.model.alae
        self.n_itr = n_itr
        self.mid_batch_reset = mid_batch_reset
        self.rets_rms = None
        self._batch_size = batch_spec.size // self.minibatches  # For logging.

        self.policy_params = itertools.chain(self.agent.model.policy.parameters(), self.agent.model.value.parameters())
        self.policy_optimizer = self.OptimCls(self.policy_params, lr=self.learning_rate)
        if self.linear_lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.policy_optimizer,
                lr_lambda=lambda itr: (self.n_itr - itr) / self.n_itr)  # Step once per itr.
            self._ratio_clip = self.ratio_clip  # Save base value.
        
        self.decoder_optimizer = LREQAdam([
            {'params': self.agent.model.alae.decoder.parameters()},
            {'params': self.agent.model.alae.mapping.parameters()}
        ], lr=self.alae_learning_rate, betas=self.adam_betas, weight_decay=0)

        self.encoder_optimizer = LREQAdam([
            {'params': self.agent.model.alae.encoder.parameters()},
            {'params': self.agent.model.alae.discriminator.parameters()},
        ], lr=self.alae_learning_rate, betas=self.adam_betas, weight_decay=0)
        
        # self.scheduler = ComboMultiStepLR(optimizers=
        #                         {
        #                         'encoder_optimizer': self.encoder_optimizer,
        #                         'decoder_optimizer': self.decoder_optimizer
        #                         },
        #                         milestones=[],
        #                         gamma=0.9,
        #                         reference_batch_size=32, base_lr=[])

        self.tracker = LossTracker('./data/alae/')

        model_dict = {
            'encoder': self.alae.encoder,
            'generator': self.alae.decoder,
            'mapping': self.alae.mapping,
            'discriminator': self.alae.discriminator,
            'policy': self.agent.model.policy,
            'value': self.agent.model.value
        }

        self.logger = logging.getLogger("logger")
        self.logger.setLevel(logging.DEBUG)

        # self.checkpointer = Checkpointer(cfg,
        #                             model_dict,
        #                             {
        #                                 'policy_optimizer': self.policy_optimizer,
        #                                 'encoder_optimizer': self.encoder_optimizer,
        #                                 'decoder_optimizer': self.decoder_optimizer,
        #                                 # 'scheduler': self.scheduler,
        #                                 'tracker': self.tracker
        #                             },
        #                             logger=self.logger,
        #                             save=True)

        # self.logger.info("Starting from epoch: %d" % (self.scheduler.start_epoch()))
        
    
    def optimize_agent(self, itr, samples):
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """

        recurrent = self.agent.recurrent
        agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        if hasattr(self.agent, "update_obs_rms"):
            self.agent.update_obs_rms(agent_inputs.observation)
        return_, advantage, valid = self.process_returns(samples, self.normalize_rewards)
        loss_inputs = LossInputs(  # So can slice all.
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            return_=return_,
            advantage=advantage,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info,
        )
        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.
        T, B = samples.env.reward.shape[:2]
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        batch_size = B if self.agent.recurrent else T * B
        mb_size = batch_size // self.minibatches
        for _ in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = slice(None) if recurrent else idxs % T
                B_idxs = idxs if recurrent else idxs // T

                batch_input = loss_inputs[T_idxs, B_idxs]
                obs = batch_input.agent_inputs.observation
                lead_dim, T, B, img_shape = infer_leading_dims(obs, 3)
                obs = obs.view(T*B, *img_shape)
                obs = obs.permute(0, 3, 1, 2).float() / 255.
                obs.requires_grad = True

                self.encoder_optimizer.zero_grad()
                loss_d = self.alae(obs, d_train=True, ae=False)
                self.tracker.update(dict(loss_d=loss_d))
                loss_d.backward()
                self.encoder_optimizer.step()

                self.decoder_optimizer.zero_grad()
                loss_g = self.alae(obs, d_train=False, ae=False)
                self.tracker.update(dict(loss_g=loss_g))
                loss_g.backward()
                self.decoder_optimizer.step()

                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()
                lae = self.alae(obs, d_train=False, ae=True)
                self.tracker.update(dict(lae=lae))
                lae.backward()
                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                self.policy_optimizer.zero_grad()
                # NOTE: if not recurrent, will lose leading T dim, should be OK.
                loss, entropy, perplexity = self.loss(*loss_inputs[T_idxs, B_idxs])
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy_params, self.clip_grad_norm)
                self.policy_optimizer.step()

                opt_info.loss.append(loss.item())
                opt_info.gradNorm.append(grad_norm)
                opt_info.entropy.append(entropy.item())
                opt_info.perplexity.append(perplexity.item())
                self.update_counter += 1
    
        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) / self.n_itr
        # self.scheduler.step()

        return opt_info

    def loss(self, agent_inputs, action, return_, advantage, valid, old_dist_info):
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agent to compute forward pass on training data, and uses
        the ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        dist_info, value = self.agent(*agent_inputs)
        dist = self.agent.distribution
        ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info,
            new_dist_info=dist_info)
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip,
            1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = - valid_mean(surrogate, valid)

        value_error = 0.5 * (value - return_) ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        entropy = dist.mean_entropy(dist_info, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy

        loss = pi_loss + value_loss + entropy_loss # + self.vae_loss_coeff * vae_loss
        
        perplexity = dist.mean_perplexity(dist_info, valid)
        return loss, entropy, perplexity

    
    def optim_state_dict(self):
        return {
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'encoder_optimizer': self.encoder_optimizer.state_dict(),
            'decoder_optimizer': self.decoder_optimizer.state_dict(),
        }

    def load_optim_state_dict(self, state_dict):
        self.policy_optimizer = self.policy_optimizer.load_state_dict(state_dict['policy_optimizer'])
        self.encoder_optimizer = self.encoder_optimizer.load_state_dict(state_dict['encoder_optimizer'])
        self.decoder_optimizer = self.decoder_optimizer.load_state_dict(state_dict['decoder_optimizer'])

