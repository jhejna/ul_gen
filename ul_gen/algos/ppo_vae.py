
import torch
from torchvision.utils import save_image
from rlpyt.algos.pg.base import PolicyGradientAlgo, OptInfo
from rlpyt.agents.base import AgentInputs, AgentInputsRnn
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs

LossInputs = namedarraytuple("LossInputs",
    ["agent_inputs", "action", "return_", "advantage", "valid", "old_dist_info"])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PPO_VAE(PolicyGradientAlgo):
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
            vae_learning_rate=0.0001,
            value_loss_coeff=1.,
            entropy_loss_coeff=0.01,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            VaeOptimCls=torch.optim.Adam,
            vae_optim_kwargs=None,
            clip_grad_norm=1.,
            initial_optim_state_dict=None,
            gae_lambda=1,
            minibatches=4,
            epochs=4,
            ratio_clip=0.1,
            linear_lr_schedule=True,
            vae_linear_lr_schedule=True,
            normalize_advantage=False,
            vae_beta=0.9,
            vae_loss_coeff=0.1,
            vae_loss_type="l2",
            vae_norm_loss=False,
            vae_update_freq=1,
            ):
        """Saves input settings."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())

    def initialize(self, *args, **kwargs):
        """
        Extends base ``initialize()`` to initialize learning rate schedule, if
        applicable.
        """
        super().initialize(*args, **kwargs)
        self._batch_size = self.batch_spec.size // self.minibatches  # For logging.
        if self.linear_lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda itr: (self.n_itr - itr) / self.n_itr)  # Step once per itr.
            self._ratio_clip = self.ratio_clip  # Save base value.
        # self.vae_optimizer = VaeOptimCls(self.agent.parameters(),lr=self.vae_learning_rate, **self.vae_optim_kwargs)
        # if self.vae_linear_lr_schedule:
        #     self.vae_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        #         optimizer=self.vae_optimizer,
        #         lr_lambda=lambda itr: (self.n_itr - itr) / self.n_itr)  # Step once per itr.
        #     )
    
    def optimize_agent(self, itr, samples):
        """
        Train the agent, for multiple epochs over minibatches taken from the
        input samples.  Organizes agent inputs from the training data, and
        moves them to device (e.g. GPU) up front, so that minibatches are
        formed within device, without further data transfer.
        """
        if hasattr(self, "beta"):
            print("############################################")
            print("#                 HAS BETA ATTRIBUTE       #")
            print("############################################")
        recurrent = self.agent.recurrent
        agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        if hasattr(self.agent, "update_obs_rms"):
            self.agent.update_obs_rms(agent_inputs.observation)
        return_, advantage, valid = self.process_returns(samples)
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
                self.optimizer.zero_grad()
                rnn_state = init_rnn_state[B_idxs] if recurrent else None
                # NOTE: if not recurrent, will lose leading T dim, should be OK.
                loss, entropy, perplexity = self.loss(
                    *loss_inputs[T_idxs, B_idxs], rnn_state)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                opt_info.loss.append(loss.item())
                opt_info.gradNorm.append(grad_norm)
                opt_info.entropy.append(entropy.item())
                opt_info.perplexity.append(perplexity.item())
                self.update_counter += 1
        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) / self.n_itr
        # if self.vae_lr_scheduler:
        #     self.vae_lr_scheduler.step()

        return opt_info

    def loss(self, agent_inputs, action, return_, advantage, valid, old_dist_info,
            init_rnn_state=None):
        """
        Compute the training loss: policy_loss + value_loss + entropy_loss
        Policy loss: min(likelhood-ratio * advantage, clip(likelihood_ratio, 1-eps, 1+eps) * advantage)
        Value loss:  0.5 * (estimated_value - return) ^ 2
        Calls the agent to compute forward pass on training data, and uses
        the ``agent.distribution`` to compute likelihoods and entropies.  Valid
        for feedforward or recurrent agents.
        """
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, value, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
        else:
            (dist_info, value, latent, reconstruction), noise_idx = self.agent(*agent_inputs)
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

        # Add VAE Losses:
        obs = buffer_to((agent_inputs.observation), "cpu")
        obs = obs.permute(0, 3, 1, 2).float() / 255.

        bs = obs.shape[0]
        mu, logsd = torch.chunk(latent, 2, dim=1)
        logvar = 2*logsd

        kl_loss = torch.sum(-0.5*(1 + logvar - mu.pow(2) - logvar.exp())) / bs
        if self.vae_loss_type == "l2":
            if self.agent.model.noise_prob:
                obs, reconstruction = obs.reshape(bs,-1), reconstruction.reshape(bs,-1)
                noise_idx = noise_idx.reshape(-1)
                noise = torch.sum((obs[:,noise_idx]-reconstruction[:,noise_idx]).pow(2)) / bs
                no_noise_idx = 1 - noise_idx
                no_noise = torch.sum((obs[:,no_noise_idx]-reconstruction[:,no_noise_idx]).pow(2)) / bs
                recon_loss = self.agent.model.noise_weight * noise + self.agent.model.no_noise_weight * no_noise
            else:
                recon_loss = torch.sum((obs - reconstruction).pow(2)) / bs

        elif self.vae_loss_type == "bce":
            recon_loss = torch.nn.functional.binary_cross_entropy(reconstruction, obs)
        else:
            raise NotImplementedError
        
        
        vae_loss = self.vae_loss_coeff * (recon_loss + self.vae_beta * kl_loss)

        policy_loss = pi_loss + value_loss + entropy_loss # + self.vae_loss_coeff * vae_loss
        
        if self.vae_norm_loss:
            vae_loss = vae_loss * torch.abs(policy_loss.detach()) / vae_loss.detach()

        loss = policy_loss +vae_loss

        perplexity = dist.mean_perplexity(dist_info, valid)
        return loss, entropy, perplexity

