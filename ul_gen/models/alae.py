import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.autograd import Variable
from torchvision.utils import save_image

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.buffer import buffer_to
from ul_gen.models import MLP
from ul_gen.models.vae import Encoder, Decoder
from ul_gen.alae import losses


class ALAE(nn.Module):
    def __init__(self, image_shape, zdim, wdim, arch_type=1, encoder_layers=[32, 64, 128, 256], decoder_layers=[256, 128, 64, 32], mapping_dim=256, mapping_layers=3):
        super().__init__()

        self.zdim = zdim
        self.wdim = wdim
        # self.ndim = ndim
        self.image_shape = image_shape

        self.mapping = MLP(zdim, wdim, [mapping_dim] * mapping_layers)
        self.decoder = Decoder(wdim, image_shape, arch_type)
        self.encoder = Encoder(wdim, image_shape, arch_type)
        self.discriminator = nn.Sequential(
            MLP(wdim, 1, [mapping_dim] * mapping_layers),
            nn.Sigmoid()
        )

    def generate(self, size=1, z=None, device=None):
        """If z is None, you must include the size and the device for the model."""
        if z is None:
            z = torch.randn(size, self.zdim, device=device)
        w = self.mapping(z)
        img = self.decoder(w)
        return img

    def discrim(self, x):
        sample, mu, logstd = self.encoder(x)
        d = self.discriminator(mu)
        return d

    def forward(self, x, d_train, ae):
        if ae:
            # self.encoder.requires_grad_(True)

            z = torch.randn(x.shape[0], self.zdim, device=x.device)
            w1 = self.mapping(z)
            w1 = w1.detach()
            gen = self.decoder(w1)
            sample, w2, logstd = self.encoder(gen)

            Lae = torch.mean(((w2 - w1)**2))
            return Lae

        elif d_train:

            with torch.no_grad():
                Xp = self.generate(size=x.shape[0], device=x.device)

            # self.encoder.requires_grad_(True)       

            d_result_real = self.discrim(x)
            d_result_fake = self.discrim(Xp)

            loss_d = losses.discriminator_logistic_simple_gp(d_result_fake, d_result_real, x)
            return loss_d
        else:
            # self.encoder.requires_grad_(False)

            rec = self.generate(size=x.shape[0], device=x.device)
            d_result_fake = self.discrim(rec)

            loss_g = losses.generator_logistic_non_saturating(d_result_fake)
            return loss_g

    def log_images(self, obs, savepath, itr,n=100):
        self.eval()
        zs = torch.randn(n, self.zdim).to(device)
        samples = self.decoder(zs)
        obs = obs.reshape(-1, 64, 64, 3)[:10]
        obs = obs.permute(0, 3, 1, 2).float() / 255.
        _, _, latent, reconstruction,_ = self.forward(obs)
        recon = torch.cat((obs, reconstruction),dim=0)
        save_image(torch.Tensor(samples.detach().cpu()), os.path.join(savepath, 'samples_' + str(itr) +'.png'), nrow=10)
        save_image(torch.Tensor(recon.detach().cpu()), os.path.join(savepath, 'recon_' + str(itr) +'.png'), nrow=10)
        self.train()
        torch.save(self.state_dict(), '%s/alae-%d' % (savepath, (itr+1 // 5)*5))

    def lerp(self, other, beta):
        if hasattr(other, 'module'):
            other = other.module
        with torch.no_grad():
            params = list(self.mapping_tl.parameters()) + list(self.mapping_fl.parameters()) + list(self.decoder.parameters()) + list(self.encoder.parameters()) + list(self.dlatent_avg.parameters())
            other_param = list(other.mapping_tl.parameters()) + list(other.mapping_fl.parameters()) + list(other.decoder.parameters()) + list(other.encoder.parameters()) + list(other.dlatent_avg.parameters())
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - beta)


class ALAEPolicy(nn.Module):

    def __init__(self, zdim, wdim, img_shape=(3,64,64), 
                        policy_layers=[64, 64, 15,], value_layers=[64, 64, 1,], 
                        encoder_layers=[32, 64, 128, 256], decoder_layers=[256, 128, 64, 32],
                        act_fn='relu', detach_encoder=False, arch_type=1,
                        noise_prob=0.,noise_weight=1.,no_noise_weight=0.,
                        ):

        """
        arch_type: 0 for Conv2d-ReLU; 1 for Conv2d-BN-LeakyReLU
        noise_prob: Makes noise_prob % of images 0 or 1 (salt-and-pepper) before encoding
        noise_weight: If salt-and-pepper, weight on noise L2 loss
        no_noise_weight: If salt-and-pepper, weight on non-noise L2 loss
        

        """
        super().__init__()
        self.zdim = zdim
        self.wdim = wdim
        self.detach_encoder = detach_encoder
        self.noise_prob = noise_prob
        self.noise_weight, self.no_noise_weight = noise_weight, no_noise_weight
        self.alae = ALAE(img_shape, zdim, wdim, arch_type=arch_type, encoder_layers=encoder_layers, decoder_layers=decoder_layers)
        
        act_fn = {
            'relu' : lambda: nn.ReLU(),
            'tanh' : lambda: nn.Tanh()
        }[act_fn]

        policy = []
        last_layer = self.wdim
        for l in policy_layers:
            policy.append(nn.Linear(last_layer, l))
            policy.append(nn.ReLU())
            last_layer = l
        policy.pop()
        policy.append(nn.Softmax(dim=-1))

        value = []
        last_layer = self.wdim
        for l in value_layers:
            value.append(nn.Linear(last_layer, l))
            value.append(nn.ReLU())
            last_layer = l
        value.pop()

        self.policy = nn.Sequential(*policy)
        self.value = nn.Sequential(*value)

    def override_policy_value(self,policy_layers=[64, 64, 15,], value_layers=[64, 64, 1,]):
        """ hack: if vae was pretrained with different policy, value networks, can reinit
        assumes shared_extractor is []"""
        policy = []
        last_layer = self.wdim
        for l in policy_layers:
            policy.append(nn.Linear(last_layer, l))
            policy.append(nn.ReLU())
            last_layer = l
        policy.pop()
        policy.append(nn.Softmax(dim=-1))
        value = []
        last_layer = self.wdim
        for l in value_layers:
            value.append(nn.Linear(last_layer, l))
            value.append(nn.ReLU())
            last_layer = l
        value.pop()
        self.policy = nn.Sequential(*policy)
        self.value = nn.Sequential(*value)

    def forward(self, observation, prev_action=None, prev_reward=None):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        obs = observation.view(T*B, *img_shape)
        obs = obs.permute(0, 3, 1, 2).float() / 255.
        noise_idx = None
        if self.noise_prob:
            obs, noise_idx = salt_and_pepper(obs,self.noise_prob)

        _, w, _ = self.alae.encoder(obs)

        if self.detach_encoder:
            w = w.detach()
        
        pi = self.policy(w)
        value = self.value(w).squeeze(-1)

        pi, value = restore_leading_dims((pi, value), lead_dim, T, B)
        return pi, value

    def log_images(self, obs, savepath, itr,n=100):
        self.eval()
        zs = torch.randn(n, self.zdim).to(device)
        samples = self.decoder(zs)
        obs = obs.reshape(-1, 64, 64, 3)[:10]
        _, _, latent, reconstruction,_ = self.forward(obs)
        obs = obs.permute(0, 3, 1, 2).float().to(device) / 255.
        recon = torch.cat((obs, reconstruction),dim=0)
        save_image(torch.Tensor(samples.detach().cpu()), os.path.join(savepath, 'samples_' + str(itr) +'.png'), nrow=10)
        save_image(torch.Tensor(recon.detach().cpu()), os.path.join(savepath, 'recon_' + str(itr) +'.png'), nrow=10)
        self.train()
        torch.save(self.state_dict(), '%s/vae-%d' % (savepath, (itr+1 // 5)*5))

    def log_samples(self, savepath, itr, n=64):
        self.eval()
        zs = torch.randn(n, self.zdim).to(device)
        samples = self.decoder(zs)
        save_image(torch.Tensor(samples.detach().cpu()), os.path.join(savepath, 'samples_' + str(itr) +'.png'), nrow=8)
        self.train()
