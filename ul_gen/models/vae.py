import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.autograd import Variable
from torchvision.utils import save_image

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.buffer import buffer_to

def salt_and_pepper(img,prob):
    assert prob <= 1
    c,h,w = img.shape[-3:]

    noisy = img.reshape(-1,c*h*w)
    rnd = np.random.rand(c*h*w)
    idxzero = (rnd < prob/2).nonzero()[0]
    idxone = (rnd > 1 - prob/2).nonzero()[0]
    noisy[:,idxzero] = 0.
    noisy[:,idxone] = 1.
    noisy=noisy.reshape(-1,c,h,w)
    # save_image(noisy, "/home/karam/Downloads/noise.png")
    return noisy, np.concatenate((idxone,idxzero),axis=0)

class Encoder(nn.Module):
    def __init__(self, zdim, img_shape, arch_type, hidden_dims=[32,64,128,256], rae=False):
        super().__init__()
        self.rae = rae
        self.zdim = zdim
        in_channels, self.h, _ = img_shape
        C = (self.h // 2 ** len(hidden_dims))**2

        if arch_type == 0:
            modules = []
            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, h_dim, 4, 2, 1),
                        nn.ReLU(True)))
                in_channels = h_dim

        elif arch_type == 1:
            modules=[nn.BatchNorm2d(in_channels)]

            for h_dim in hidden_dims:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, h_dim, 4, 2, 1),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU(.2)))
                in_channels = h_dim

        self.main = nn.Sequential(*modules)

        self.mu = nn.Linear(hidden_dims[-1] * C, zdim)
        if not self.rae:
            self.logsd = nn.Linear(hidden_dims[-1] * C, zdim)

    def forward(self,x):
        bs = x.shape[0]
        x = self.main(x).reshape(bs,-1)
        mu = self.mu(x)
        if self.rae:
            return mu
        else:
            logsd = self.logsd(x)
            eps = buffer_to(Variable(torch.randn([bs, self.zdim])), x.device)
            z = eps * logsd.exp() + mu
            return z, mu, logsd

class Decoder(nn.Module):
    def __init__(self, zdim, img_shape, arch_type, hidden_dims=[256,128,64,32]):
        """
        """

        super().__init__()

        channel_out, self.h, _ = img_shape
        self.odim = self.h*self.h*channel_out
        self.init_dim = (self.h // (2 ** len(hidden_dims)))
        C = self.init_dim **2
        self.lin = nn.Linear(zdim, C * hidden_dims[0] )
        self.relu = nn.ReLU(True)
        modules=[nn.ConvTranspose2d(hidden_dims[0],hidden_dims[1], 4, 2, 1)] 
        in_channels = hidden_dims[1]    

        if arch_type == 0:
            # self.main = nn.Sequential(
            #     nn.ConvTranspose2d(128, 128, 4, 2, 1), # h/4 x h/4
            #     nn.ReLU(True),
            #     nn.ConvTranspose2d(128, 64, 4, 2, 1), # h/2 x h/2
            #     nn.ReLU(True),
            #     nn.ConvTranspose2d(64, 32, 4, 2, 1), # h x h
            #     nn.ReLU(True),
            #     nn.ConvTranspose2d(32, 3, 3, 1, 1), # h x h
            #     nn.Sigmoid()
            # )
            for h_dim in hidden_dims[2:]:
                modules.append(
                    nn.Sequential(
                        nn.ReLU(True),
                        nn.ConvTranspose2d(in_channels, h_dim, 4, 2, 1)))
                in_channels = h_dim
        elif arch_type==1:  
            for h_dim in hidden_dims[2:]:
                modules.append(
                    nn.Sequential(
                        nn.BatchNorm2d(in_channels),
                        nn.LeakyReLU(.2),
                        nn.ConvTranspose2d(in_channels, h_dim, 4, 2, 1)))
                in_channels = h_dim
        modules.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels, channel_out, 4, 2, 1),
            nn.Sigmoid()))
        self.main = nn.Sequential(*modules)

    def forward(self, z):
        bs = z.shape[0]
        z = self.relu(self.lin(z)).reshape(bs, -1, self.init_dim, self.init_dim)
        x = self.main(z)
        return x

class VaePolicy(nn.Module):

    def __init__(self, zdim, img_shape=(3,64,64), shared_layers=[], 
                        policy_layers=[64, 64, 15,], value_layers=[64, 64, 1,], 
                        encoder_layers=[32, 64, 128, 256], decoder_layers=[256, 128, 64, 32],
                        act_fn='relu', deterministic=False,
                        detach_vae=False, detach_value=False,
                        detach_policy=False, arch_type=0.,
                        noise_prob=0.,noise_weight=1.,no_noise_weight=0.,
                        rae=False, greyscale=False):

        """
        arch_type: 0 for Conv2d-ReLU; 1 for Conv2d-BN-LeakyReLU
        noise_prob: Makes noise_prob % of images 0 or 1 (salt-and-pepper) before encoding
        noise_weight: If salt-and-pepper, weight on noise L2 loss
        no_noise_weight: If salt-and-pepper, weight on non-noise L2 loss
        

        """
        super().__init__()
        self.zdim = zdim
        self.rae = rae
        self.greyscale=greyscale
        self.detach_value = detach_value
        self.detach_policy = detach_policy
        self.detach_vae = detach_vae
        self.deterministic = deterministic
        self.noise_prob = noise_prob
        self.noise_weight, self.no_noise_weight = noise_weight, no_noise_weight
        self.encoder = Encoder(zdim,img_shape,arch_type,hidden_dims=encoder_layers, rae=rae)
        self.decoder = Decoder(zdim,img_shape,arch_type,hidden_dims=decoder_layers)
        act_fn = {
            'relu' : lambda: nn.ReLU(),
            'tanh' : lambda: nn.Tanh()
        }[act_fn]
        last_layer = self.zdim
        shared_extractor = [act_fn()]
        for l in shared_layers:
            shared_extractor.append(nn.Linear(last_layer, l))
            shared_extractor.append(act_fn())
            last_layer = l
        policy = []
        extractor_out = last_layer
        for l in policy_layers:
            policy.append(nn.Linear(last_layer, l))
            policy.append(nn.ReLU())
            last_layer = l
        policy.pop()
        policy.append(nn.Softmax(dim=-1))
        value = []
        last_layer = extractor_out
        for l in value_layers:
            value.append(nn.Linear(last_layer, l))
            value.append(nn.ReLU())
            last_layer = l
        value.pop()
        self.shared_extractor = nn.Sequential(*shared_extractor)
        self.policy = nn.Sequential(*policy)
        self.value = nn.Sequential(*value)

    def override_policy_value(self,policy_layers=[64, 64, 15,], value_layers=[64, 64, 1,]):
        """ hack: if vae was pretrained with different policy, value networks, can reinit
        assumes shared_extractor is []"""
        policy = []
        last_layer = self.zdim
        for l in policy_layers:
            policy.append(nn.Linear(last_layer, l))
            policy.append(nn.ReLU())
            last_layer = l
        policy.pop()
        policy.append(nn.Softmax(dim=-1))
        value = []
        last_layer = self.zdim
        for l in value_layers:
            value.append(nn.Linear(last_layer, l))
            value.append(nn.ReLU())
            last_layer = l
        value.pop()
        self.policy = nn.Sequential(*policy)
        self.value = nn.Sequential(*value)

    def forward(self, observation, prev_action=None, prev_reward=None):
        self.device = observation.device
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        obs = observation.view(T*B, *img_shape)
        obs = obs.permute(0, 3, 1, 2).float() / 255.
        if self.greyscale:
            obs = torch.mean(obs,dim=1, keepdims=True)
        noise_idx = None
        if self.noise_prob:
            obs, noise_idx = salt_and_pepper(obs,self.noise_prob)

        z, mu, logsd = self.encoder(obs)
        reconstruction = self.decoder(z)
        extractor_in = mu if self.deterministic else z

        if self.detach_vae:
            extractor_in = extractor_in.detach()
        extractor_out = self.shared_extractor(extractor_in)

        if self.detach_policy:
            policy_in = extractor_out.detach()
        else:
            policy_in = extractor_out
        if self.detach_value:
            value_in = extractor_out.detach()
        else:
            value_in = extractor_out
        
        act_dist = self.policy(policy_in)
        value = self.value(value_in).squeeze(-1)

        if self.rae:
            latent = mu
        else:
            latent = torch.cat((mu, logsd), dim=1)
        
        act_dist, value, latent, reconstruction = restore_leading_dims((act_dist, value, latent, reconstruction), lead_dim, T, B)
        return act_dist, value, latent, reconstruction, noise_idx

    def loss(self, loss_type, inputs):
        _, _, latent, reconstruction,noise_idx = self.forward(*inputs)
        obs = inputs.observation.permute(0, 1, 4, 2, 3).float() / 255.
        bs = obs.shape[0]* obs.shape[1]

        if self.rae:
            latent_loss = (0.5 * latent.pow(2).sum()) / bs
        else:
            mu, logsd = torch.chunk(latent, 2, dim=1)
            logvar = 2*logsd
            latent_loss = torch.sum(-0.5*(1 + logvar - mu.pow(2) - logvar.exp())) / bs
        if loss_type == "l2":
            if noise_idx is not None:
                obs, reconstruction = obs.reshape(bs,-1), reconstruction.reshape(bs,-1)
                noise_idx = noise_idx.reshape(-1)
                noise = torch.sum((obs[:,noise_idx]-reconstruction[:,noise_idx]).pow(2)) / bs
                no_noise_idx = 1 - noise_idx
                no_noise = torch.sum((obs[:,no_noise_idx]-reconstruction[:,no_noise_idx]).pow(2)) / bs
                recon_loss = self.noise_weight * noise + self.no_noise_weight * no_noise
            else:
                recon_loss = torch.sum((obs - reconstruction).pow(2)) / bs
        elif loss_type == "bce":
            recon_loss = nn.BCELoss()(reconstruction, obs)*100
        else:
            raise NotImplementedError
        return recon_loss, latent_loss

    def log_images(self, obs, savepath, itr, device, n=100):
        self.eval()
        zs = torch.randn(n, self.zdim).to(device)
        samples = self.decoder(zs)
        obs = obs.reshape(-1, 64, 64, 3)[:10]
        _, _, latent, reconstruction,_ = self.forward(obs)
        obs = obs.permute(0, 3, 1, 2).float().to(device) / 255.
        if self.greyscale:
            obs = torch.mean(obs,dim=1, keepdims=True)
        recon = torch.cat((obs, reconstruction),dim=0)
        save_image(torch.Tensor(samples.detach().cpu()), os.path.join(savepath, 'samples_' + str(itr) +'.png'), nrow=10)
        save_image(torch.Tensor(recon.detach().cpu()), os.path.join(savepath, 'recon_' + str(itr) +'.png'), nrow=10)
        self.train()
        torch.save(self.state_dict(), '%s/vae-%d' % (savepath, (itr+1 // 5)*5))

    def log_samples(self, savepath, itr, device, n=64):
        self.eval()
        zs = torch.randn(n, self.zdim).to(device)
        samples = self.decoder(zs)
        save_image(torch.Tensor(samples.detach().cpu()), os.path.join(savepath, 'samples_' + str(itr) +'.png'), nrow=8)
        self.train()

'''
TEST POLICY
'''

class BaselinePolicy(nn.Module):
    def __init__(self, img_shape=(3, 64, 64), shared_layers=[], 
                        zdim=128, arch_type=0,encoder_layers=[32, 64, 128, 256],
                        policy_layers=[64, 64, 15], value_layers=[64, 64, 1,], act_fn='relu',
                        noise_prob=0):

        super().__init__()

        self.encoder =  Encoder(zdim,img_shape,arch_type,hidden_dims=encoder_layers)
        self.noise_prob=noise_prob
        act_fn = {
            'relu' : lambda: nn.ReLU(),
            'tanh' : lambda: nn.Tanh()
        }[act_fn]

        last_layer = zdim
        shared_extractor = [act_fn()]
        for l in shared_layers:
            shared_extractor.append(nn.Linear(last_layer, l))
            shared_extractor.append(act_fn())
            last_layer = l
        extractor_out = last_layer
        policy = []
        for l in policy_layers:
            policy.append(nn.Linear(last_layer, l))
            policy.append(nn.ReLU())
            last_layer = l
        policy.pop()
        policy.append(nn.Softmax(dim=-1))
        value = []
        last_layer = extractor_out
        for l in value_layers:
            value.append(nn.Linear(last_layer, l))
            value.append(nn.ReLU())
            last_layer = l
        value.pop()
        self.shared_extractor = nn.Sequential(*shared_extractor)
        self.policy = nn.Sequential(*policy)
        self.value = nn.Sequential(*value)
    
    def forward(self, observation, prev_action, prev_reward):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        obs = observation.view(T*B, *img_shape)
        obs = obs.permute(0, 3, 1, 2).float() / 255.
        if self.noise_prob:
            obs, noise_idx = salt_and_pepper(obs,self.noise_prob)
        _,extractor_in,_ = self.encoder(obs)
        extractor_out = self.shared_extractor(extractor_in)
        act_dist = self.policy(extractor_out)
        value = self.value(extractor_out).squeeze(-1)
        act_dist, value = restore_leading_dims((act_dist, value), lead_dim, T, B)
        return act_dist, value