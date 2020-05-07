import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.autograd import Variable
from torchvision.utils import save_image

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.buffer import buffer_to
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            z = mu
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
        self.init_dim = (self.h // 2 ** len(hidden_dims))
        C = self.init_dim **2

        self.lin = nn.Linear(zdim, C * hidden_dims[0] )
        self.relu = nn.ReLU(True)
        modules=[nn.ConvTranspose2d(hidden_dims[0],hidden_dims[1], 4, 2, 1)] 
        in_channels = hidden_dims[1]    

        if arch_type == 0:
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
            nn.ConvTranspose2d(in_channels, channel_out, 4, 2, 1),
            nn.Sigmoid()))
        self.main = nn.Sequential(*modules)

    def forward(self, z):
        bs = z.shape[0]
        z = self.relu(self.lin(z)).reshape(bs, -1, self.init_dim, self.init_dim)
        x = self.main(z)
        return x

class RadVaePolicy(nn.Module):

    def __init__(self, zdim, img_shape=(3,64,64), shared_layers=[], 
                        policy_layers=[64, 64, 15,], value_layers=[64, 64, 1,], 
                        encoder_layers=[32, 64, 128, 256], decoder_layers=[256, 128, 64, 32],
                        act_fn='relu', deterministic=False,
                        detach_vae=False, detach_value=False,
                        detach_policy=False, arch_type=0,
                        rae=False):

        """
        arch_type: 0 for Conv2d-ReLU; 1 for Conv2d-BN-LeakyReLU
        noise_prob: Makes noise_prob % of images 0 or 1 (salt-and-pepper) before encoding
        noise_weight: If salt-and-pepper, weight on noise L2 loss
        no_noise_weight: If salt-and-pepper, weight on non-noise L2 loss
        """
        super().__init__()
        self.zdim = zdim
        self.rae = rae
        self.detach_value = detach_value
        self.detach_policy = detach_policy
        self.detach_vae = detach_vae
        self.deterministic = deterministic
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

    def forward(self, observation, prev_action=None, prev_reward=None):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        obs = observation.view(T*B, *img_shape)

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
        return act_dist, value, latent, reconstruction

    def log_images(self, obs, savepath, itr,n=100):
        self.eval()
        zs = torch.randn(n, self.zdim).to(device)
        samples = self.decoder(zs)
        obs = obs.reshape(-1, 64, 64, 3)[:10]
        _, _, latent, reconstruction,_ = self.forward(obs)
        obs = obs.float().to(device) / 255.
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

class BaselinePolicy(nn.Module):
    def __init__(self, img_shape=(3, 64, 64), shared_layers=[], 
                        zdim=128, arch_type=0,encoder_layers=[32, 64, 128, 256],
                        policy_layers=[64, 64, 15,], value_layers=[64, 64, 1,], act_fn='relu',
                        ):

        super().__init__()

        self.encoder =  Encoder(zdim,img_shape,arch_type,hidden_dims=encoder_layers)
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
        _,extractor_in,_ = self.encoder(obs)
        extractor_out = self.shared_extractor(extractor_in)
        act_dist = self.policy(extractor_out)
        value = self.value(extractor_out).squeeze(-1)
        act_dist, value = restore_leading_dims((act_dist, value), lead_dim, T, B)
        return act_dist, value