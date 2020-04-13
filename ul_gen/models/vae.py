import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.buffer import buffer_to

class Encoder(nn.Module):
    def __init__(self,zdim,channel_in,img_height):
        super().__init__()

        self.zdim = zdim
        self.h = img_height
        final_dim = (self.h//8)**2

        self.use_cuda = torch.cuda.is_available()

        self.main = nn.Sequential(
            nn.Conv2d(channel_in, 32, 1), # h x h
            nn.ReLU(True),

            nn.Conv2d(32, 64, 3, 2, 1), # h/2 x h/2
            nn.ReLU(True), 

            nn.Conv2d(64, 128, 3, 2, 1), # h/4 x h/4
            nn.ReLU(True),

            nn.Conv2d(128, 256, 3, 2, 1), # h/8 x h/8
            nn.ReLU(True)
            )

        self.mu = nn.Linear(final_dim * 256, zdim)
        self.logsd = nn.Linear(final_dim * 256, zdim)

    def forward(self,x):
        bs = x.shape[0]
        x = self.main(x).reshape(bs,-1)
        mu = self.mu(x)
        logsd = self.logsd(x)
        if self.use_cuda: 
            eps = buffer_to(Variable(torch.randn([bs, self.zdim])), x.device)
        else:
            eps = Variable(torch.randn([bs, self.zdim]))
        z = eps * logsd.exp() + mu
        return z, mu, logsd

class Decoder(nn.Module):
    def __init__(self, zdim, channel_in, img_height):
        super().__init__()

        self.h = img_height
        self.odim = self.h*self.h*channel_in
        self.init_dim = (self.h//8)

        self.lin = nn.Linear(zdim, self.init_dim * self.init_dim * 128)
        self.relu = nn.ReLU(True)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1), # h/4 x h/4
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # h/2 x h/2
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # h x h
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, 1, 1), # h x h
            nn.Sigmoid()
        )

    def forward(self, z):
        bs = z.shape[0]
        z = self.relu(self.lin(z)).reshape(bs, 128, self.init_dim, self.init_dim)
        x = self.main(z)
        return x

class VaePolicy(nn.Module):

    def __init__(self, zdim, channel_in=3, img_height=64, shared_layers=[128,], 
                        policy_layers=[15,], value_layers=[1,], act_fn='relu', deterministic=False,
                        detach_vae=False):
        super().__init__()
        self.zdim = zdim
        self.detach_vae = detach_vae
        self.deterministic = deterministic
        self.encoder = Encoder(zdim=zdim,channel_in=channel_in,img_height=img_height)
        self.decoder = Decoder(zdim=zdim,channel_in=channel_in,img_height=img_height)
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

    def forward(self, observation, prev_action, prev_reward):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        obs = observation.view(T*B, *img_shape)
        obs = obs.permute(0, 3, 1, 2).float() / 255.

        z, mu, logsd = self.encoder(obs)
        extractor_in = mu if self.deterministic else z
        reconstruction = self.decoder(extractor_in)
        if self.detach_vae:
            extractor_in = extractor_in.detach()
        extractor_out = self.shared_extractor(extractor_in)
        act_dist = self.policy(extractor_out)
        value = self.value(extractor_out).squeeze(-1)
        latent = torch.cat((mu, logsd), dim=1)
        act_dist, value, latent, reconstruction = restore_leading_dims((act_dist, value, latent, reconstruction), lead_dim, T, B)
        return act_dist, value, latent, reconstruction


class BaselinePolicy(nn.Module):
    def __init__(self, channel_in=3, img_height=64, shared_layers=[128,], 
                        policy_layers=[15,], value_layers=[1,], act_fn='relu',):

        super().__init__()

        self.h = img_height
        final_dim = (self.h//8)**2
        act_fn = {
            'relu' : lambda: nn.ReLU(),
            'tanh' : lambda: nn.Tanh()
        }[act_fn]

        self.use_cuda = torch.cuda.is_available()

        self.main = nn.Sequential(
            nn.Conv2d(channel_in, 32, 1), # h x h
            nn.ReLU(True),

            nn.Conv2d(32, 64, 3, 2, 1), # h/2 x h/2
            nn.ReLU(True), 

            nn.Conv2d(64, 128, 3, 2, 1), # h/4 x h/4
            nn.ReLU(True),

            nn.Conv2d(128, 256, 3, 2, 1), # h/8 x h/8
            nn.ReLU(True),
            
            nn.Flatten(),
            nn.Linear(final_dim * 256, 256) # finally convert to FC.
            )
        last_layer = 256
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
        extractor_in = self.main(obs)
        extractor_out = self.shared_extractor(extractor_in)
        act_dist = self.policy(extractor_out)
        value = self.value(extractor_out).squeeze(-1)
        act_dist, value = restore_leading_dims((act_dist, value), lead_dim, T, B)
        return act_dist, value
