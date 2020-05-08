import torch
import torch.nn as nn
import numpy as np
import os
from torch.autograd import Variable
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, z_dim, img_shape=(3,64,64), hidden_dims=[256,128,64,32]):
        super(Generator, self).__init__()
        self.slope=.2
        self.h=64
        self.init_dim = (self.h // 2 ** len(hidden_dims))
        C = self.init_dim **2

        self.lin = self.classifier =nn.Sequential(
            nn.Linear(z_dim,hidden_dims[0]*C),
            nn.ReLU(True))
            # nn.Sigmoid())nn.Linear(z_dim, C * hidden_dims[0] )        
        self.main = nn.Sequential(
            # input dim: z_dim x 1 x 1
            nn.ConvTranspose2d(256, 256, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.slope, inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.slope, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.slope, inplace=True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(self.slope, inplace=True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh())
    
    def forward(self,z):
        z = self.lin(z).reshape(z.shape[0], -1, self.init_dim, self.init_dim)
        x = self.main(z)
        return x

    # def forward(self, z):
    #     bs = z.shape[0]
    #     z = self.relu(self.lin(z)).reshape(bs, -1, self.init_dim, self.init_dim)
    #     x = self.main(z)
    #     return x

class Discriminator(nn.Module):
    def __init__(self, zdim, img_shape=(3,64,64), lyrs=[16,32,64,128],dropout=.6 ):
        super(Discriminator, self).__init__()
        in_channels,self.h,_ = img_shape
        C = (self.h // 2 ** len(lyrs))**2
        odim = self.h*self.h*in_channels

        self.slope=.2
        self.main = nn.Sequential(
            nn.Conv2d(3, lyrs[0], 4, 2, 1),
            nn.BatchNorm2d(lyrs[0]),
            nn.LeakyReLU(self.slope, inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(lyrs[0], lyrs[1], 4, 2, 1),
            nn.BatchNorm2d(lyrs[1]),
            nn.LeakyReLU(self.slope, inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(lyrs[1], lyrs[2], 4, 2, 1),
            nn.BatchNorm2d(lyrs[2]),
            nn.LeakyReLU(self.slope, inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(lyrs[2], lyrs[3], 4, 2, 1),
            # nn.BatchNorm2d(256),
            # nn.LeakyReLU(self.slope, inplace=True),
            # nn.Dropout2d(p),

            # nn.Conv2d(256, 256, 1, 1),
            )

        self.classifier =nn.Sequential(
            nn.Linear(lyrs[-1]*C + zdim,1),
            # nn.ReLU(True),
            # nn.Linear(1024,1),
            nn.Sigmoid())

    def forward(self, z, x):
        x=self.main(x.reshape(-1,3,64,64)).reshape(x.shape[0],-1)
        return self.classifier(torch.cat((x,z),dim=1))


class Encoder(nn.Module):
    def __init__(self, zdim, img_shape=(3,64,64),hidden_dims=[32,64,128,256]):
        super(Encoder, self).__init__()
        self.slope=.2
        in_channels,self.h,_ = img_shape
        C = (self.h // 2 ** len(hidden_dims))**2
        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(self.slope, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.slope, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.slope, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.slope, inplace=True),

            nn.Conv2d(256, 256, 1, 1),            
            )
        self.lin = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(hidden_dims[-1]*C,zdim))

    def forward(self, x):
        x=self.main(x).reshape(x.shape[0],-1)
        return self.lin(x)

from torchvision.utils import save_image

class BiGAN(object):

    def __init__(self, zdim=128, shared_layers=[], policy_layers=[64, 64, 15,], value_layers=[64, 64, 1,],
            detach_encoder=False, detach_policy=False, detach_value=False, act_fn='relu' ):
        self.zdim=zdim
        self.detach_encoder=detach_encoder
        self.detach_value=detach_value
        self.detach_policy=detach_policy

        self.d = Discriminator(zdim).to(device)
        self.e = Encoder(zdim).to(device)
        self.g = Generator(zdim).to(device)
        # act_fn = {
        #     'relu' : lambda: nn.ReLU(),
        #     'tanh' : lambda: nn.Tanh()
        # }[act_fn]
        # last_layer = self.zdim
        # shared_extractor = [act_fn()]
        # for l in shared_layers:
        #     shared_extractor.append(nn.Linear(last_layer, l))
        #     shared_extractor.append(act_fn())
        #     last_layer = l
        # policy = []
        # extractor_out = last_layer
        # for l in policy_layers:
        #     policy.append(nn.Linear(last_layer, l))
        #     policy.append(nn.ReLU())
        #     last_layer = l
        # policy.pop()
        # policy.append(nn.Softmax(dim=-1))
        # value = []
        # last_layer = extractor_out
        # for l in value_layers:
        #     value.append(nn.Linear(last_layer, l))
        #     value.append(nn.ReLU())
        #     last_layer = l
        # value.pop()
        # self.shared_extractor = nn.Sequential(*shared_extractor).to(device)
        # self.policy = nn.Sequential(*policy).to(device)
        # self.value = nn.Sequential(*value).to(device)

    def forward(self, observation):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        obs = observation.view(T*B, *img_shape)
        obs = (obs.permute(0, 3, 1, 2).float() / 255.)*2 - 1

        z_fake = torch.normal(torch.zeros(obs.shape[0], self.zdim), torch.ones(obs.shape[0], self.zdim)).to(device)
        z_real = self.e(obs).reshape(obs.shape[0], self.zdim)
        x_fake = self.g(z_fake).reshape(obs.shape[0], -1)
        x_real = obs.view(obs.shape[0], -1)

        # extractor_in=z_fake
        # if self.detach_encoder:
        #     extractor_in = extractor_in.detach()
        # extractor_out = self.shared_extractor(extractor_in)

        # if self.detach_policy:
        #     policy_in = extractor_out.detach()
        # else:
        #     policy_in = extractor_out
        # if self.detach_value:
        #     value_in = extractor_out.detach()
        # else:
        #     value_in = extractor_out
        
        # act_dist = self.policy(policy_in)
        # value = self.value(value_in).squeeze(-1)
        act_dist, value = 0, 0
        label_real = self.d(z_real, x_real)
        label_fake = self.d(z_fake, x_fake)
        latent, reconstruction = restore_leading_dims((z_fake, x_fake), lead_dim, T, B)
        return act_dist, value, latent, reconstruction, label_real, label_fake

    def loss(self, obs, d_loss=True):
        _, _, _, _, label_real, label_fake = self.forward(obs)
        if d_loss:
            loss = - (label_real + 1e-6).log().mean() - (1 - label_fake - 1e-6).log().mean()
        else:
            loss = -(1- label_real - 1e-6).log().mean() - (1e-6+label_fake).log().mean()

        return loss

    def sample(self, n):
        self.g.eval()
        with torch.no_grad():
            z = torch.randn(n, self.zdim).to(device)
            samples = self.g(z).reshape(-1, 3, 64, 64)
        self.g.train()
        return (samples+1)/2

   
    def get_reconstructions(self, observation):

        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        obs = observation.view(T*B, *img_shape)[:10]
        obs =(obs.permute(0, 3, 1, 2).float() / 255.)*2 - 1

        z = self.e(obs)
        recons = self.g(z).reshape(-1, 3, 64, 64)

        return (torch.cat((obs,recons),dim=0)+1)/2

    def save_images(self,path, x, itr):
        samples = self.sample(64)
        recons = self.get_reconstructions(x)
        save_image(torch.Tensor(samples.detach().cpu()), os.path.join(path, 'samples_' + str(itr) +'.png'), nrow=8)
        save_image(torch.Tensor(recons.detach().cpu()), os.path.join(path, 'recon_' + str(itr) +'.png'), nrow=10)


    def save_models(self, path, itr):
        torch.save(self.g.state_dict(), path+"g_%d"%itr)
        torch.save(self.d.state_dict(), path+"d_%d"% itr)
        torch.save(self.e.state_dict(), path+"e_%d"% itr)

