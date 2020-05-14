import os
import torch
import torchvision
import numpy as np
import random
import time
from PIL import Image
from torch import nn
from torchvision.transforms.functional import pad, resize, to_tensor, normalize, rotate
from torchvision.utils import save_image

from ul_gen.models import Reshape
from ul_gen.models.vae import salt_and_pepper


class PairedAug(object):

    def __init__(self, noise_prob):
        self.noise_prob = noise_prob
    
    def aug_img(self, img):
        noisy, idx = salt_and_pepper(img, self.noise_prob)
        return noisy.squeeze(0)

    def __call__(self, orig):
        aug = self.aug_img(orig.clone())
        assert aug.shape == orig.shape, (f"{aug.shape}  {orig.shape}")
        return {'orig': orig*2 - 1, 'aug': aug*2 - 1}


class MappedCelebA(torchvision.datasets.CelebA):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapping = {k:[] for k in range(len(self.attr_names))}
        for index in range(len(self.attr)):
            for k in range(len(self.attr_names)):
                if self.attr[index][k] == 1:
                    self.mapping[k].append(index)


class ConstrainedVAE(torch.nn.Module):

  def __init__(self, img_dim=64, img_channels=3, z_dim=32):
    super().__init__()
    self.img_dim = img_dim
    self.z_dim = z_dim
    self.k_dim = k_dim
    self.img_channels = img_channels

    final_feature_dim = img_dim // 8
    self.encoder_net = torch.nn.Sequential(nn.Conv2d(self.img_channels, 32, 3, 1 ,1),
                                           nn.ReLU(),
                                           nn.Conv2d(32, 64, 3, 2, 1),
                                           nn.ReLU(),
                                           nn.Conv2d(64, 128, 3, 2, 1),
                                           nn.ReLU(),
                                           nn.Conv2d(128, 128, 3, 2, 1),
                                           nn.ReLU(),
                                           nn.Flatten(),
                                           nn.Linear(final_feature_dim*final_feature_dim*128, 2*self.z_dim),
                                        )
    
    self.decoder_net = torch.nn.Sequential(nn.Linear(self.z_dim, final_feature_dim*final_feature_dim*128),
                                           Reshape((128, final_feature_dim, final_feature_dim)),
                                           nn.ConvTranspose2d(128, 128, 4, 2, 1),
                                           nn.ReLU(),
                                           nn.ConvTranspose2d(128, 64, 4, 2, 1),
                                           nn.ReLU(),
                                           nn.ConvTranspose2d(64, 32, 4, 2, 1),
                                           nn.ReLU(),
                                           nn.Conv2d(32, self.img_channels, 3, 1, 1),
                                           nn.Tanh())

  def encoder(self, x):
    mu, log_var = torch.chunk(self.encoder_net(x), 2, dim=1)
    return mu, log_var
    
    # kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), axis=1)
    # if deterministic:
    #     z = mu
    # else:
    #     z = torch.exp(0.5*log_var) * torch.randn_like(mu) + mu
    # return z, kl_loss

  def decoder(self, z):
    return self.decoder_net(z)
    
  def forward(self, x, deterministic=False):
    mu, log_var = self.encoder(x)
    if deterministic:
        z = mu
    else:
        z = torch.exp(0.5*log_var) * torch.randn_like(mu) + mu
    x_hat = self.decoder(z)
    return x_hat, z, mu, log_var


##### Hyper Parameters #####
img_dim = 64
img_channels = 3
epochs = 50
batch_size = 64
lr = 5e-4
sim_loss_coef = 6
z_dim = 50
k_dim = 40
beta = 1.1
save_freq = 5
noise_prob = 0
savepath = 'constrained_vae/kl_divergence_64/'
############################
os.makedirs(savepath, exist_ok=True)
device = torch.device("cuda:3")

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((64,64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: 2*x-1)
])

dataset = torchvision.datasets.CelebA('./data/', split='train', target_type='attr', download=True, transform=transforms)

loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Debug: print aug pairs next to each other.
# from matplotlib import pyplot as plt
# sample, target = next(iter(loader))
# print(sample['orig'].shape)
# print(sample['aug'].shape)
# print(target.shape)
# test = sample[0].permute(1,2,0)*.5 + .5
# aug =  sample['aug'][0].permute(1,2,0)*.5 + .5
# print(test)
# print(target[0])
# plt.imsave('test_celeb.png', (test*255).byte().numpy())
# plt.imsave('aug_celeb.png', (aug*255).byte().numpy())
# exit()

model = ConstrainedVAE(img_dim=img_dim, img_channels=img_channels, z_dim=z_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    start = time.time()
    for batch, attr in loader:
        optimizer.zero_grad()
        # Concatenate to feed all the data through
        # x = torch.cat((batch['orig'], batch['aug']), dim=0).to(device)
        x = batch.float().to(device)
        attr = attr.to(device)
        x_hat, z, mu, log_var = model(x)
        kl_loss = torch.sum(-0.5*(1 + log_var - mu.pow(2) - log_var.exp())) / len(x) # Divide by batch size
        recon_loss = torch.sum((x - x_hat).pow(2)) / len(x)

        bs, c, h, w = x.shape
        paired_mu = mu.view(bs//2, 2, -1).permute(0, 2, 1)  # 32 x 50 x 2
        paired_logvar = log_var.view(bs//2, 2, -1).permute(0, 2, 1)  # 32 x 50 x 2
        paired_attr = attr.view(bs//2, 2, -1).permute(0, 2, 1)  # 32 x 40 x 2
        # Get common attributes among pairs
        common = torch.bitwise_and(paired_attr[:, :, 0], paired_attr[:, :, 1])  # 32 x 40 
        common = torch.cat((common, torch.zeros((bs//2, z_dim - k_dim), dtype=torch.long, device=device)), dim=1) # 32 x 50
        common = common.unsqueeze(2)
        # Mask latents by the attributes we will compare  # 32 x 50 x 1
        masked_mu = paired_mu * common
        masked_logvar = paired_logvar * common

        # L2 loss among common latents of pairs
        # diff_z = (masked_z[:, :, 0] - masked_z[:, :, 1])**2
        # sim_loss = torch.sum(diff_z) / len(diff_z)

        # mu_orig, mu_aug = torch.chunk(mu, 2, dim=0)
        # log_var_orig, log_var_aug = torch.chunk(log_var, 2, dim=0)
        # mu_orig, mu_aug = mu_orig[:, :k_dim], mu_aug[:, :k_dim]
        # log_var_orig, log_var_aug = log_var_orig[:, :k_dim], log_var_aug[:, :k_dim]
        mu_orig, mu_aug = masked_mu[:,:,0], masked_mu[:,:,1]
        logvar_orig, logvar_aug = masked_logvar[:,:,0], masked_logvar[:,:,1]
        
        # # KL divergence between original and augmented.
        sim_loss = 0.5* torch.sum(logvar_aug - logvar_orig + (logvar_orig.exp() + (mu_orig - mu_aug).pow(2))/logvar_aug.exp() - 1) / len(masked_mu)
        # sim_loss = torch.sum(0.5*(mu_orig - mu_aug).pow(2)) / len(x)

        loss = recon_loss + beta * kl_loss + sim_loss_coef * sim_loss
        loss.backward()
        optimizer.step()

    print('Epoch %d Recon Loss: %.3f KL Loss: %.3f Sim Loss: %.3f Time: %.3f' % (epoch+1, recon_loss.item(), kl_loss.item(), sim_loss.item(), time.time() - start))

    if (epoch + 1) % save_freq == 0:
        # Save reconstructions and samples:
        model.eval()
        recon = torch.cat((x[:8], x_hat[:8]),dim=0) 
        recon = (recon + 1)/2
        save_image(recon.detach().cpu(), os.path.join(savepath, 'recon_' + str(epoch+1) +'.png'), nrow=8)

        zs = torch.randn(16, z_dim).to(device)
        samples = model.decoder(zs)
        samples = (samples + 1)/2
        save_image(samples.detach().cpu(), os.path.join(savepath, 'samples_' + str(epoch+1) +'.png'), nrow=8)

        # Now, save the interpolations.
        n_interp = 8
        interp_length = 8
        x_orig, x_aug = torch.chunk(x, 2, dim=0)
        x_orig, x_aug  = x_orig[:n_interp], x_aug[:n_interp]
        z_orig, _ = model.encoder(x_orig)
        z_aug, _ = model.encoder(x_aug)
        diff_vec = z_aug - z_orig

        interpolations = []
        interpolations.append(x_orig)
        for i in range(1, interp_length-1):
            interpolations.append(model.decoder(z_orig + i/(interp_length-1)*diff_vec))
        interpolations.append(x_aug)
        out_interp = torch.zeros(n_interp*interp_length, img_channels, img_dim, img_dim)
        for i in range(interp_length):
            for j in range(n_interp):
                out_interp[interp_length*j + i, :, :, :] = interpolations[i][j, :, :, :]
        out_interp = (out_interp + 1)/2
        save_image(out_interp.detach().cpu(), os.path.join(savepath, 'interp_reg_' + str(epoch+1) +'.png'), nrow=interp_length)


        # Set the first k components of diff vec to be zero, so we only vary along aug 
        
        # Set the last z - k components to zero, so we only vary across attributes
        diff_vec[:, k_dim:] = 0

        interpolations = []
        interpolations.append(x_orig)
        for i in range(1, interp_length-1):
            interpolations.append(model.decoder(z_orig + i/(interp_length-1)*diff_vec))
        interpolations.append(x_aug)
        out_interp = torch.zeros(n_interp*interp_length, img_channels, img_dim, img_dim)
        for i in range(interp_length):
            for j in range(n_interp):
                out_interp[interp_length*j + i, :, :, :] = interpolations[i][j, :, :, :]
        out_interp = (out_interp + 1)/2
        save_image(out_interp.detach().cpu(), os.path.join(savepath, 'interp_constrained_' + str(epoch+1) +'.png'), nrow=interp_length) 
            
        torch.save(model.state_dict(), '%s/model-%d' % (savepath, epoch+1))
        model.train()


