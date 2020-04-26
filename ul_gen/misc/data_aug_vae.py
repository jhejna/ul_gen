import os
import torch
import torchvision
import numpy as np
import random
from PIL import Image
from torch import nn
from torchvision.transforms.functional import pad, resize, to_tensor, normalize, rotate
from torchvision.utils import save_image

class PairedAug(object):

    def __init__(self, output_size, resize=None, rotate=None):
        self.output_size = output_size
        self.resize = resize
        self.rotate = rotate
    
    def aug_img(self, img):
        if not self.rotate is None:
            angle = random.uniform(*self.rotate)
            img = rotate(img, angle, fill=(0,))
        if not self.resize is None:
            w, h = img.size
            # assert w == h, "Image must be square"
            rescale = int(w * random.uniform(*self.resize))
            # img = img.resize((rescale, rescale), Image.BILINEAR)
            img = resize(img, rescale)
        
        w, h = img.size
        left_pad = random.randint(0, self.output_size - w)
        top_pad = random.randint(0, self.output_size - h)
        right_pad = self.output_size - w - left_pad
        bottom_pad = self.output_size - h - top_pad
        img = pad(img, (left_pad, top_pad, right_pad, bottom_pad), fill=0)

        return img

    def __call__(self, sample):
        aug = sample.copy()
        orig = 2*to_tensor(self.aug_img(sample)) - 1
        aug = 2*to_tensor(self.aug_img(aug)) - 1

        return {'orig': orig, 'aug': aug}

class Reshape(torch.nn.Module):
  def __init__(self, output_shape):
    super(Reshape, self).__init__()
    self.output_shape = output_shape

  def forward(self, x):
    return x.view(*((len(x),) + self.output_shape))

class PrintNode(torch.nn.Module):
  def __init__(self, identifier="print:"):
    super(PrintNode, self).__init__()
    self.identifier = identifier

  def forward(self, x):
    print(self.identifier, x.shape)
    return x



class AugVAE(torch.nn.Module):

  def __init__(self, img_dim=64, img_channels=3, z_dim=32):
    super(AugVAE, self).__init__()
    self.img_dim = img_dim
    self.z_dim = z_dim
    self.k_dim = k_dim
    self.img_channels = img_channels

    final_feature_dim = img_dim // 8
    self.encoder_net = torch.nn.Sequential(torch.nn.Conv2d(self.img_channels, 32, 3, 1 ,1),
                                           torch.nn.ReLU(),
                                           torch.nn.Conv2d(32, 64, 3, 2, 1),
                                           torch.nn.ReLU(),
                                           torch.nn.Conv2d(64, 128, 3, 2, 1),
                                           torch.nn.ReLU(),
                                           torch.nn.Conv2d(128, 256, 3, 2, 1),
                                           torch.nn.ReLU(),
                                           torch.nn.Flatten(),
                                           torch.nn.Linear(final_feature_dim*final_feature_dim*256, 2*self.z_dim))
    
    self.decoder_net = torch.nn.Sequential(torch.nn.Linear(self.z_dim, final_feature_dim*final_feature_dim*128),
                                           Reshape((128, final_feature_dim, final_feature_dim)),
                                           torch.nn.ConvTranspose2d(128, 128, 4, 2, 1),
                                           torch.nn.ReLU(),
                                           torch.nn.ConvTranspose2d(128, 64, 4, 2, 1),
                                           torch.nn.ReLU(),
                                           torch.nn.ConvTranspose2d(64, 32, 4, 2, 1),
                                           torch.nn.ReLU(),
                                           torch.nn.Conv2d(32, self.img_channels, 3, 1, 1),
                                           torch.nn.Tanh())

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
    return x_hat, mu, log_var


##### Hyper Parameters #####
img_dim = 48
img_channels = 1
epochs = 50
batch_size = 96
lr = 5e-4
sim_loss_coef = 1.0
z_dim = 36
k_dim = 28
beta = 1.1
scale_range = (0.9, 0.91)
rot_range = (-75, 75)
save_freq = 10
savepath = 'vae_aug_test_rot'
############################
os.makedirs(savepath, exist_ok=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mnist_data = torchvision.datasets.MNIST('~/.pytorch/mnist', train=True, download=True, transform=PairedAug(img_dim, resize=scale_range, rotate=rot_range))
loader = torch.utils.data.DataLoader(mnist_data, batch_size=batch_size, shuffle=True)

# Debug: print aug pairs next to each other.
# from matplotlib import pyplot as plt
# sample, _ = next(iter(loader))
# plt.imshow(sample['orig'][0][0])
# plt.show()
# plt.imshow(sample['aug'][0][0])
# plt.show()
# exit()

model = AugVAE(img_dim=img_dim, img_channels=img_channels, z_dim=z_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    for batch, _ in loader:
        optimizer.zero_grad()
        # Concatenate to feed all the data through
        x = torch.cat((batch['orig'], batch['aug']), dim=0).to(device)
        x_hat, mu, log_var = model(x)
        kl_loss = torch.sum(-0.5*(1 + log_var - mu.pow(2) - log_var.exp())) / len(x) # Divide by batch size
        recon_loss = torch.sum((x - x_hat).pow(2)) / len(x)
        # Compute the similarity loss
        mu_orig, mu_aug = torch.chunk(mu, 2, dim=0)
        log_var_orig, log_var_aug = torch.chunk(log_var, 2, dim=0)
        mu_orig, mu_aug = mu_orig[:, :k_dim], mu_aug[:, :k_dim]
        log_var_orig, log_var_aug = log_var_orig[:, :k_dim], log_var_aug[:, :k_dim]
        # KL divergence between original and augmented.
        sim_loss = torch.sum(log_var_aug - log_var_orig + 0.5*(log_var_orig.exp() + (mu_orig - mu_aug).pow(2))/log_var_aug.exp() - 0.5)/ (len(x) // 2)
        # sim_loss = torch.sum(0.5*(mu_orig - mu_aug).pow(2)) / len(x)

        loss = recon_loss + beta * kl_loss + sim_loss_coef * sim_loss
        loss.backward()
        optimizer.step()

    print('Epoch %d Recon Loss: %.3f KL Loss: %.3f Sim Loss: %.3f' % (epoch+1, recon_loss.item(), kl_loss.item(), sim_loss.item()))
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
        x_orig, x_aug = torch.chunk(x, 2, dim=0)
        x_orig, x_aug  = x_orig[:n_interp], x_aug[:n_interp]
        z_orig, _ = model.encoder(x_orig)
        z_aug, _ = model.encoder(x_aug)
        diff_vec = z_aug - z_orig
        # Set the first k components of diff vec to be zero, so we only vary along aug components
        diff_vec[:, :k_dim] = 0

        interpolations = []
        interpolations.append(x_orig)
        for i in range(1, 9):
            interpolations.append(model.decoder(z_orig + 0.111111*i*diff_vec))
        interpolations.append(x_aug)
        out_interp = torch.zeros(n_interp*10, img_channels, img_dim, img_dim)
        for i in range(10):
            for j in range(n_interp):
                out_interp[10*j + i, :, :, :] = interpolations[i][j, :, :, :]

        interpolations = (out_interp + 1)/2
        save_image(interpolations.detach().cpu(), os.path.join(savepath, 'interp_' + str(epoch+1) +'.png'), nrow=10)
            
        torch.save(model.state_dict(), '%s/aug-vae-%d' % (savepath, epoch+1))
        model.train()


