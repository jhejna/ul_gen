import os
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import numpy as np
import random
import time
from torchvision.utils import save_image

from ul_gen.models import Reshape
from ul_gen.misc.constrained_vae import ConstrainedVAE

if __name__ == '__main__':

    ##### Hyper Parameters #####
    img_dim = 64
    img_channels = 3
    epochs = 10
    batch_size = 64
    lr = 1e-3
    sim_loss_coef = 6
    z_dim = 50
    k_dim = 40
    beta = 1.1
    save_freq = 5
    noise_prob = 0
    deterministic = True
    savepath = 'constrained_vae/kl_divergence_64'
    checkpoint = 'model-100'
    device = torch.device("cuda:2")

    ############################
    os.makedirs(savepath, exist_ok=True)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((64,64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Lambda(lambda x: 2*x-1)
    ])

    dataset = torchvision.datasets.CelebA('./data/', split='train', target_type='attr', download=True, transform=transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torchvision.datasets.CelebA('./data/', split='test', target_type='attr', download=True, transform=transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


    model = ConstrainedVAE(z_dim, k_dim, img_dim=img_dim, img_channels=img_channels).to(device)
    state_dict = torch.load(os.path.join(savepath, checkpoint))
    model.load_state_dict(state_dict)
    model.eval()
    print(f'Loaded Model: {savepath}/{checkpoint}')


    linear = nn.Linear(z_dim, k_dim).to(device)
    optimizer = torch.optim.Adam(linear.parameters(), lr=lr)

    for i in range(epochs):
        start = time.time()
        for batch, attr in loader:
            x = batch.float().to(device)
            attr = attr.float().to(device)
            mu, logvar = model.encoder(x)
            if deterministic:
                z = mu
            else:
                z = torch.exp(0.5*log_var) * torch.randn_like(mu) + mu
            z = z.detach()
            y = linear(z)
            loss = F.binary_cross_entropy_with_logits(y, attr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_acc = 0
        total_num = 0
        with torch.no_grad():
            for batch, attr in test_loader:
                x = batch.float().to(device)
                if type(attr) == tuple:
                    print(attr)
                attr = attr.float().to(device)
                mu, logvar = model.encoder(x)
                if deterministic:
                    z = mu
                else:
                    z = torch.exp(0.5*log_var) * torch.randn_like(mu) + mu
                y = linear(z)
                y = F.sigmoid(y)
                acc = (y * attr).sum(dim=1) / torch.sum(attr, dim=1)
                total_acc += acc.sum()
                total_num += len(batch)
        print(f"Epoch {i}: Test Accuracy {total_acc/total_num} Time {time.time() - start}")

        if (i + 1) % save_freq:
            torch.save(linear.state_dict(), os.path.join(savepath, f'linear-{i+1}'))

    num_samples = 10
    attr_idx = 4
    with torch.no_grad():
        attr_vec = torch.eye(k_dim, device=device)[attr_idx].repeat(num_samples, 1)
        # print(attr_vec)

        z = torch.cat((attr_vec, torch.rand((num_samples, z_dim-k_dim), device=device)), dim=1)
        samples = model.decoder(z)
        samples = (samples + 1)/2
        save_image(samples.cpu(), os.path.join(savepath, f'samples_{dataset.attr_names[attr_idx]}_{checkpoint}.png'), nrow=5)
        print(f'Saved samples: {dataset.attr_names[attr_idx]}')


