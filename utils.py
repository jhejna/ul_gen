import torch
import os.path
import numpy as np
from torchvision.utils import save_image


def log_images_vae(x, vae, path, epoch):
	vae.eval()
	samples = vae.sample(100)
	recon = vae(x)[-1]
	save_image(torch.Tensor(samples.detach().cpu()), os.path.join(path, 'sample_' + str(epoch) +'.png'), nrow=10)
	save_image(torch.Tensor(recon.detach().cpu()), os.path.join(path, 'recon_' + str(epoch) +'.png'), nrow=10)
	vae.train()