import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class Encoder(nn.Module):
	def __init__(self,zdim,channel_in,img_height):
		super().__init__()

		self.zdim = zdim
		self.h = img_height
		final_dim = (self.h//8)**2

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
		eps = Variable(torch.randn([bs, self.zdim]).cuda())
		z = eps * logsd.exp() + mu
		return z, logsd, mu


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

class VAE(nn.Module):

	def __init__(self,zdim,beta,channel_in=3,img_height=32):
		super().__init__()

		self.zdim = zdim
		self.encoder = Encoder(zdim=zdim,channel_in=channel_in,img_height=img_height)
		self.decoder = Decoder(zdim=zdim,channel_in=channel_in,img_height=img_height)
		self.beta = beta

	def forward(self, x, determ=False):
		z, zlogsd, zmu = self.encoder(x)
		if determ:
			reconx  = self.decoder(zmu)
		else:
			reconx = self.decoder(z)
		return z, zmu, zlogsd, reconx

	def loss(self, x, determ=False): 
		bs = x.shape[0]
		z, zmu, zlogsd, reconx  = self(x)
		zlogvar = zlogsd + zlogsd
		KLD = -0.5 * torch.sum(1 + zlogvar - zmu.pow(2) - zlogvar.exp()) / bs
		recon_loss = (x-reconx).pow(2).sum()/bs   
		return (recon_loss + self.beta * KLD), recon_loss , KLD

	def sample(self, n):
		# Sample a vector of n x zdim
		z = Variable(torch.randn([n, self.zdim]).cuda())
		return self.decoder(z)
