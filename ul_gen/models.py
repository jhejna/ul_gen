import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class ResBlock(nn.Module):

	def __init__(self, filters):
		super().__init__()
		self.layers = nn.Sequential(
			nn.ReLU(),
			nn.Conv2d(filters, filters, 3, padding=1),
			nn.ReLU(),
			nn.Conv2d(filters, filters, 3, padding=1),
		)

	def forward(self, x):
		return x + self.layers(x)


class Impala_CNN(nn.Module):

	def conv_sequence(self, in_filters, out_filters):
		return nn.Sequential(
			nn.Conv2d(in_filters, out_filters, 3, padding=1),
			nn.MaxPool2d(3, stride=2), # Not sure what the padding should be
			ResBlock(out_filters),
			ResBlock(out_filters)
		)

	def __init__(self, in_filters, filters, cnn_output):
		super().__init__()
		self.filters = filters
		self.layers = []
		last_filter = in_filters
		for f in self.filters:
			self.layers.append(conv_sequence(last_filter, f))
			last_filter = f
		self.layers = nn.Sequential(*self.layers)
		self.linear = nn.Linear(filters[-1]*32*32, cnn_output)

	def forward(self, x):
		out = self.layers(x)
		out = F.ReLU(out)
		flat = out.flatten()
		out = self.linear(out) # Add ReLU after?
		return out


class ProcgenPPOModel(nn.Module):

	def __init__(self, cnn_extractor=Impala_CNN, cnn_filters=[32, 64, 64], cnn_output=256, \
				policy_layers=[64, 64, 15], value_layers=[64, 64, 1],):
		super().__init__()
		self.cnn = cnn_extractor(3, cnn_filters, cnn_output)
		policy = []
		value = []
		last_layer = cnn_output
		for l in policy_layers:
			policy.append(nn.Linear(last_layer, l))
			policy.append(nn.ReLU())
			last_layer = l
		policy.pop()
		policy.append(nn.Softmax(dim=1))
		last_layer = cnn_output
		for l in value_layers:
			value.append(nn.Linear(last_layer, l))
			value.append(nn.ReLU())
			last_layer = l
		value.pop()
		self.policy = nn.Sequential(*policy)
		self.value = nn.Sequential(*value)

	def forward(self, observation, prev_action, prev_reward):
		features = self.cnn(observation)
		return self.policy(features), self.value(features)

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
