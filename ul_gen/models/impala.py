import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


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
			nn.MaxPool2d(3, stride=2, padding=1), # Not sure what the padding should be
			ResBlock(out_filters),
			ResBlock(out_filters)
		)

	def __init__(self, in_filters, filters, cnn_output):
		super().__init__()
		self.filters = filters
		self.layers = []
		last_filter = in_filters
		for f in self.filters:
			self.layers.append(self.conv_sequence(last_filter, f))
			last_filter = f
		self.layers = nn.Sequential(*self.layers)
		self.linear = nn.Linear(filters[-1]*8*8, cnn_output) # Hard coded final dimension

	def forward(self, x):
		out = self.layers(x)
		out = F.relu(out)
		out = out.view(len(out), -1) # Flatten for linear
		out = self.linear(out) # Add ReLU after?
		return out


class ProcgenPPOModel(nn.Module):

	def __init__(self, channel_in=3, cnn_extractor=Impala_CNN, cnn_filters=[32, 64, 64], cnn_output=256, \
				policy_layers=[15], value_layers=[1],):
		super().__init__()
		self.cnn = cnn_extractor(channel_in, cnn_filters, cnn_output)
		policy = []
		value = []
		last_layer = cnn_output
		for l in policy_layers:
			policy.append(nn.Linear(last_layer, l))
			policy.append(nn.ReLU())
			last_layer = l
		policy.pop()
		policy.append(nn.Softmax(dim=-1))
		last_layer = cnn_output
		for l in value_layers:
			value.append(nn.Linear(last_layer, l))
			value.append(nn.ReLU())
			last_layer = l
		value.pop()
		self.policy = nn.Sequential(*policy)
		self.value = nn.Sequential(*value)

	def forward(self, observation, prev_action, prev_reward):
		lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
		obs = observation.view(T*B, *img_shape)
		obs = obs.permute(0, 3, 1, 2).float() / 255.
		features = self.cnn(obs)
		policy = self.policy(features)
		value = self.value(features).squeeze(-1)
		policy, value = restore_leading_dims((policy, value), lead_dim, T, B)
		return policy, value