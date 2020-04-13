from models import VAE
from utils import * 
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import argparse
import json
from tensorboard_logger import configure
from tensorboard_logger import log_value


parser = argparse.ArgumentParser()


#Data Hyperparameters
parser.add_argument("--savepath",type=str,default="/home/karam/Downloads/ul_gen")
parser.add_argument("--datapath",type=str,default="/home/karam/Downloads/procgen.npy")

# VAE Hyperparameters
parser.add_argument("--zdim", type=int, default=100)
parser.add_argument("--bs", type=int, default=128)
parser.add_argument("--vae_beta", type=int, default=1,
					help="beta weight term in vae loss")
parser.add_argument("--lr_vae", type=float, default=0.0001)
parser.add_argument("--n_epochs_full_vae", type=int, default=100)
parser.add_argument("--vae_path", help="path to load trained vae from",type=str,
					default="")

# Policy Hyperparameters
parser.add_argument("--env", type=str, default="block",
					help="block")
parser.add_argument("--n_epochs_policy", type=int, default=100)

args = parser.parse_args()



# VAE
vae = VAE(zdim=args.zdim,beta=args.vae_beta, img_height=64).cuda()
if args.vae_path:
	vae.load_state_dict(torch.load(args.vae_path))



# Configure tensorboard
configure('%s/var_log' % args.savepath, flush_secs=5)

# Load Data
npy_data = np.load(args.datapath,allow_pickle=True)[:,0]/255
data_loader = data.DataLoader(npy_data,batch_size=args.bs, shuffle=True)
n_batch = len(data_loader)
print("Num. batches: ",n_batch)
if args.n_epochs_full_vae > 0:
	
	vae_opt = optim.Adam(vae.parameters(),lr=args.lr_vae)
	
	for epoch in range(args.n_epochs_full_vae):
		for it,dta in enumerate(data_loader):
			if dta.shape[0] == args.bs:
				x = dta.float().cuda()
				vae_opt.zero_grad()
				loss, recon_loss, kld = vae.loss(x)
				loss.backward()
				vae_opt.step()

				if it % (n_batch//5) == 0:
					log_value('vae_loss', loss, it + n_batch * epoch)
					log_value('kl_loss', kld, it + n_batch * epoch)
					log_value('kl_loss', kld, it + n_batch * epoch)
		
		if epoch % 5 == 0:	
			print("Train Loss: %.3f   Recon: %.3f   KLD: %.3f"%(loss.item(), recon_loss.item(), kld.item()))
			torch.save(vae.state_dict(), '%s/vae-%d' % (args.savepath, epoch))
			log_images_vae(x[:20], vae, args.savepath, epoch)






	


















