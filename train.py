from models import VAE
from utils import * 
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import argparse
from tensorboard_logger import configure
from tensorboard_logger import log_value


parser = argparse.ArgumentParser()


#Data Hyperparameters
parser.add_argument("--savepath",type=str,default="")
parser.add_argument("--datapath",type=str,default="")

# VAE Hyperparameters
parser.add_argument("--zdim", type=int, default=32)
parser.add_argument("--bs", type=int, default=32)
parser.add_argument("--vae_beta", type=int, default=1,
                    help="beta weight term in vae loss")
parser.add_argument("--lr_vae", type=float, default=0.001)
parser.add_argument("--n_epochs_full_vae", type=int, default=100)
parser.add_argument("--vae_path", help="path to load trained vae from",type=str,
					default="")

# Policy Hyperparameters
parser.add_argument("--env", type=str, default="block",
                    help="block")
parser.add_argument("--n_epochs_policy", type=int, default=100)

args = parser.parse_args()



# VAE
vae = VAE(zdim=args.zdim,beta=args.vae_beta).cuda()
if not args.vae_path:
	vae.load_state_dict(torch.load(args.vae_path))



# Configure tensorboard
configure('%s/var_log' % args.savepath, flush_secs=5)
with open('%s/params.json' % args.savepath, 'w') as fp:
    json.dump(args, fp, indent=4, sort_keys=True)

# Load Data
data = np.load(args.datapath)
data.shuffle()
data_loader = data.DataLoader(data, batch_size=args.bs, shuffle=True)
n_batch = len(data_loader)

if n_epochs_full_vae > 0:
	
	vae_opt = optim.Adam(vae.parameters(),lr=lr_vae)
	
	for n in range(args.n_epochs_full_vae):
		for it, data in enumerate(data_loader):
			
			x = data[0].float().cuda()
			vae_opt.zero_grad()
			loss, recon_loss, kld = vae.loss(x)
			loss.backward()
			vae_opt.step()

			if it % (n_batch//5) == 0:
    			log_value('vae_loss', loss, it + n_batch * n)
    			log_value('kl_loss', kld, it + n_batch * n)
    			log_value('kl_loss', kld, it + n_batch * n)
	    		print("Train Loss: %.3f   Recon: %.3f   KLD: %.3f"%(loss.item(), recon_loss.item(), kld.item()))

		torch.save(vae.state_dict(), '%s/vae-%d' % (args.savepath, (epoch // 5)*5))
		log_images_vae(x[:20],vae,args.savepath,n)






    


















