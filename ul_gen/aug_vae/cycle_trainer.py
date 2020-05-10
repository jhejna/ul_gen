import torch
import ul_gen
import os
from ul_gen.aug_vae.vae import VAE
from ul_gen.aug_vae.datasets import get_dataset
from torchvision.utils import save_image
import json

def cycle_train(params):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    dataset = get_dataset(params)
    loader = torch.utils.data.DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

    # Setup the save path
    savepath = params["savepath"]
    if not savepath.startswith("/"):
        savepath = os.path.join(os.path.dirname(ul_gen.__file__) + "/aug_vae/output", savepath)

    os.makedirs(savepath, exist_ok=True)
    with open(os.path.join(savepath, 'params.json'), 'w') as fp:
        json.dump(params, fp)

    model = VAE(img_dim=params["img_dim"], img_channels=params["img_channels"], 
                                        z_dim=params["z_dim"], final_act=params["final_act"], 
                                        fc_size=params["fc_size"], arch_type=params["arch_type"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    z_dim = params["z_dim"]
    k_dim = params["k_dim"]
    beta = params["beta"]
    img_dim = params["img_dim"]
    img_channels = params["img_channels"]
    sim_loss_coef = None # params["sim_loss_coef"]
    loss_type = params["loss_type"]
    
    for epoch in range(params["epochs"]):
        for batch, _ in loader:
            optimizer.zero_grad()
            x1, x2 = batch['orig'].to(device), batch['aug'].to(device)
            
            # Pass X1 through the encoder
            mu1, logvar1 = model.encoder(x1)
            z1  = torch.exp(0.5*logvar1) * torch.randn_like(mu1) + mu1
            kl1 = torch.sum(-0.5*(1 + logvar1 - mu1.pow(2) - logvar1.exp())) / len(x1)

            # Pass X2 through the encoder
            mu2, logvar2 = model.encoder(x2)
            z2  = torch.exp(0.5*logvar2) * torch.randn_like(mu2) + mu2
            kl2 = torch.sum(-0.5*(1 + logvar2 - mu2.pow(2) - logvar2.exp())) / len(x1)

            # Swap the features past k_dim
            # Everything past k_dim can distinguish the images.
            recon_x2_in = torch.cat( (z1[:, :k_dim], z2[:, k_dim:]), dim=1)
            recon_x1_in = torch.cat( (z2[:, :k_dim], z1[:, k_dim:]), dim=1)

            xhat1 = model.decoder(recon_x1_in)
            xhat2 = model.decoder(recon_x2_in)

            if loss_type == "l2":
                recon1 = torch.sum((x1 - xhat1).pow(2)) / len(x1)
                recon2 = torch.sum((x2 - xhat2).pow(2)) / len(x2)
            elif loss_type == "bce":
                recon1 = torch.nn.functional.binary_cross_entropy(xhat1, x1, reduction='sum') / len(x1)
                recon2 = torch.nn.functional.binary_cross_entropy(xhat2, x2, reduction='sum') / len(x2)
            
            loss = 0.5 * (recon1 + recon2 + beta*(kl1 + kl2))
            
            loss.backward()
            optimizer.step()
            
        if params["sim_loss_coef"] > 0:
            sim_loss_item = sim_loss.item()
        else:
            sim_loss_item = -1.0
        print('Epoch %d Recon Loss: %.3f KL Loss: %.3f Sim Loss: %.3f' % (epoch+1, ((recon1 + recon2)/2).item(), ((kl1+kl2)/2).item(), sim_loss_item))
        
        if (epoch + 1) % params["save_freq"] == 0:
            # Save reconstructions and samples:
            model.eval()
            recon = torch.cat((x[:8], x_hat[:8]),dim=0) 
            if params["final_act"] == "tanh":
                recon = (recon + 1)/2
            save_image(recon.detach().cpu(), os.path.join(savepath, 'recon_' + str(epoch+1) +'.png'), nrow=8)

            zs = torch.randn(16, z_dim).to(device)
            samples = model.decoder(zs)
            if params["final_act"] == "tanh":
                samples = (samples + 1)/2
            save_image(samples.detach().cpu(), os.path.join(savepath, 'samples_' + str(epoch+1) +'.png'), nrow=8)
            
            # Prep For Interpolations
            n_interp = 8
            x_orig, x_aug = torch.chunk(x, 2, dim=0)
            x_orig, x_aug  = x_orig[:n_interp], x_aug[:n_interp]
            z_orig, _ = model.encoder(x_orig)
            z_aug, _ = model.encoder(x_aug)

            # Regular Interpolations
            diff_vec = z_aug - z_orig
            interpolations = []
            interpolations.append(x_orig)
            for i in range(1, 9):
                interpolations.append(model.decoder(z_orig + 0.111111*i*diff_vec))
            interpolations.append(x_aug)
            out_interp = torch.zeros(n_interp*10, img_channels, img_dim, img_dim)
            for i in range(10):
                for j in range(n_interp):
                    out_interp[10*j + i, :, :, :] = interpolations[i][j, :, :, :]
            if params["final_act"] == "tanh":
                out_interp = (out_interp + 1)/2
            save_image(out_interp.detach().cpu(), os.path.join(savepath, 'interp_reg_' + str(epoch+1) +'.png'), nrow=10)
            
            # Aug Interpolations
            diff_vec = z_aug - z_orig
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
            if params["final_act"] == "tanh":
                out_interp = (out_interp + 1)/2
            save_image(out_interp.detach().cpu(), os.path.join(savepath, 'interp_aug_' + str(epoch+1) +'.png'), nrow=10)

            # Set the first k components of diff vec to be zero, so we only vary along aug components
            diff_vec = z_aug - z_orig
            dists_per_z_dim = torch.sum(torch.pow(diff_vec, 2), axis=0)
            _, zeroing_inds = torch.topk(dists_per_z_dim, k_dim, largest=False)
            diff_vec[:, zeroing_inds] = 0
            interpolations = []
            interpolations.append(x_orig)
            for i in range(1, 9):
                interpolations.append(model.decoder(z_orig + 0.111111*i*diff_vec))
            interpolations.append(x_aug)
            out_interp = torch.zeros(n_interp*10, img_channels, img_dim, img_dim)
            for i in range(10):
                for j in range(n_interp):
                    out_interp[10*j + i, :, :, :] = interpolations[i][j, :, :, :]
            if params["final_act"] == "tanh":
                out_interp = (out_interp + 1)/2
            save_image(out_interp.detach().cpu(), os.path.join(savepath, 'interp_topk_' + str(epoch+1) +'.png'), nrow=10)
                
            torch.save(model.state_dict(), '%s/aug-vae-%d' % (savepath, epoch+1))
            model.train()
