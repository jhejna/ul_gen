import torch
import ul_gen
import os
from ul_gen.aug_vae.vae import VAE
from ul_gen.aug_vae.datasets import get_dataset
from torchvision.utils import save_image
import json

def train(params):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    dataset = get_dataset(params)
    loader = torch.utils.data.DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

    # # Debug: print aug pairs next to each other.
    # from matplotlib import pyplot as plt
    # sample, _ = next(iter(loader))
    # plt.imshow(sample['orig'][0][0])
    # plt.show()
    # plt.imshow(sample['aug'][0][0])
    # plt.show()
    # exit()

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
    sim_loss_coef = params["sim_loss_coef"]
    loss_type = params["loss_type"]

    for epoch in range(params["epochs"]):
        for batch, _ in loader:
            optimizer.zero_grad()
            # Concatenate to feed all the data through
            x = torch.cat((batch['orig'], batch['aug']), dim=0).to(device)
            if params["final_act"] == "tanh":
                x = 2*x - 1
            x_hat, mu, log_var = model(x)
            kl_loss = torch.sum(-0.5*(1 + log_var - mu.pow(2) - log_var.exp())) / len(x) # Divide by batch size
            
            if loss_type == "l2":
                recon_loss = torch.sum((x - x_hat).pow(2)) / len(x)
            elif loss_type == "bce":
                recon_loss = torch.nn.functional.binary_cross_entropy(x_hat, x, reduction='sum') / len(x)
            
            # Compute the similarity loss
            if params["sim_loss_coef"] > 0:
                mu_orig, mu_aug = torch.chunk(mu, 2, dim=0)
                log_var_orig, log_var_aug = torch.chunk(log_var, 2, dim=0)
                mu_orig, mu_aug = mu_orig[:, :k_dim], mu_aug[:, :k_dim]
                log_var_orig, log_var_aug = log_var_orig[:, :k_dim], log_var_aug[:, :k_dim]
                # KL divergence between original and augmented.
                sim_loss = torch.sum(log_var_aug - log_var_orig + 0.5*(log_var_orig.exp() + (mu_orig - mu_aug).pow(2))/log_var_aug.exp() - 0.5)/ (len(x) // 2)
                # sim_loss = torch.sum(0.5*(mu_orig - mu_aug).pow(2)) / len(x)
                loss = recon_loss + beta * kl_loss + sim_loss_coef * sim_loss
            else:
                loss = recon_loss + beta * kl_loss
            
            loss.backward()
            optimizer.step()
            
        if params["sim_loss_coef"] > 0:
            sim_loss_item = sim_loss.item()
        else:
            sim_loss_item = -1.0
        print('Epoch %d Recon Loss: %.3f KL Loss: %.3f Sim Loss: %.3f' % (epoch+1, recon_loss.item(), kl_loss.item(), sim_loss_item))
        
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
