import torch
import ul_gen
import os
from ul_gen.aug_vae.vae import VAE
from ul_gen.aug_vae.datasets import get_dataset
from torchvision.utils import save_image
from torch.nn import functional as F
import json

def train_bias(params):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    dataset = get_dataset(params)
    loader = torch.utils.data.DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

    params["dataset_args"]["test"] = True
    test_data = get_dataset(params)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=params["batch_size"], shuffle=True)
    params["dataset_args"]["test"] = False
    # # Debug: print aug pairs next to each other.
    # from matplotlib import pyplot as plt
    # sample, _, _ = next(iter(loader))
    # plt.imshow(sample[0].permute(1, 2, 0))
    # plt.show()
    # plt.imshow(sample[1].permute(1, 2, 0))
    # plt.show()

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
    loss_type = params["loss_type"]
    pred_loss = params["pred_loss"]

    num_color_bins = 4
    color_predictor = torch.nn.Sequential(torch.nn.Linear(k_dim, 128),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(128, 3*num_color_bins)).to(device)
    color_predictor_optim = torch.optim.Adam(color_predictor.parameters(), lr=50*params['lr'])
    bias_criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(params["epochs"]):
        for x, bias_label, y in loader:
            optimizer.zero_grad()
            # Concatenate to feed all the data through
            x = x.to(device)
            if params["final_act"] == "tanh":
                x = 2*x - 1

            x_hat, mu, log_var = model(x)
            kl_loss = torch.sum(-0.5*(1 + log_var - mu.pow(2) - log_var.exp())) / len(x) # Divide by batch size
            
            if loss_type == "l2":
                recon_loss = torch.sum((x - x_hat).pow(2)) / len(x)
            elif loss_type == "bce":
                recon_loss = torch.nn.functional.binary_cross_entropy(x_hat, x, reduction='sum') / len(x)
            
            loss = recon_loss + beta * kl_loss

            if pred_loss > 0.0:
                # add the prediction loss.
                r_logits, g_logits, b_logits = torch.chunk(color_predictor(mu[:, :k_dim]), 3, dim=1)
                r_pred, g_pred, b_pred = F.softmax(r_logits, dim=1), F.softmax(g_logits, dim=1), F.softmax(b_logits, dim=1)

                loss_r = torch.mean(torch.sum(r_pred * torch.log(r_pred),1))
                loss_b = torch.mean(torch.sum(b_pred * torch.log(b_pred),1))
                loss_g = torch.mean(torch.sum(g_pred * torch.log(g_pred),1))
                vae_bloss = (loss_r + loss_b + loss_g) / 3.
                vae_bloss_item = vae_bloss.item()
                loss = loss + pred_loss * vae_bloss
            else:
                vae_bloss_item = -1.0
            loss.backward()
            optimizer.step()

            
            # Train the bias predictor network.
            bias_label = bias_label.to(device)
            color_predictor_optim.zero_grad()
            mu, _ = model.encoder(x)
            r_logits, g_logits, b_logits = torch.chunk(color_predictor(mu[:, :k_dim]), 3, dim=1)
                # print(bias_label[:,0].shape)
            bloss_r = bias_criterion(r_logits, bias_label[:, 0])
            bloss_g = bias_criterion(b_logits, bias_label[:, 1])
            bloss_b = bias_criterion(g_logits, bias_label[:, 2])
            bias_loss = bloss_r + bloss_g + bloss_b

            bias_loss.backward()
            color_predictor_optim.step()
                
            bias_loss_item = bias_loss.item()
            

        print('Epoch %d Recon Loss: %.3f KL Loss: %.3f Bias Loss: %.3f Vae Bias Loss: %.3f' % (epoch+1, recon_loss.item(), kl_loss.item(), bias_loss_item, vae_bloss_item))
        
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
            

            # Test Set Reconstructions
            test_x, _, _ = next(iter(test_loader))
            test_x = test_x.to(device)
            
            test_x_hat, _, _ = model(test_x[:24])
            recon = torch.cat((test_x[:24], test_x_hat),dim=0) 
            if params["final_act"] == "tanh":
                recon = (recon + 1)/2
            save_image(recon.detach().cpu(), os.path.join(savepath, 'test_recon_' + str(epoch+1) +'.png'), nrow=8)

            # Prep For Interpolations
            n_interp = 8
            x_orig, x_aug  = test_x[:n_interp], test_x[n_interp:2*n_interp]
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

            torch.save(model.state_dict(), '%s/aug-vae-%d' % (savepath, epoch+1))
            model.train()
