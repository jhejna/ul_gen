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


    num_labels = 3
    color_predictor = torch.nn.Sequential(torch.nn.Linear(k_dim, 128),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(128, num_labels)).to(device)
    color_predictor_optim = torch.optim.Adam(color_predictor.parameters(), lr=params['lr'])
    bias_criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    # bias_criterion = torch.nn.MSELoss()
    for epoch in range(params["epochs"]):
        for (x, bias_label), y in loader:
            optimizer.zero_grad()
            # Concatenate to feed all the data through
            x = x.to(device)
            y = y.to(device)
            bias_label = bias_label.to(device) #.float().unsqueeze(-1)
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
                logits = color_predictor(mu[:, :k_dim])
                pred_dist = F.softmax(logits, dim=1)
                ent_loss = torch.mean(torch.sum(pred_dist * torch.log(pred_dist),1))
                # ent_loss = -1 * bias_criterion(logits, bias_label)
                vae_bloss_item = ent_loss.item()
                loss = loss + pred_loss * ent_loss
            else:
                vae_bloss_item = 0.0

            loss.backward()
            optimizer.step()

            color_predictor_optim.zero_grad()
            mu, _ = model.encoder(x)
            logits = color_predictor(mu[:, :k_dim])
            
            bias_loss = bias_criterion(logits, bias_label)

            bias_loss.backward()
            color_predictor_optim.step()
            
            bias_loss_item = bias_loss.item()
            

        print('Epoch %d Recon Loss: %.3f KL Loss: %.3f Bias Loss: %.3f Vae Bloss: %.3f' % (epoch+1, recon_loss.item(), kl_loss.item(), bias_loss_item, vae_bloss_item))
        
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
            
            torch.save(model.state_dict(), '%s/aug-vae-%d' % (savepath, epoch+1))
            model.train()


