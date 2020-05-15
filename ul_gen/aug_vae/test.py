import torch
import torchvision
import json
import os
import numpy as np
from torchvision.utils import save_image
from datasets import MnistAug
from ul_gen.aug_vae.vae import VAE
from ul_gen.aug_vae.datasets import get_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def mnist_rot_test(model_path):
    save_path = os.path.dirname(model_path)
    params_path = os.path.join(save_path, 'params.json')
    with open(params_path, 'r') as fp:
        params = json.load(fp)
    
    mnist = torchvision.datasets.MNIST('~/.pytorch/mnist', train=True, download=True, transform=None)
    
    model = VAE(img_dim=params["img_dim"], img_channels=params["img_channels"], 
                                        z_dim=params["z_dim"], final_act=params["final_act"], 
                                        fc_size=params["fc_size"], arch_type=params["arch_type"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    n_interp = 8
    img_channels = params["img_channels"]
    img_dim = params["img_dim"]
    k_dim = params["k_dim"]
    x_orig = torch.zeros(n_interp, img_channels, img_dim, img_dim)
    x_aug = torch.zeros(n_interp, img_channels, img_dim, img_dim)

    test_indices = [9*i for i in range(n_interp)]
    mnist_aug = MnistAug(output_size=28)

    for i, index in enumerate(test_indices):
        x_orig[i] = mnist_aug.manual_img_aug(mnist[index][0], rescale=None, rotation=-50)
        x_aug[i] = mnist_aug.manual_img_aug(mnist[index][0], rescale=None, rotation=50)
    
    # Prep For Interpolations
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
    save_image(out_interp.detach().cpu(), os.path.join(save_path, 'test_interp_reg.png'), nrow=10)

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
    save_image(out_interp.detach().cpu(), os.path.join(save_path, 'test_interp_aug.png'), nrow=10)

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
    save_image(out_interp.detach().cpu(), os.path.join(save_path, 'test_interp_topk.png'), nrow=10)

def recon_test(model_path):
    save_path = os.path.dirname(model_path)
    params_path = os.path.join(save_path, 'params.json')
    with open(params_path, 'r') as fp:
        params = json.load(fp)
        
    model = VAE(img_dim=params["img_dim"], img_channels=params["img_channels"], 
                                        z_dim=params["z_dim"], final_act=params["final_act"], 
                                        fc_size=params["fc_size"], arch_type=params["arch_type"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    # Get the test dataset.
    params["dataset_args"]["test"] = True

    dataset = get_dataset(params)

    n_interp = 8
    img_channels = params["img_channels"]
    img_dim = params["img_dim"]
    k_dim = params["k_dim"]

    test_indices = [i for i in range(n_interp)]
    print("Test Indices", test_indices)

    x = torch.zeros(n_interp, img_channels, img_dim, img_dim)
    
    # 4, 22, 47, 51, 80, 68, 90, 104
    # test_indices = [3586, 5462, 5763, 6030, 6344, 6886, 9597, 9510]
    test_indices = [312, 314, 347, 478, 22, 581, 182, 111]
    for i, index in enumerate(test_indices):
        x[i] = dataset[index][0]

    x_hat, _, _ = model(x)
    recon = torch.cat((x, x_hat),dim=0) 
    if params["final_act"] == "tanh":
        recon = (recon + 1)/2
    save_image(recon.detach().cpu(), os.path.join(save_path, 'test_on_unseen_reconstructions.png'), nrow=n_interp)


def pick_recons(model_path_1, model_path_2):
    save_path = os.path.dirname(model_path_1)
    params_path = os.path.join(save_path, 'params.json')
    with open(params_path, 'r') as fp:
        params = json.load(fp)
        
    model1 = VAE(img_dim=params["img_dim"], img_channels=params["img_channels"], 
                                        z_dim=params["z_dim"], final_act=params["final_act"], 
                                        fc_size=params["fc_size"], arch_type=params["arch_type"]).to(device)
    model1.load_state_dict(torch.load(model_path_1, map_location=torch.device(device)))

    model2 = VAE(img_dim=params["img_dim"], img_channels=params["img_channels"], 
                                        z_dim=params["z_dim"], final_act=params["final_act"], 
                                        fc_size=params["fc_size"], arch_type=params["arch_type"]).to(device)
    model2.load_state_dict(torch.load(model_path_2, map_location=torch.device(device)))

    # Get the test dataset.
    params["dataset_args"]["test"] = True

    dataset = get_dataset(params)

    n_interp = 8
    img_channels = params["img_channels"]
    img_dim = params["img_dim"]
    k_dim = params["k_dim"]
    inds = []
    errors1 = 0
    errors2 = 0
    for i, (x, _, _) in enumerate(dataset):
        x = x.unsqueeze(0)
        x_hat1, _, _ = model1(x)
        x_hat2, _, _ = model2(x)

        error1 = torch.sum((x_hat1 - x)**2)
        error2 = torch.sum((x_hat2 - x)**2)
        # diff = error1 - error2
        # if diff > 4.0:
        #     print(i, diff.item())
        errors1 += error1.item()
        errors2 += error2.item()

        print(i ,errors1, errors2)
    
    # x_hat, _, _ = model(x)
    # recon = torch.cat((x, x_hat),dim=0) 
    # if params["final_act"] == "tanh":
    #     recon = (recon + 1)/2
    # save_image(recon.detach().cpu(), os.path.join(save_path, 'picked_test_recon.png'), nrow=n_interp)


def interp_test(model_path):
    save_path = os.path.dirname(model_path)
    params_path = os.path.join(save_path, 'params.json')
    with open(params_path, 'r') as fp:
        params = json.load(fp)
        
    model = VAE(img_dim=params["img_dim"], img_channels=params["img_channels"], 
                                        z_dim=params["z_dim"], final_act=params["final_act"], 
                                        fc_size=params["fc_size"], arch_type=params["arch_type"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

    # Get the test dataset.
    params["dataset_args"]["test"] = True

    dataset = get_dataset(params)

    n_interp = 5
    img_channels = params["img_channels"]
    img_dim = params["img_dim"]
    k_dim = params["k_dim"]
    x_orig = torch.zeros(n_interp, img_channels, img_dim, img_dim)
    x_aug = torch.zeros(n_interp, img_channels, img_dim, img_dim)

    x_orig[0] = dataset[30][0].unsqueeze(0)
    x_aug[0] = dataset[32][0].unsqueeze(0)
    x_orig[1] = dataset[34][0].unsqueeze(0)
    x_aug[1] = dataset[17][0].unsqueeze(0)
    x_orig[2] = dataset[29][0].unsqueeze(0)
    x_aug[2] = dataset[31][0].unsqueeze(0)
    x_orig[3] = dataset[11][0].unsqueeze(0)
    x_aug[3] = dataset[22][0].unsqueeze(0)
    x_orig[4] = dataset[3][0].unsqueeze(0)
    x_aug[4] = dataset[28][0].unsqueeze(0)
    
    # Prep For Interpolations
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
    save_image(out_interp.detach().cpu(), os.path.join(save_path, 'test_interp_reg.png'), nrow=10)

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
    save_image(out_interp.detach().cpu(), os.path.join(save_path, 'test_interp_aug.png'), nrow=10)

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
    save_image(out_interp.detach().cpu(), os.path.join(save_path, 'test_interp_topk.png'), nrow=10)

def double_recon_test(model_path_1, model_path_2):
    save_path = os.path.dirname(model_path_1)
    params_path = os.path.join(save_path, 'params.json')
    with open(params_path, 'r') as fp:
        params = json.load(fp)

    img_channels = 3
    img_dim = 28
    
    model1 = VAE(img_dim=params["img_dim"], img_channels=params["img_channels"], 
                                        z_dim=params["z_dim"], final_act=params["final_act"], 
                                        fc_size=params["fc_size"], arch_type=params["arch_type"]).to(device)
    model1.load_state_dict(torch.load(model_path_1, map_location=torch.device(device)))

    model2 = VAE(img_dim=params["img_dim"], img_channels=params["img_channels"], 
                                        z_dim=params["z_dim"], final_act=params["final_act"], 
                                        fc_size=params["fc_size"], arch_type=params["arch_type"]).to(device)
    model2.load_state_dict(torch.load(model_path_2, map_location=torch.device(device)))

    # Get the test dataset.
    params["dataset_args"]["test"] = True
    params["dataset_args"]["holdout"] = [0,1,2,3,4,5,6]

    dataset = get_dataset(params)

    n_interp = 10
    x = torch.zeros(n_interp, img_channels, img_dim, img_dim)
    
    # Good indicies: 7, 97

    test_indices = [7 +10*i for i in range(n_interp)]
    for i, index in enumerate(test_indices):
        x[i] = dataset[index][0]

    x_hat, _, _ = model1(x)
    recon = torch.cat((x, x_hat),dim=0) 
    if params["final_act"] == "tanh":
        recon = (recon + 1)/2
    save_image(recon.detach().cpu(), os.path.join(save_path, 'model1_recon.png'), nrow=n_interp)

    x_hat, _, _ = model2(x)
    recon = torch.cat((x, x_hat),dim=0) 
    if params["final_act"] == "tanh":
        recon = (recon + 1)/2
    save_image(recon.detach().cpu(), os.path.join(save_path, 'model2_recon.png'), nrow=n_interp)



if __name__ == "__main__":
    # pick_recons("/home/joey/berkeley/sp20/cs294-158/final_models/cmnist_vae_fix/aug-vae-50",
    #             "/home/joey/berkeley/sp20/cs294-158/final_models/cmnist_bias_vae_high/aug-vae-50")
    # interp_test("/home/joey/berkeley/sp20/cs294-158/final_models/cmnist_vae_fix/aug-vae-50")

    pick_recons("/home/joey/misc/cmnist_holdout_vae/aug-vae-40",
                      "/home/joey/misc/cmnist_holdout_bias_vae3/aug-vae-40")

    # two fives, same color: 15, 23
    # Three different color: 31, 33 