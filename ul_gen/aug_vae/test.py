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

def classifier_test(model_path, num_classes):
    save_path = os.path.dirname(model_path)
    params_path = os.path.join(save_path, 'params.json')
    with open(params_path, 'r') as fp:
        params = json.load(fp)

    model = VAE(img_dim=params["img_dim"], img_channels=params["img_channels"], 
                                        z_dim=params["z_dim"], final_act=params["final_act"], 
                                        fc_size=params["fc_size"], arch_type=params["arch_type"]).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    
    dataset = get_dataset(params)
    loader = torch.utils.data.DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

    # Define a linear classifier on top of the latents
    classifier = torch.nn.Linear(params["z_dim"], num_classes, bias=False)
    # classifier = torch.nn.Sequential(torch.nn.Linear(params["z_dim"], 64),
    #                                  torch.nn.Tanh(),
    #                                  torch.nn.Linear(64, num_classes))

    optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-3)
    criterion = torch.nn.CrossEntropyLoss()

    epochs = 5
    for epoch in range(epochs):
        for batch, y in loader:
            x, y = batch['orig'].to(device), y.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                z, _ = model.encoder(x)
            preds = classifier(z)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
        print("Finished epoch", epoch + 1, "Loss:", loss.item())

    # Assess final accuracy on 10,000
    num_eval_pts = 0
    correct_pts = 0.0
    classifier.eval()
    for batch, y in loader:
        num_eval_pts += len(batch)
        x, y = batch['orig'].to(device), y.to(device)
        logits = classifier(model.encoder(x)[0])
        preds = torch.argmax(logits, dim=1)
        correct_pts += torch.sum(preds == y).float()
        if num_eval_pts > num_eval_pts:
            break
    
    final_acc = correct_pts.cpu().numpy() / num_eval_pts
    print("FINAL ACCURACY", final_acc)


if __name__ == "__main__":
    classifier_test("/home/joey/misc/mnist_vae_cyc/aug-vae-50", 10)
    # mnist_rot_test("/home/joey/misc/mnist_vae_cyc/aug-vae-50")