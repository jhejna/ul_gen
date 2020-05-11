import torch
import ul_gen
import os
from ul_gen.aug_vae.vae import VAE
from ul_gen.aug_vae.datasets import get_dataset
import json

def train_classifier(params, num_classes):
    
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
    
    model = VAE(img_dim=params["img_dim"], img_channels=params["img_channels"], 
                                        z_dim=params["z_dim"], final_act=params["final_act"], 
                                        fc_size=params["fc_size"], arch_type=params["arch_type"]).to(device)

    classifier = torch.nn.Linear(params["z_dim"], num_classes, bias=False).to(device)

    parameters = []
    parameters.extend(model.parameters())
    parameters.extend(classifier.parameters())
    
    optimizer = torch.optim.Adam(parameters, lr=params['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    z_dim = params["z_dim"]
    k_dim = params["k_dim"]
    beta = params["beta"]
    img_dim = params["img_dim"]
    img_channels = params["img_channels"]
    loss_type = params["loss_type"]

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
        num_eval_pts += len(batch // 2)
        x, y = batch['orig'].to(device), y.to(device)
        logits = classifier(model.encoder(x)[0])
        preds = torch.argmax(logits, dim=1)
        correct_pts += torch.sum(preds == y).float()
        if num_eval_pts > num_eval_pts:
            break
    
    final_acc = correct_pts.cpu().numpy() / num_eval_pts
    print("FINAL ACCURACY", final_acc)