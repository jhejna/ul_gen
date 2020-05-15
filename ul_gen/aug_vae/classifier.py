import torch
import ul_gen
import os
from ul_gen.aug_vae.vae import VAE
from ul_gen.aug_vae.datasets import get_dataset
import json

DATASET_TO_CLASSES = {
    "mnist" : 10,
    "mnist_aug" : 10,
    "colored_mnist" : 10,
    "chairs" : 1393,
}

def train_classifier(model_path, load_checkpoint=True, finetune=False,
                                 lr=1e-3, epochs=5,
                                 deterministic=False, hidden_layers=[], 
                                 activation="relu",
                                 test_dataset=None):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not load_checkpoint:
        # We're training a brand new model.
        finetune = True
        deterministic = True

    save_path = os.path.dirname(model_path)
    params_path = os.path.join(save_path, 'params.json')

    with open(params_path, 'r') as fp:
        params = json.load(fp)
    #params["dataset_args"]["test"] = False
    #params["dataset_args"]["bias_label"] = False
    print("PARAMS", params)
    # Load the dataset
    dataset = get_dataset(params)
    loader = torch.utils.data.DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

    # Create the model
    model = VAE(img_dim=params["img_dim"], img_channels=params["img_channels"], 
                                        z_dim=params["z_dim"], final_act=params["final_act"], 
                                        fc_size=params["fc_size"], arch_type=params["arch_type"]).to(device)
    if load_checkpoint:
        model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    
    num_classes = DATASET_TO_CLASSES[params["dataset"]]
    classifier_layers = []
    k_dim = params["k_dim"]
    last_dim = params["z_dim"]
    for hidden_size in hidden_layers:
        classifier_layers.append(torch.nn.Linear(last_dim, hidden_size))
        last_dim = hidden_size
        if activation == "relu":
            classifier_layers.append(torch.nn.ReLU())
        elif activation == "tanh":
            classifier_layers.append(torch.nn.Tanh())
        else:
            raise ValueError("Didn't provide a correct activation")
    classifier_layers.append(torch.nn.Linear(last_dim, num_classes, bias=False))
    classifier = torch.nn.Sequential(*classifier_layers).to(device) 
    parameters = []
    if finetune:
        parameters.extend(model.parameters())
    parameters.extend(classifier.parameters())
    
    optimizer = torch.optim.Adam(parameters, lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for batch, y in loader:
            x, y = batch['orig'].to(device), y.to(device)
            optimizer.zero_grad()
            if finetune:
                mu, log_var = model.encoder(x)
                if deterministic:
                    z = mu
                else:
                    z = torch.exp(0.5*log_var) * torch.randn_like(mu) + mu
            else:
                with torch.no_grad():
                    mu, log_var = model.encoder(x)
                    if deterministic:
                        z = mu
                    else:
                        z = torch.exp(0.5*log_var) * torch.randn_like(mu) + mu

            #preds = classifier(z[:, :k_dim])
            preds = classifier(z)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
        print("Finished epoch", epoch + 1, "Loss:", loss.item())

    # Assess final accuracy on 10,000
    #params["dataset_args"]["test"] = True
    #params["dataset_args"]["bias_label"] = False
    dataset = get_dataset(params)
    loader = torch.utils.data.DataLoader(dataset, batch_size=params["batch_size"], shuffle=True)

    num_eval_pts = 0
    correct_pts = 0.0
    classifier.eval()
    model.eval()
    for batch, y in loader:
        x, y = batch['orig'].to(device), y.to(device)
        num_eval_pts += len(y)
        mu, log_var = model.encoder(x)
        if deterministic:
            z = mu
        else:
            z = torch.exp(0.5*log_var) * torch.randn_like(mu) + mu
        #logits = classifier(z[:,:k_dim])
        logits = classifier(z)
        preds = torch.argmax(logits, dim=1)
        correct_pts += torch.sum(preds == y).float()
        if num_eval_pts > num_eval_pts:
            break
    
    final_acc = correct_pts.cpu().numpy() / num_eval_pts
    print("FINAL ACCURACY", final_acc)

if __name__ == "__main__":
    train_classifier("output/mnist_vae_rot2/aug-vae-40", load_checkpoint=True,
            finetune=False, lr=5e-3, epochs=3, deterministic=True,
            hidden_layers=[], activation="relu")
