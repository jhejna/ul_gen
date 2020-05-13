from ul_gen.aug_vae.trainer import train
from ul_gen.aug_vae.cycle_trainer import cycle_train
from ul_gen.aug_vae.classifier import train_classifier

params = {
    "img_dim" : 28,
    "img_channels": 1,
    "batch_size": 96,
    "lr": 1e-4,
    "sim_loss_coef": 0.6,
    "z_dim": 11,
    "k_dim": 10,
    "beta": 1.0,
    "epochs" : 50,
    "save_freq": 10,
    "savepath": "mnist_aug_vae_tilted",
    "dataset": "mnist_aug",
    "final_act" : "sigmoid",
    "arch_type" : 1,
    "loss_type" : "bce", 
    "fc_size" : 512,
    "dataset_args": {
        "output_size": 28,
        "resize": None,
        "rotate": (-65, -25, 25, 65),
    }
}

# train(params)
# cycle_train(params)
train_classifier(params, 10)
