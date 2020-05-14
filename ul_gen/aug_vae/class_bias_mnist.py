from ul_gen.aug_vae.trainer import train
from ul_gen.aug_vae.cycle_trainer import cycle_train
from ul_gen.aug_vae.classifier import train_classifier
from ul_gen.aug_vae.bias_trainer2 import train_bias

params = {
    "img_dim" : 28,
    "img_channels": 3,
    "batch_size": 96,
    "lr": 1e-4,
    "z_dim": 15,
    "k_dim": 12,
    "beta": 1.0,
    "epochs" : 20,
    "save_freq": 10,
    "pred_loss" : 10,
    "savepath": "cmnist_holdout_bias_vae",
    "dataset": "colored_mnist",
    "final_act" : "sigmoid",
    "arch_type" : 2,
    "loss_type" : "bce", 
    "fc_size" : 128,
    "dataset_args": {
        "test" : False,
        "holdout" : [7,8,9]
    }
}

train_bias(params)
