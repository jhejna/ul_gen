from ul_gen.aug_vae.trainer import train
from ul_gen.aug_vae.cycle_trainer import cycle_train
from ul_gen.aug_vae.classifier import train_classifier
from ul_gen.aug_vae.bias_trainer import train_bias

params = {
    "img_dim" : 28,
    "img_channels": 3,
    "batch_size": 96*2,
    "lr": 1e-4,
    "z_dim": 15,
    "k_dim": 12,
    "beta": 1.0,
    "epochs" : 80,
    "save_freq": 10,
    "pred_loss" : 20.0,
    "savepath": "cmnist_bias_vae_high",
    "dataset": "colored_mnist",
    "final_act" : "sigmoid",
    "arch_type" : 2,
    "loss_type" : "bce", 
    "fc_size" : 128,
    "dataset_args": {
        "test" : False
    }
}

train_bias(params)
