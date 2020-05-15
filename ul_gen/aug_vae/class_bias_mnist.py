from ul_gen.aug_vae.trainer import train
from ul_gen.aug_vae.cycle_trainer import cycle_train
from ul_gen.aug_vae.classifier import train_classifier
from ul_gen.aug_vae.bias_trainer2 import train_bias

params = {
    "img_dim" : 28,
    "img_channels": 1,
    "batch_size": 96,
    "lr": 1e-4,
    "pred_loss": 0.0,
    "z_dim": 11,
    "k_dim": 10,
    "beta": 1.0,
    "epochs" : 50,
    "save_freq": 10,
    "savepath": "mnist_pred_vae",
    "dataset": "mnist_aug",
    "final_act" : "sigmoid",
    "arch_type" : 1,
    "loss_type" : "bce", 
    "fc_size" : 512,
    "dataset_args": {
        "output_size": 28,
        "rot_labels" : True,
        "resize": None,
        "rotate": (-60, 60),
    }
}

train_bias(params)
