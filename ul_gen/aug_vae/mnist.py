from ul_gen.aug_vae.trainer import train

params = {
    "img_dim" : 48,
    "img_channels": 1,
    "batch_size": 96,
    "lr": 2.5e-4,
    "sim_loss_coef": 1.0,
    "z_dim": 16,
    "k_dim": 12,
    "beta": 1.1,
    "epochs" : 60,
    "save_freq": 10,
    "savepath": "mnist_aug_vae",
    "dataset": "mnist",
    "final_act" : "tanh",
    "dataset_args": {
        "output_size": 48,
        "resize": (0.7, 1.05),
        "rotate": (-75, 75),
    }
}

train(params)
