from ul_gen.aug_vae.trainer import train

params = {
    "img_dim" : 48,
    "img_channels": 1,
    "batch_size": 96,
    "lr": 1e-4,
    "sim_loss_coef": 0.0,
    "z_dim": 16,
    "k_dim": 12,
    "beta": 1.1,
    "epochs" : 60,
    "save_freq": 10,
    "savepath": "mnist_vae_new",
    "dataset": "mnist",
    "final_act" : "sigmoid",
    "loss_type" : "bce", 
    "dataset_args": {
        "output_size": 48,
        "resize": (0.7, 1.1),
        "rotate": (-45, 45),
    }
}

train(params)
