from ul_gen.aug_vae.trainer import train

params = {
    "img_dim" : 64,
    "img_channels": 1,
    "batch_size": 96,
    "lr": 1e-4,
    "sim_loss_coef": 1.0,
    "z_dim": 32,
    "k_dim": 28,
    "beta": 1.1,
    "epochs" : 50,
    "save_freq": 10,
    "savepath": "chairs_aug_vae",
    "dataset": "chairs",
    "final_act" : "tanh",
    "dataset_args": {}
}

train(params)
