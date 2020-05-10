from ul_gen.aug_vae.trainer import train

params = {
    "img_dim" : 64,
    "img_channels": 1,
    "batch_size": 96,
    "lr": 1e-4,
    "sim_loss_coef": 0.0,
    "z_dim": 32,
    "k_dim": 28,
    "beta": 1.0,
    "epochs" : 120,
    "save_freq": 10,
    "fc_size": 256,
    "savepath": "chairs_aug_vae_long",
    "dataset": "chairs",
    "final_act" : "tanh",
    "loss_type" : "l2", 
    "dataset_args": {}
}

train(params)
