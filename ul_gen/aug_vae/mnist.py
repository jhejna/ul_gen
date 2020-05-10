from ul_gen.aug_vae.trainer import train

params = {
    "img_dim" : 64,
    "img_channels": 1,
    "batch_size": 96,
    "lr": 8e-5,
    "sim_loss_coef": 0.0,
    "z_dim": 20,
    "k_dim": 15,
    "fc_size" : 196,
    "beta": 1.0,
    "epochs" : 120,
    "save_freq": 10,
    "savepath": "mnist_vae_large2",
    "dataset": "mnist",
    "final_act" : "tanh",
    "loss_type" : "l2", 
    "dataset_args": {
        "output_size": 64,
        "resize": (1.0, 1.9),
        "rotate": (-60, 60),
    }
}

train(params)
