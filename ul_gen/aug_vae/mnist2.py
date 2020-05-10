from ul_gen.aug_vae.trainer import train

params = {
    "img_dim" : 28,
    "img_channels": 1,
    "batch_size": 96,
    "lr": 1e-4,
    "sim_loss_coef": 0.0,
    "z_dim": 11,
    "k_dim": 10,
    "beta": 1.0,
    "epochs" : 50,
    "save_freq": 10,
    "savepath": "mnist_vae_rot",
    "dataset": "mnist",
    "final_act" : "sigmoid",
    "arch_type" : 1,
    "loss_type" : "bce", 
    "fc_size" : 512,
    "dataset_args": {
        "output_size": 28,
        "resize": None,
        "rotate": (-60, 60),
    }
}

train(params)
