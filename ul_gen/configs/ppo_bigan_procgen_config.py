configs = dict()

config = dict(
    checkpoint='/home/karam/Downloads/ul_gen/vae_data/experiment/vae1000easyclimber',
    override=dict(
        override_policy_value=True, 
        policy_layers=[64,64,15],
        value_layers=[64,64,1]),
    agent=dict(),
    algo=dict(
        discount=0.999,
        learning_rate=3e-4,        
        value_loss_coeff=1.,
        entropy_loss_coeff=0.01,
        clip_grad_norm=1.,
        gae_lambda=0.95,
        linear_lr_schedule=False,
        minibatches=64,
        epochs=3,
        ratio_clip=.2,
        normalize_advantage=True,
        vae_beta=1,
        vae_loss_coeff=0.5,
        vae_loss_type="l2",
        vae_norm_loss=False,


    ),
    env={
        "id": "procgen:procgen-climber-v0",
        "num_levels": 1000,
        "start_level": 0,
        "distribution_mode": "easy"
    },
    eval_env={
        "id": "procgen:procgen-climber-v0",
        "num_levels": 100,
        "start_level": 1000,
        "distribution_mode": "easy"
    },
    model=dict(
        zdim=200,
        img_shape=(3,64,64),
        detach_policy=False,
        detach_vae=True,
        detach_value=False,
        deterministic=True,
        policy_layers=[64,64,15],
        value_layers=[64,64,1],
        noise_prob=0.25,        
        noise_weight=1.,
        no_noise_weight=.25,
        arch_type=1,
        rae=False
    ),
    optim=dict(),
    runner=dict(
        n_steps=5e6,
        log_interval_steps=1e5,
    ),
    sampler=dict(
        batch_T=256,
        batch_B=16,
        eval_n_envs=8,
        eval_max_steps=20000,
        eval_max_trajectories=100,
    ),
)

pretrain_config = dict(
    load_path="",
    env={
        "id": "procgen:procgen-climber-v0",
        "num_levels": 1000,
        "start_level": 0,
        "distribution_mode": "easy"
    },
    sampler=dict(
        batch_T=24,
        batch_B=4,
        eval_n_envs=0,
        eval_max_steps=0,
        eval_max_trajectories=0
    ),
    algo=dict(),
    optim=dict(
        lr=5e-4
    ),
    model=dict(
        zdim=150,
        detach_encoder=False,
        glyrs=[2*i for i in [256,128,64,32]],
        dlyrs=[i for i in [16,32,64,128]],
        policy_layers=[15],
        value_layers=[1],
    ),
    train_steps=int(5e5),
    log_freq=1000,
    eval_freq=5000,
)

configs["ppo_bigan"] = config
configs["pretrain"] = pretrain_config