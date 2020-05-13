configs = dict()

config = dict(
    checkpoint=None,
    # override=dict(
    #     override_policy_value=True, 
    #     policy_layers=[64,64,15],
    #     value_layers=[64,64,1]),
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
        normalize_rewards=True,
        alae_learning_rate=2e-3,
    ),
    env={
        "id": "procgen:procgen-coinrun-v0",
        "num_levels": 200,
        "start_level": 0,
        "distribution_mode": "hard"
    },
    eval_env={
        "id": "procgen:procgen-coinrun-v0",
        "num_levels": 100,
        "start_level": 2000,
        "distribution_mode": "hard"
    },
    model=dict(
        zdim=128,
        wdim=128,
        img_shape=(3,64,64),
        detach_encoder=True,
        policy_layers=[64,64,15],
        value_layers=[64,64,1],
        noise_prob=0.,        
        arch_type=1,
    ),
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
        "id": "procgen:procgen-coinrun-v0",
        "num_levels": 1000,
        "start_level": 0,
        "distribution_mode": "hard"
    },
    sampler=dict(
        batch_T=32,
        batch_B=8,
        eval_n_envs=0,
        eval_max_steps=0,
        eval_max_trajectories=0
    ),
    algo=dict(

    ),
    model=dict(
        zdim=128,
        wdim=128,
        img_shape=(3,64,64),
        detach_encoder=True,
        policy_layers=[64,64,15],
        value_layers=[64,64,1],
        noise_prob=0.,
        arch_type=1,
    ),
    train_steps=int(1e6),
    log_freq=1000,
    eval_freq=5000,
)

configs["ppo_alae"] = config
configs["pretrain"] = pretrain_config