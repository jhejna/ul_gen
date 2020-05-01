
configs = dict()

config = dict(
    agent=dict(),
    algo=dict(
        batch_size=256,
        replay_ratio=256,
        learning_rate=3e-4,
        ae_learning_rate=3e-4,
        target_entropy="auto",
        min_steps_learn=500,
        clip_grad_norm=10.0,
        bootstrap_timelimit=False,
        ae_beta=1.0,
    ),
    encoder=dict(img_channels=3,
                z_dim=72,
                rae=False,
                detach_at_conv=True
    ),
    actor=dict(),
    critic=dict(),

    env={
        "id": "procgen:procgen-fruitbot-v0",
        "num_levels": 500,
        "start_level": 0,
        "distribution_mode": "easy"
    },
    eval_env={
        "id": "procgen:procgen-fruitbot-v0",
        "num_levels": 100,
        "start_level": 1000,
        "distribution_mode": "easy"
    },
    runner=dict(
        n_steps=int(2e6),
        log_interval_steps=int(1e4),
    ),
    sampler=dict(
        batch_T=16,
        batch_B=1,
        eval_n_envs=10,
        eval_max_steps=200000,
        eval_max_trajectories=100,
    ),
    optim=dict(),
    ae_optim=dict(),
    pretrain_steps=int(1e6)
)

configs["discrete_sac_ae"] = config
