
configs = dict()


config = dict(
    agent=dict(),
    algo=dict(
        discount=0.999,
        learning_rate=8e-4,
        value_loss_coeff=0.5,
        entropy_loss_coeff=0.01,
        clip_grad_norm=1.,
        gae_lambda=0.95,
        linear_lr_schedule=True,
        minibatches=8,
        epochs=3,
        ratio_clip=.2,
        normalize_advantage=True,
    ),
    env={
        "id": "procgen:procgen-coinrun-v0",
        "num_levels": 500,
        "start_level": 0,
        "distribution_mode": "easy"
    },
    eval_env={
        "id": "procgen:procgen-coinrun-v0",
        "num_levels": 100,
        "start_level": 1000,
        "distribution_mode": "easy"
    },
    model=dict(
        img_height=64,
        policy_layers=[15],
        value_layers=[1]
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

configs["ppo"] = config
