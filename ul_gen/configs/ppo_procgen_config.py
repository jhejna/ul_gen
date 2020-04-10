
configs = dict()


config = dict(
    agent=dict(),
    algo=dict(
        discount=0.999,
        learning_rate=5e-4,
        value_loss_coeff=1.,
        entropy_loss_coeff=0.01,
        clip_grad_norm=1.,
        gae_lambda=0.95,
        linear_lr_schedule=True,
        minibatches=8,
        epochs=3,
        ratio_clip=.2,
        normalize_advantage=True
    ),
    env={
        "id": "procgen:procgen-coinrun-v0",
        "num_levels": 500,
        "start_level": 0,
        "distribution_mode": "easy"
    },
    model=dict(
        cnn_filters=[16, 32, 32],
        cnn_output=256,
        policy_layers=[15],
        value_layers=[1]
    ),
    optim=dict(),
    runner=dict(
        n_steps=1e6,
        log_interval_steps=1e5,
    ),
    sampler=dict(
        batch_T=256,
        batch_B=16,
        eval_n_envs=10,
        eval_max_steps=20000,
        eval_max_trajectories=100,
    ),
)

configs["ppo"] = config
