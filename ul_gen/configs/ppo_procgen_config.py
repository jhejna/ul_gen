
configs = dict()


config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        learning_rate=5e-4,
        value_loss_coeff=1.,
        entropy_loss_coeff=0.01,
        clip_grad_norm=1.,
        gae_lambda=0.98,
        linear_lr_schedule=True,
        minibatches=8,
        epochs=3,
        ratio_clip=.2,
    ),
    env={
        "id": "procgen:procgen-coinrun-v0",
        "num_levels" : 500,
        "start_level": 0,
        "distribution_mode" : "easy"
    },
    model=dict(
        num_filters=128,
    ),
    optim=dict(),
    runner=dict(
        n_steps=1e5,
        log_interval_steps=1e4,
    ),
    sampler=dict(
        batch_T=64,
        batch_B=16,
        max_decorrelation_steps=1000,
        eval_n_envs=10,
        eval_max_steps=20000,
        eval_max_trajectories=50,
    ),
)

configs["ppo"] = config
