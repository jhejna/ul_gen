
configs = dict()


config = dict(
    agent=dict(),
    algo=dict(
        discount=0.999,
        learning_rate=1e-4,
        value_loss_coeff=0.5,
        entropy_loss_coeff=0.01,
        clip_grad_norm=1.,
        gae_lambda=0.95,
        linear_lr_schedule=False,
        minibatches=64,
        epochs=3,
        ratio_clip=.2,
        normalize_advantage=True,
        beta=0.9,
        vae_loss_coef=1.0,
    ),
    env={
        "id": "procgen:procgen-coinrun-v0",
        "num_levels": 500,
        "start_level": 0,
        "distribution_mode": "easy"
    },
    model=dict(
        zdim=256,
        detach_vae=False,
        deterministic=False,
        img_height=64,
        shared_layers=[128,], 
        policy_layers=[15,],
        value_layers=[1,]
    ),
    optim=dict(),
    runner=dict(
        n_steps=1e6,
        log_interval_steps=1e5,
    ),
    sampler=dict(
        batch_T=64,
        batch_B=1,
        eval_n_envs=10,
        eval_max_steps=20000,
        eval_max_trajectories=100,
    ),
)

configs["ppo"] = config