
configs = dict()

config = dict(
    agent=dict(),
    algo=dict(
        batch_size=256,
        replay_ratio=256,
        learning_rate=3e-4,
        target_entropy="auto",
        min_steps_learn=400,
        clip_grad_norm=5.0,
        bootstrap_timelimit=False,
    ),
    env={
        "id": "CartPole-v1",
    },
    eval_env={
        "id": "CartPole-v1",
    },
    runner=dict(
        n_steps=1e4,
        log_interval_steps=1e3,
    ),
    sampler=dict(
        batch_T=1,
        batch_B=1,
        eval_n_envs=10,
        eval_max_steps=200000,
        eval_max_trajectories=50,
    ),
    optim=dict(),
    model=dict(
        hidden_sizes=[64, 64],
    ),
    q_model=dict(
        hidden_sizes=[64, 64],
    )
)

configs["discrete_sac"] = config
