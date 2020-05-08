configs = dict()

config = dict(
    checkpoint="rad/bigfish_low_beta/ae-aug_vae_data-10000",
    override=dict(
        policy_layers=[64,64,15],
        value_layers=[64,64,1]),
    agent=dict(vae_loss_type="l2",
                vae_beta=0.05,
                sim_loss_coef=0.025,
                k_dim=96,
                data_augs="crop"),
    algo=dict(
        discount=0.999,
        learning_rate=4e-4,        
        value_loss_coeff=0.5,
        entropy_loss_coeff=0.01,
        clip_grad_norm=1.,
        gae_lambda=0.95,
        linear_lr_schedule=False,
        minibatches=16,
        epochs=3,
        ratio_clip=.2,
        normalize_advantage=True,
    ),
    env={
        "id": "procgen:procgen-bigfish-v0",
        "num_levels": 100,
        "start_level": 0,
        "distribution_mode": "easy"
    },
    model=dict(
        zdim=128,
        img_shape=(3,64,64),
        detach_policy=True,
        detach_vae=False,
        detach_value=True,
        deterministic=True,
        policy_layers=[64,64,15],
        value_layers=[64,64,1],
        encoder_layers=[32, 64, 96, 128],
        decoder_layers=[128, 96, 64, 32],
        arch_type=0,
        rae=False
    ),
    optim=dict(),
    runner=dict(
        n_steps=10e6,
        log_interval_steps=1e5,
    ),
    sampler=dict(
        batch_T=256,
        batch_B=32,
        eval_n_envs=8,
        eval_max_steps=200000,
        eval_max_trajectories=100,
    ),
)

configs["ppo_vae"] = config
