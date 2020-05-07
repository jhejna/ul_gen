
configs = dict()
# aug_to_func = {
#                 'crop':rad.random_crop,
#                 'crop_horiz': rad.random_crop_horizontile,
#                 'grayscale':rad.random_grayscale,
#                 'cutout':rad.random_cutout,
#                 'cutout_color':rad.random_cutout_color,
#                 'flip':rad.random_flip,
#                 'rotate':rad.random_rotation,
#                 'rand_conv':rad.random_convolution,
#                 'color_jitter':rad.random_color_jitter,
#                 'no_aug':rad.no_aug,
#             }

config = dict(
    checkpoint=None,
    agent=dict(data_augs="crop_horiz-cutout_color-color_jitter"),
    algo=dict(
        discount=0.999,
        learning_rate=5e-4,
        entropy_loss_coeff=0.01,
        clip_grad_norm=1.,
        gae_lambda=0.95,
        linear_lr_schedule=False,
        minibatches=8,
        epochs=3,
        ratio_clip=0.2,
        normalize_advantage=True,
    ),
    env={
        "id": "procgen:procgen-fruitbot-v0",
        "num_levels": 200,
        "start_level": 0,
        "distribution_mode": "easy"
    },
    model=dict(
    ),
    optim=dict(),
    runner=dict(
        n_steps=5e6,
        log_interval_steps=1e5,
    ),
    sampler=dict(
        batch_T=224,
        batch_B=32,
        eval_n_envs=8,
        eval_max_steps=20000,
        eval_max_trajectories=100,
    ),
)

configs["ppo"] = config

config = dict(
    checkpoint=None,
    agent=dict(data_augs="crop_horiz-cutout_color-flip", both_actions=True),
    algo=dict(
        discount=0.999,
        learning_rate=5e-4,
        entropy_loss_coeff=0.01,
        clip_grad_norm=1.,
        gae_lambda=0.95,
        linear_lr_schedule=False,
        minibatches=8,
        epochs=3,
        ratio_clip=0.2,
        normalize_advantage=True,
        similarity_loss=True,
        similarity_coeff=0.1
    ),
    env={
        "id": "procgen:procgen-coinrun-v0",
        "num_levels": 200,
        "start_level": 0,
        "distribution_mode": "easy"
    },
    model=dict(
    ),
    optim=dict(),
    runner=dict(
        n_steps=5e6,
        log_interval_steps=1e5,
    ),
    sampler=dict(
        batch_T=256,
        batch_B=32,
        eval_n_envs=8,
        eval_max_steps=20000,
        eval_max_trajectories=1000,
    ),
)

configs['ppo_aug'] = config
