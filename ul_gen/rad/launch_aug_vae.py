
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

affinity_code = encode_affinity(
    n_cpu_core=4,
    n_gpu=1,
    n_socket=1,
    contexts_per_gpu=2,
)

runs_per_setting = 2

variant_levels = list()

# Later extend this to cover more games
tasks = ["color_jitter"]
values = list(zip(tasks))
dir_names = ["{}".format(*v) for v in values]
keys = [("agent", "data_augs")]
variant_levels.append(VariantLevel(keys, values, dir_names))

variants, log_dirs = make_variants(*variant_levels)

print("Variants", variants)
print("Log_dirs", log_dirs)

script = "rad/train_aug_vae.py"
experiment_title = "aug_vae_procgen_aug_step"
default_config_key = "ppo_vae"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
