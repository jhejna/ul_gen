
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel
import ul_gen
import os

affinity_code = encode_affinity(
    n_cpu_core=4,
    n_gpu=0,
    # hyperthread_offset=20,
    contexts_per_gpu=0,
    n_socket=1
    # cpu_per_run=2,
)

runs_per_setting = 3

variant_levels = list()

tasks = ['gym:CartPole-v1']
values = list(zip(tasks))
dir_names = ["{}".format(*v) for v in values]
keys = [("env", "id")]
variant_levels.append(VariantLevel(keys, values, dir_names))

variants, log_dirs = make_variants(*variant_levels)

print("Variants", variants)
print("Log_dirs", log_dirs)

script = "launch/train_discrete_sac.py"
experiment_title = "cartpole_dis_sac_lr_low"
default_config_key = "discrete_sac"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
