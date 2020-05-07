import os
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel
from ul_gen.configs.ppo_procgen_config import configs

affinity_code = encode_affinity(
    n_cpu_core=4,
    n_gpu=1,
    # hyperthread_offset=20,
    n_socket=2
    # cpu_per_run=2,
)

runs_per_setting = 1

variant_levels = list()

tasks = task= [configs['ppo']['env']['id']]
values = list(zip(tasks))
dir_names = ["{}".format(*v) for v in values]
keys = [("env", "id")]
variant_levels.append(VariantLevel(keys, values, dir_names))

variants, log_dirs = make_variants(*variant_levels)

print("Variants", variants)
print("Log_dirs", log_dirs)

script = os.getcwd() + "/ul_gen/launch/train_ppo_procgen.py"
experiment_title = "ppo_procgen"
default_config_key = "ppo"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
