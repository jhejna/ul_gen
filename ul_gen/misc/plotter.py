
'''
USAGE:
This plotter assumes that files are stored in a specific way, related to RLpyt and rename things as needed. 
All you should do is copy over the experiment folders to the following structure:

root:
- Experiment 1 (name is the legend label)
    - run_0
    - run_1
    - run_2
- Experiment 2 
    - run_0
    ..
    ..
..

'''
import seaborn as sns
import os
import pandas as pd
from matplotlib import pyplot as plt

def generate_plots(base_dir, x, y):
    assert os.path.isdir(base_dir), "Must pass in a dir"
    experiments = [os.path.join(base_dir, d) for d in os.listdir(base_dir)]
    for experiment in experiments:
        assert any([d.startswith('run') for d in os.listdir(experiment)]), "Experiment %s did not contain a dir that starts with run".format(experiment)
    # Configure SNS (for use in papers)
    sns.set_context(context="paper", font_scale=1.5)
    sns.set_style("darkgrid", {'font.family': 'serif'})

    for experiment in experiments:
        data = None
        for run in [os.path.join(experiment, r) for r in os.listdir(experiment) if r.startswith('run')]:
            run_data = pd.read_csv(os.path.join(run, 'progress.csv'))
            if data is None:
                data = run_data
            else:
                data = data.append(run_data)
        sns.lineplot(x=x, y=y, data=data, ci="sd", label=os.path.basename(experiment))
    
    plt.title(os.path.basename(base_dir))
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, required=True, help="Experiment directory")
    parser.add_argument("--x", "-x", type=str, required=False, default="Diagnostics/CumSteps", help="X axis")
    parser.add_argument("--y", "-y", type=str, required=False, default="ReturnAverage", help="Y axis")
    args = parser.parse_args()
    generate_plots(args.dir, args.x, args.y)
    
    