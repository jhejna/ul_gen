
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

def generate_plots(dirs, x, y, title):
    for experiment in dirs:
        assert any([d.startswith('run') for d in os.listdir(experiment)]), "Experiment %s did not contain a dir that starts with run".format(experiment)
        # Configure SNS (for use in papers)
        sns.set_context(context="paper", font_scale=1.5)
        sns.set_style("darkgrid", {'font.family': 'serif'})
        data = None
        runs = [os.path.join(experiment, r) for r in os.listdir(experiment) if r.startswith('run')]
        for run in runs:
            run_data = pd.read_csv(os.path.join(run, 'progress.csv'))
            if data is None:
                data = run_data
            else:
                data = data.append(run_data)
            print(data)
        splits = experiment.split('/')[-2:]
        if 'CumSteps' in data.columns:
            x = 'CumSteps'
        sns.lineplot(x=x, y=y, data=data, ci="sd", label='/'.join(splits))
    
    plt.title(title)
    plt.savefig(f'./data/plots/{title}.png')
    # plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", type=str, required=True, action='append', help="Experiment directories")
    parser.add_argument('--title', type=str, required=False, default='', help='Title of plot')
    parser.add_argument("--x", "-x", type=str, required=False, default="Diagnostics/CumSteps", help="X axis")
    parser.add_argument("--y", "-y", type=str, required=False, default="ReturnAverage", help="Y axis")
    args = parser.parse_args()
    generate_plots(args.dir, args.x, args.y, args.title)
    
    