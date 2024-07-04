import argparse
import matplotlib.pyplot as plt
import os
import numpy as np


def list_txt_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]


def plot_all_training_costs(directory):
    file_paths = list_txt_files(directory)
    plt.figure(figsize=(10, 5))
    for file_path in file_paths:
        itrs, costs = read_costs(file_path)
        # add cubic spline interpolation for smoother plot
        itrs = np.linspace(0, itrs[-1], 300)
        costs = np.interp(itrs, range(len(costs)), costs)

        label = os.path.basename(file_path).split('.')[0]
        plt.plot(itrs, costs, label=label)

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()


def read_costs(filename):
    itrs, costs = [], []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split(',')
            if len(parts) == 2:
                itrs.append(int(parts[0]))
                costs.append(float(parts[1]))
    return itrs, costs


parser = argparse.ArgumentParser(description='Plot loss functions')
parser.add_argument('--gen_frm_dir', type=str, default='results/mnist_predrnn_v2')
args = parser.parse_args()
plot_all_training_costs(f'{args.gen_frm_dir}/loss')
