import argparse
import matplotlib.pyplot as plt
import os


def list_txt_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]


def plot_all_training_costs(directory):
    file_paths = list_txt_files(directory)
    plt.figure(figsize=(10, 5))
    for file_path in file_paths:
        itrs, costs = read_costs(file_path)
        label = os.path.basename(file_path).split('.')[0]
        plt.plot(itrs, costs, label=label)

    plt.xlabel('Iteration')
    plt.ylabel('Cost')
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
