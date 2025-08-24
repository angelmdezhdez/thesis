"""
@author: Antonio Mendez
@date: 2024-08-22
@coding: utf-8
"""

#####################################
# Libraries
#####################################

import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
plt.rcParams['figure.dpi'] = 200

#####################################
# Functions
#####################################

def plot_heatmap(data, dates, title, dir):
    plt.figure(figsize=(4, 6))
    plt.imshow(data, cmap='viridis', interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.xticks(ticks=np.arange(len(dates)), labels=dates, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(dir, f"{title}.png"))
    plt.close()


######################################
# Main
######################################

if __name__ == "__main__":

    # python3 plot_nodes_heatmaps.py -flows <flows_path> -dates <dates_path> -indexes <indexes_to_use> -nodes <nodes_to_visualize> -odir <output_directory>

    parser = argparse.ArgumentParser(description="Plot heatmaps from node data.")
    parser.add_argument('-flows', '--flows_path', type=str, help="Path to the flows data file.")
    parser.add_argument('-dates', '--dates_path', type=str, help="Path to the dates data file.")
    parser.add_argument('-indexes', '--indexes_to_use', type=str, help="Comma-separated list of indexes to use.")
    parser.add_argument('-nodes', '--nodes_to_visualize', type=str, help="Comma-separated list of nodes to visualize.")
    parser.add_argument('-odir', '--output_directory', type=str, help="Directory to save the output heatmaps.")
    args = parser.parse_args()

    # Load data
    flows = np.load(args.flows_path)
    dates = np.load(args.dates_path)

    if args.indexes_to_use[0] == 'i':
        indexes = [i for i in range(int(args.indexes_to_use[2:-1].split(',')[0]), int(args.indexes_to_use[2:-1].split(',')[1])+1)]
    elif args.indexes_to_use == 'all':
        indexes = list(range(flows.shape[0]))
    else:
        indexes = [int(i) for i in args.indexes_to_use[1:-1].split(',')]

    flows = flows[indexes]
    dates = dates[indexes]

    if args.nodes_to_visualize[0] == 'i':
        nodes = [i for i in range(int(args.nodes_to_visualize[2:-1].split(',')[0]), int(args.nodes_to_visualize[2:-1].split(',')[1])+1)]
    elif args.nodes_to_visualize == 'all':
        nodes = list(range(flows.shape[2]))
    else:
        nodes = [int(i) for i in args.nodes_to_visualize[1:-1].split(',')]

    odir = args.output_directory
    if not os.path.exists(odir):
        os.makedirs(odir)

    for node in nodes:
        nodes_matrix = []
        for i in range(len(flows)):
            flow = flows[i].T
            nodes_matrix.append(flow[node])
        nodes_matrix = np.array(nodes_matrix).T

        plot_heatmap(nodes_matrix, dates, f"Node_{node}_Heatmap", odir)
    print("Heatmaps generated successfully.")