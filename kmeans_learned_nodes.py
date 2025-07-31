import argparse
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='KMeans Clustering with Learned Nodes')
parser.add_argument('-input', '--input_dir', type=str, required=True, help='Directory containing the data files')
parser.add_argument('-index', '--flow_index', type=int, required=True, help='Index of the flow to process')
parser.add_argument('-int', '--interval', type=str, required=True, help='Interval for number of clusters')
parser.add_argument('-out', '--output_dir', type=str, required=True, help='Directory to save the output files')

args = parser.parse_args()

weight = np.load(args.input_dir)
index = args.flow_index
interval = args.interval
ints = [int(x) for x in interval[1:-1].split(',')]
interval = [i for i in range(ints[0], ints[1] + 1)]
output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

weight = weight[index]

weight = np.transpose(weight)

inertias = []

for num_clusters in interval:
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(weight)
    labels = kmeans.labels_
    inertias.append(kmeans.inertia_)

    # Save the labels and cluster centers
    np.save(f"{output_dir}/labels_{num_clusters}.npy", labels)
    np.save(f"{output_dir}/centers_{num_clusters}.npy", kmeans.cluster_centers_)

# Plotting the inertia
plt.figure(figsize=(10, 6))
plt.plot(interval, inertias, marker='o')
plt.title('Inertia vs Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid() 
plt.savefig(f"{output_dir}/inertia_plot.png")
plt.close()

print(f"Clustering completed. Results saved in {output_dir}.")