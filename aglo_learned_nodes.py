# python3 aglo_learned_nodes.py -sys ecobici -part 6 -input /Users/antoniomendez/Desktop/Tesis/thesis/results_dictionary_learning_eco/46n_12a_noreg -st /Users/antoniomendez/Desktop/Tesis/Datos/Adj_eco/matrices_estaciones/est_2024.npy -cell /Users/antoniomendez/Desktop/Tesis/thesis/station_cells/station_cells_ecobici_2024_6.pkl -nodes /Users/antoniomendez/Desktop/Tesis/thesis/station_cells/nodes_eco_6.npy -index 10 -int '[2,5]' -out test_kmeans1

import argparse
import os
import pickle
import numpy as np
import folium
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import abstract_flows.flows as flows
import abstract_flows.arrow as arrow
import abstract_flows.grid as grid
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_samples, silhouette_score


def plot_silhouette(x, y, dir=None):
    sample_silhouette_values = silhouette_samples(x, y)
    silhouette_avg = silhouette_score(x, y)
    fig, ax = plt.subplots(figsize=(8, 6))

    y_lower = 10
    n_clusters = len(np.unique(y))

    for i in range(n_clusters):
        # Valores silhouette de las muestras en el clúster i
        ith_cluster_silhouette_values = sample_silhouette_values[y == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10  # espacio entre clústeres

    ax.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax.set_xlabel("Coeficiente silhouette")
    ax.set_ylabel("Muestras")
    ax.set_title(f"Silhouette plot para {n_clusters} clusters - Promedio: {silhouette_avg:.3f}")
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1])
    plt.savefig(f"{dir}/silhouette_{n_clusters}.png")
    plt.close()

def plot_dendro_with_cutlines(Z, title, dir=None):
    plt.figure(figsize=(6, 6))
    # Distancias a las que se forman fusiones (ordenadas de menor a mayor)
    distances = Z[:, 2]

    # Para k clusters, la línea de corte se dibuja justo por debajo de la fusión (por eso quitamos un pequeño epsilon)
    for d in distances:
        plt.axhline(y=d, color='gray', linestyle='--', linewidth=0.3)

    dendrogram(Z)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Distance")

    plt.tight_layout()

    plt.savefig(f"{dir}/dendrogram.png")
    plt.close()

parser = argparse.ArgumentParser(description='Aglomerative Clustering with Learned Nodes')
parser.add_argument('-sys', '--system', type=str, required=True, help='System type (e.g., ecobici, mibici)')
parser.add_argument('-part', '--partition', type=int, required=True, help='Partition type (e.g., 1, 2, 3)')
parser.add_argument('-input', '--input_dir', type=str, required=True, help='Directory containing the data files')
parser.add_argument('-st', '--stations_location', type=str, required=True, help='Path to the stations location file')
parser.add_argument('-cell', '--cell_info', type=str, required=True, help='Path to the cell information file')
parser.add_argument('-nodes', '--nodes_used', type=str, required=True, help='Path to the nodes used file')
parser.add_argument('-index', '--flow_index', type=int, required=True, help='Index of the flow to process')
parser.add_argument('-int', '--interval', type=str, required=True, help='Interval for number of clusters')
parser.add_argument('-out', '--output_dir', type=str, required=True, help='Directory to save the output files')

args = parser.parse_args()

weight = np.load(args.input_dir + '/weights.npy', allow_pickle=True)
index = args.flow_index
nodes = np.load(args.nodes_used, allow_pickle=True)
interval = args.interval
ints = [int(x) for x in interval[1:-1].split(',')]
interval = [i for i in range(ints[0], ints[1] + 1)]
output_dir = args.output_dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

labels_dir = f"{output_dir}/labels"
if not os.path.exists(labels_dir):
    os.makedirs(labels_dir)

maps_dir = f"{output_dir}/maps"
if not os.path.exists(maps_dir):
    os.makedirs(maps_dir)

silhouettes_dir = f"{output_dir}/silhouettes"
if not os.path.exists(silhouettes_dir):
    os.makedirs(silhouettes_dir)

weight = weight[index]

weight = np.transpose(weight)


for num_clusters in interval:

    labels = AgglomerativeClustering(n_clusters=num_clusters, linkage='complete').fit_predict(weight)

    if args.system == 'ecobici':
        grid_ = grid.Grid(int(1.8*args.partition), args.partition, 'ecobici')
    elif args.system == 'mibici':
        grid_ = grid.Grid(args.partition, args.partition, 'mibici')

    with open(args.cell_info, 'rb') as f:
        cell_info = pickle.load(f)
    
    stations = np.load(args.stations_location, allow_pickle=True)
    map_ = grid_.map_around()

    # Usamos colores distintivos según el número máximo de clusters
    color_palette = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    colors = {i: color_palette[i % len(color_palette)] for i in range(len(set(labels)))}

    # Convertimos nodes en diccionario para búsqueda rápida
    nodes_dict = {tuple(cell): idx for idx, cell in enumerate(nodes)}

    for station in stations:
        station_id = station[0]
        lat = station[1]
        lon = station[2]

        cell = cell_info.get(station_id)
        if cell is None:
            continue

        node_index = nodes_dict.get(tuple(cell))
        if node_index is None or node_index >= len(labels):
            continue

        cluster_label = labels[node_index]
        cluster_color = colors[cluster_label]

        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color=cluster_color,
            fill=True,
            fill_color=cluster_color,
            fill_opacity=0.5,
            popup=f"Station ID: {station_id}\nCluster: {cluster_label}"
        ).add_to(map_)

    map_.save(f"{output_dir}/maps/map_{num_clusters}.html")

    np.save(f"{output_dir}/labels/labels_{num_clusters}.npy", labels)
    plot_silhouette(weight, labels, dir=silhouettes_dir)

    # Dendrogram
Z = linkage(weight, method='ward')
plot_dendro_with_cutlines(Z, title=f"Dendrogram for {num_clusters} Clusters", dir=output_dir)

print(f"Clustering completed. Results saved in {output_dir}.")