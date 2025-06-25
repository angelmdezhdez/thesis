import numpy as np
import matplotlib.pyplot as plt

gamma_values = [0.0, 1e-5, 5e-5, 0.0001, 0.0005, 0.001]
system = 'mibici'

def mean_distance(flows_w: np.ndarray, laplacian: np.ndarray) -> float:
    '''Calculate the mean distance between flow vectors and their neighbors.'''
    distances = []
    for i in range(flows_w.shape[0]):
        alpha = flows_w[i].T
        distances.append(np.trace(alpha.T @ laplacian @ alpha))
    return np.mean(distances)

if system == 'ecobici':
    #index_dir = [30, 174, 228, 300, 420, 666]
    index_dir = [174]
    laplacian = np.load(f'test_neigh_results/lap_eco1.npy')
else:
    index_dir = [6, 132, 234, 330, 426, 570, 642]
    #index_dir = [642]
    laplacian = np.load(f'test_neigh_results/lap_mibici1.npy')

n_nodes = laplacian.shape[0]

for i, index in enumerate(index_dir): 
    dist = []
    for ii in range(len(gamma_values)):
        if system == 'ecobici':
            flows_w = np.load(f'test_neigh_results/results_ecobici_experiment_{index + ii}/weights.npy')
        else:
            flows_w = np.load(f'test_neigh_results/results_mibici_experiment_{index + ii}/weights.npy')
        dist.append(mean_distance(flows_w, laplacian))
    natoms = flows_w.shape[1]
    plt.plot(gamma_values, dist, marker='o', label=f'Natoms: {natoms}')

plt.xlabel(r'$\gamma$')
plt.ylabel('Mean Distance')
plt.title(fr'Mean Distance vs $\gamma$ ({n_nodes} nodes)')
plt.legend()
plt.grid()
plt.show()