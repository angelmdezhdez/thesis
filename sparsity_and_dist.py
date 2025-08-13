import numpy as np
import argparse

def mean_distance(flows_w: np.ndarray, laplacian: np.ndarray) -> float:
    '''Calculate the mean distance between flow vectors and their neighbors.'''
    distances = []
    for i in range(flows_w.shape[0]):
        alpha = flows_w[i].T
        distances.append(np.trace(alpha.T @ laplacian @ alpha))
    return np.mean(distances)

def mean_sparsity(flows_w: np.ndarray) -> float:
    '''Calculate the mean sparsity of flow vectors.'''
    sparsities = []
    for i in range(flows_w.shape[0]):
        alpha = flows_w[i].T
        sparsities.append(1 - np.count_nonzero(alpha) / alpha.size)
    return np.mean(sparsities)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate mean distance and sparsity of flow vectors.')
    parser.add_argument('-dir', '--directory', type=str, required=True, help='Path to the directory containing results files.')
    parser.add_argument('-lap', '--laplacian', type=str, required=True, help='Path to the Laplacian matrix file.')
    args = parser.parse_args()

    with open(f"{args.directory}/params.txt", "r") as f:
        text_results = f.read()

    lambda_value = float([line for line in text_results.splitlines() if line.startswith("lambda:")][0].split(": ")[1].strip())
    gamma_value = float([line for line in text_results.splitlines() if line.startswith("gamma:")][0].split(": ")[1].strip())

    flows = np.load(f"{args.directory}/weights.npy")
    laplacian = np.load(args.laplacian)

    print(f'Metrics for the flows in the directory {args.directory}:')
    print(f'Lambda: {lambda_value}, Gamma: {gamma_value}')
    print(f'Mean distance: {mean_distance(flows, laplacian):.4f}')
    print(f'Mean sparsity: {mean_sparsity(flows):.4f}')

    #n_nodes = [77, 90]
    #n_atoms = [
    #    [19, 38, 57, 77],
    #    [22, 45, 67, 90]
    #]
#
    #file_metrics = "metrics_mibici2.txt"
    #with open(file_metrics, "w") as f:
    #    for i, n_node in enumerate(n_nodes):
    #        laplacian = np.load(f"results_dictionary_learning_mibici/laplacian_{n_node}.npy")
    #        for j in range(4):
    #            n_atom = n_atoms[i][j]
    #            weights_noreg = np.load(f"results_dictionary_learning_mibici/{n_node}n_{n_atom}a_noreg/weights.npy")
    #            weights_reg = np.load(f"results_dictionary_learning_mibici/{n_node}n_{n_atom}a_reg/weights.npy")
    #            f.write(f"{n_node} nodes, {n_atom} atoms, no reg... values: & {mean_distance(weights_noreg, laplacian):.4f}        & {mean_sparsity(weights_noreg):.4f}           & {mean_distance(weights_reg, laplacian):.4f}        & {mean_sparsity(weights_reg):.4f}\n")
#
    #print(f"Metrics written to {file_metrics}")
    #print("Mean distance and sparsity calculations completed.")