# libraries
import os
import time
import multiprocessing
import sys
import numpy as np
import argparse
sys.stdout.flush()

############################################################################
# logarithmic transformation
############################################################################

def log_transform(matrix):
    matrix = np.asarray(matrix)
    result = np.zeros_like(matrix, dtype=float)
    positive_mask = matrix > 0
    result[positive_mask] = -np.log(matrix[positive_mask])
    return result

############################################################################
# Floyd-Warshall algorithm
############################################################################

def floyd_warshall(AdjMatrix):
    n = len(AdjMatrix)
    cost = np.copy(AdjMatrix).astype(float)
    cost[cost == 0] = np.inf
    np.fill_diagonal(cost, 0)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                cost[i, j] = min(cost[i, j], cost[i, k] + cost[k, j])
    return cost

############################################################################
# Kernel functions to paths
############################################################################

def dirac_kernel(a, b):
    return np.asarray(a) == np.asarray(b)
    
def gaussian_kernel_ecobici(a,b, sigma = 0.1429):
    return np.exp(-((a-b)**2)*sigma)

def gaussian_kernel_mibici(a,b, sigma = 0.1632):
    return np.exp(-((a-b)**2)*sigma)

#############################################################################
# SP Kernel
#############################################################################

# generate index for parallel processing
def index_gen(n):
    for i in range(n):
        for j in range(n):
            yield i, j


# task for parallel processing
def task1(args):
    index, S1, S2, fnc = args
    i, j = index
    if not np.isfinite(S1[i, j]):
        return 0  # Skip if the value is not finite
    
    # Precompute valid elements in S2
    valid_mask = np.isfinite(S2)
    valid_S2 = S2[valid_mask]

    # Vectorized computation
    s = np.sum(fnc(S1[i, j], valid_S2))
    return s

def task2(args):
    index, S1, S2, fnc = args
    i, j = index
    s = 0
    n = len(S2)
    if np.isfinite(S1[i, j]):
        for k in range(n):
            for l in range(n):
                if np.isfinite(S2[k, l]):
                    s += fnc(S1[i, j], S2[k, l])
        return s
    else:
        return 0
    
def task(args):
    try:
        return task1(args)
    except Exception as e:
        print('Error with args:', args[0], 'Error:', e)
        print('Trying task2...')
        return task2(args)


# shortest path kernel parallel implementation
def sp_kernel(S1, S2, ker_func = None):
    n = len(S1)
    m = len(S2)
    k = 0
    pool = multiprocessing.Pool(processes=os.cpu_count())
    try:
        tasks = [(index, S1, S2, ker_func) for index in index_gen(n)]
        results = pool.imap(task, tasks, chunksize = 100)
        k = sum(results)
    except Exception as e:
        raise e
    finally:
        pool.close()
        pool.join()
    return k

###############################################################################
# Kernel K-means
###############################################################################

def kernel_kmeans(S, k, max_iter=100, kernel_func=None, initial_centroids_random=False):
    n_samples = S.shape[0]
    print('Computing kernel matrix...')
    sys.stdout.flush()
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i, n_samples):
            print(f'\rProcessing kernel matrix: {i+1} of {n_samples}, {j+1} of {n_samples}', end='')
            sys.stdout.flush()
            K[i, j] = kernel_func(S[i], S[j])
            K[j, i] = K[i, j]

    if initial_centroids_random:
        labels = np.random.randint(0, k, size=n_samples)
    else:
        labels = np.arange(n_samples) % k 

    print('\nStarting kernel k-means clustering...')
    sys.stdout.flush()
    for _ in range(max_iter):
        start_time = time.time()
        distances = np.zeros((n_samples, k))
        
        for cluster in range(k):
            idx = np.where(labels == cluster)[0]
            n_c = len(idx)
            if n_c == 0:
                distances[:, cluster] = np.inf
                continue

            K_ic = np.sum(K[:, idx], axis=1)  
            K_cc = np.sum(K[np.ix_(idx, idx)]) 

            distances[:, cluster] = (
                K.diagonal()
                - (2 / n_c) * K_ic
                + (1 / n_c**2) * K_cc
            )

        new_labels = np.argmin(distances, axis=1)
        
        if np.array_equal(new_labels, labels):
            break
        
        labels = new_labels
        elapsed_time = time.time() - start_time
        print(f"\rIteration {_ + 1} completed in {elapsed_time:.2f} seconds", end='')
        sys.stdout.flush()

    centroids = np.array([S[labels == j].mean(axis=0) for j in range(k)])

    return labels, centroids, K

################################################################################
# Silhouette Score
################################################################################

def kernel_silhouette_score(K, labels):
    n_samples = len(labels)
    unique_labels = np.unique(labels)
    k = len(unique_labels)
    
    D = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i, n_samples):
            dist = K[i, i] - 2 * K[i, j] + K[j, j]
            D[i, j] = dist
            D[j, i] = dist

    silhouette_scores = np.zeros(n_samples)

    for i in range(n_samples):
        label_i = labels[i]
        in_cluster = (labels == label_i)
        out_clusters = [l for l in unique_labels if l != label_i]

        if np.sum(in_cluster) > 1:
            a_i = np.mean(D[i, in_cluster][D[i, in_cluster] != 0])
        else:
            a_i = 0  

        b_i = np.inf
        for l in out_clusters:
            idx = (labels == l)
            if np.any(idx):
                b_i = min(b_i, np.mean(D[i, idx]))

        denom = max(a_i, b_i)
        silhouette_scores[i] = 0 if denom == 0 else (b_i - a_i) / denom

    return np.mean(silhouette_scores)


################################################################################
# Main
################################################################################

if __name__ == "__main__":
    # python3 spk_cluster.py -dir spk_kmeans -flows data_mibici_2018_4/flows.npy -sys mibici -k 7 -m_it 100 -init 0 

    parser = argparse.ArgumentParser(description="SP Kernel Clustering")
    parser.add_argument('-dir', '--directory', type=str, default=None, help='Directory to save results', required=False)
    parser.add_argument('-flows', '--flows', type=str, default=None, help='Path to flows file', required=True)
    parser.add_argument('-sys', '--system', type=str, default='ecobici', help='System to use (ecobici or mibici)', required=False)
    parser.add_argument('-k', '--clusters', type=int, default=10, help='Number of clusters', required=False)
    parser.add_argument('-m_it', '--max_iter', type=int, default=100, help='Maximum number of iterations for k-means', required=False)
    parser.add_argument('-init', '--initial_centroids_random', type=int, default=0, help='Use random initial centroids for k-means (1 for yes, 0 for no)', required=False)
    args = parser.parse_args()

    # arguments
    directory = args.directory
    flows_path = args.flows
    system = args.system
    k = args.clusters
    max_iter = args.max_iter
    initial_centroids_random = bool(args.initial_centroids_random)

    flows = np.load(flows_path, allow_pickle=True)
    flows = flows[:20]

    if system == 'ecobici':
        kernel = lambda x, y: sp_kernel(x, y, gaussian_kernel_ecobici)
    elif system == 'mibici':
        kernel = lambda x, y: sp_kernel(x, y, gaussian_kernel_mibici)
    else:
        raise ValueError("System must be 'ecobici' or 'mibici'")
    
    # distances matrices
    distances = []

    for i in range(len(flows)):
        print(f'\rProcessing flow: {i+1} of {len(flows)}', end='')
        sys.stdout.flush()
        distances.append(floyd_warshall(log_transform(flows[i])))

    distances = np.array(distances)

    # kernel k-means clustering
    print('\n\nStarting kernel k-means clustering...')
    sys.stdout.flush()
    start_time = time.time()
    labels, centroids, K = kernel_kmeans(distances, k, max_iter=max_iter, kernel_func=kernel, initial_centroids_random=initial_centroids_random)
    elapsed_time = time.time() - start_time
    print(f"Kernel k-means clustering completed in {elapsed_time:.2f} seconds")
    sys.stdout.flush()
    sc = kernel_silhouette_score(K, labels)
    print('Silhouette score:', sc)
    sys.stdout.flush()

    # save results
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_results = open(os.path.join(directory, 'results.txt'), 'w')
    file_results.write(f"System: {system}\n")
    file_results.write(f"Flows path: {flows_path}\n")
    file_results.write(f"Directory: {directory}\n")
    file_results.write(f'Flows shape: {flows.shape}\n')
    file_results.write(f"Number of clusters: {k}\n")
    file_results.write(f"Max iterations: {max_iter}\n")
    file_results.write(f"Initial centroids random: {initial_centroids_random}\n")
    file_results.write(f"Silhouette score: {sc}\n")
    file_results.close()

    np.save(os.path.join(directory, 'labels.npy'), labels)
    np.save(os.path.join(directory, 'centroids.npy'), centroids)
    np.save(os.path.join(directory, 'kernel_matrix.npy'), K)

    print('Results saved in:', directory)
    sys.stdout.flush()

    os.system(f'curl -d "Finishing kernel KMeans clustering with {system}" ntfy.sh/aamh_091099_ntfy')