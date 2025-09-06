# libraries
import os
import time
import multiprocessing
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
sys.stdout.flush()

############################################################################
# logarithmic transformation
############################################################################

def log_transform(matrix):
    matrix = np.asarray(matrix)
    result = np.zeros_like(matrix, dtype=float)
    positive_mask = matrix > 0
    result[positive_mask] = 1 / matrix[positive_mask]
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
    
#def gaussian_kernel_ecobici(a,b, sigma = 0.05516): # true sigma = 3.010661068145328
#    return np.exp(-((a-b)**2)*sigma)
#
#def gaussian_kernel_mibici(a,b, sigma = 0.02375): # true sigma = 4.587915301043796
#    return np.exp(-((a-b)**2)*sigma)

def gaussian_kernel(a,b, sigma = 0.05516): # true sigma = 3.010661068145328
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
#def task1(args):
#    index, S1, S2, fnc = args
#    i, j = index
#    if not np.isfinite(S1[i, j]):
#        return 0  # Skip if the value is not finite
#    
#    # Precompute valid elements in S2
#    valid_mask = np.isfinite(S2)
#    valid_S2 = S2[valid_mask]
#
#    # Vectorized computation
#    s = np.sum(fnc(S1[i, j], valid_S2) * dirac_kernel(i, np.arange(len(S2)))[:, None] * dirac_kernel(j, np.arange(len(S2))))
#    return s

def task1(args):
    index, S1, S2, fnc = args
    i, j = index
    if not np.isfinite(S1[i, j]):
        return 0

    valid_mask = np.isfinite(S2)
    valid_indices = np.argwhere(valid_mask)  # posiciones [(k, l), ...]

    if valid_indices.size == 0:
        return 0

    k_vals = valid_indices[:, 0]
    l_vals = valid_indices[:, 1]

    s2_vals = S2[k_vals, l_vals]

    kernel_vals = fnc(S1[i, j], s2_vals)

    dirac_i = dirac_kernel(i, k_vals)  # shape (n_valid,)
    dirac_j = dirac_kernel(j, l_vals)  # shape (n_valid,)
    dirac_mask = dirac_i & dirac_j     # element-wise

    result = np.sum(kernel_vals * dirac_mask)
    return result


def task2(args):
    index, S1, S2, fnc = args
    i, j = index
    s = 0
    n = len(S2)
    if np.isfinite(S1[i, j]):
        for k in range(n):
            for l in range(n):
                if np.isfinite(S2[k, l]):
                    s += fnc(S1[i, j], S2[k, l]) * dirac_kernel(i, k) * dirac_kernel(j, l)
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

def spk_matrix(S, ker_func = None):
    n = len(S)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            print(f'\rComputing kernel between {i} and {j} of an {n}x{n} matrix', end='')
            K[i, j] = ker_func(S[i], S[j])
            K[j, i] = K[i, j]
    return K


def normalize_matrix(K):
    """
    Normaliza una matriz de kernel K de acuerdo a:
        K_norm[i, j] = K[i, j] / (sqrt(K[i, i]) * sqrt(K[j, j]))

    Parámetros
    ----------
    K : np.ndarray
        Matriz de kernel (simétrica, cuadrada).

    Retorna
    -------
    K_norm : np.ndarray
        Matriz de kernel normalizada.
    """
    # Extraemos los valores diagonales (K(xi, xi))
    diag = np.diag(K)
    
    # Evitamos divisiones por cero
    #diag[diag == 0] = 1e-12
    
    # Calculamos denominador como producto externo de sqrt(diag)
    denom = np.outer(np.sqrt(diag), np.sqrt(diag))
    
    # Normalizamos
    K_norm = K / denom
    return K_norm



def make_gk(sigma):
    def gk(a, b):
        return gaussian_kernel(a, b, sigma=sigma)
    return gk

def make_kernel(sigma):
    gk = make_gk(sigma)
    def kernel(x, y):
        return sp_kernel(x, y, gk)
    return kernel

################################################################################
# Main
################################################################################

if __name__ == "__main__":
    multiprocessing.set_start_method("fork")
    # python3 spk_matrix.py -dir spk_kmeans -flows data_mibici_2018_4/flows.npy -sys mibici -ind 'i[2,12]'
    parser = argparse.ArgumentParser(description="SP Kernel Matrix")
    parser.add_argument('-dir', '--directory', type=str, default=None, help='Directory to save results', required=False)
    parser.add_argument('-flows', '--flows', type=str, default=None, help='Path to flows file', required=True)
    parser.add_argument('-sys', '--system', type=str, default='ecobici', help='System to use (ecobici or mibici)', required=False)
    parser.add_argument('-ind', '--indexes', type=str, default='i[0,10]', help='Indexes to use', required=False)
    parser.add_argument('-trips', '--total_trips', type=str, default=None, help='Path to total trips file', required=False)
    parser.add_argument('-s', '--sigma', type=float, default=None, help='Sigma value')
    parser.add_argument('-norm', '--normalize', type=int, default=0, help='Normalize the kernel matrix', required=False)
    args = parser.parse_args()

    # arguments
    directory = args.directory
    flows_path = args.flows
    trips_path = args.total_trips
    system = args.system
    s = args.sigma
    normalized = True if args.normalize == 1 else False

    if system == 'ecobici' and s is None:
        s = 0.05965
    elif system == 'mibici' and s is None:
        s = 0.02387

    def gk(a, b): 
        return gaussian_kernel(a,b, sigma = s)

    def kernel(x, y):
        return sp_kernel(x, y, gk)

    if args.indexes[0] == 'i':
        indexes = [i for i in range(int(args.indexes[2:-1].split(',')[0]), int(args.indexes[2:-1].split(',')[1])+1)]
    else:
        indexes = [int(x) for x in args.indexes[1:-1].split(',')]

    flows = np.load(flows_path, allow_pickle=True)
    total_trips = np.load(trips_path, allow_pickle=True)

    flows_ = flows[indexes]
    total_trips = total_trips[indexes]

    flows = []

    for i in range(len(flows_)):
        flow = flows_[i]
        t = total_trips[i]
        flows.append(t*flow)

    flows = np.array(flows)

    # distances matrices
    distances = []

    for i in range(len(flows)):
        print(f'\rProcessing flow: {i+1} of {len(flows)}', end='')
        sys.stdout.flush()
        distances.append(floyd_warshall(log_transform(flows[i])))

    distances = np.array(distances)

    # kernel matrix
    print('\n\nComputing kernel matrix')
    sys.stdout.flush()
    start_time = time.time()

    K = spk_matrix(distances, kernel)

    if normalized:
        print('Normalizando...')
        sys.stdout.flush()
        K = normalize_matrix(K)

    print(f"Kernel matrix completed")
    sys.stdout.flush()
    elapsed_time = time.time() - start_time
    print(f'Kernel matrix computed in {elapsed_time:.2f} seconds')
    sys.stdout.flush()

    if os.path.exists(directory):
        np.save(os.path.join(directory, 'kernel_matrix.npy'), K)
        with open(os.path.join(directory, 'km_params.txt'), 'w') as f:
            f.write(f'system: {system}\n')
            f.write(f'flows dir: {flows_path}\n')
            f.write(f'sigma: {s}\n')
            f.write(f'indexes: {indexes}\n')
            f.write(f'time: {elapsed_time/60:.2f} minutes\n')
            f.write(f'normalized: {normalized}\n')
        plt.figure(figsize=(10, 8))
        plt.imshow(K, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title('Kernel Matrix Heatmap')
        plt.savefig(os.path.join(directory, 'kernel_matrix.png'))
        plt.close()
    else:
        os.makedirs(directory)
        np.save(os.path.join(directory, 'kernel_matrix.npy'), K)
        with open(os.path.join(directory, 'km_params.txt'), 'w') as f:
            f.write(f'system: {system}\n')
            f.write(f'flows dir: {flows_path}\n')
            f.write(f'sigma: {s}\n')
            f.write(f'indexes: {indexes}\n')
            f.write(f'time: {elapsed_time/60:.2f} minutes\n')
            f.write(f'normalized: {normalized}\n')
        plt.figure(figsize=(10, 8))
        plt.imshow(K, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title('Kernel Matrix Heatmap')
        plt.savefig(os.path.join(directory, 'kernel_matrix.png'))
        plt.close()