#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import time
import gc
sys.stdout.flush()

# Floyd-Warshall algorithm
def floyd_warshall(AdjMatrix):
    n = len(AdjMatrix)
    cost = np.copy(AdjMatrix)
    cost[cost == 0] = np.inf
    for k in range(n):
        for i in range(n):
            for j in range(n):
                cost[i, j] = min(cost[i, j], cost[i, k] + cost[k, j])
    return cost

# shortest path kernel

def shortest_path_kernel1(S1, S2, k_walk):
    # Obtener índices donde las entradas son finitas
    indices_S1 = np.transpose(np.triu_indices_from(S1))
    indices_S2 = np.transpose(np.triu_indices_from(S2))
    
    # Filtrar valores finitos
    indices_S1 = indices_S1[np.isfinite(S1[indices_S1[:, 0], indices_S1[:, 1]])]
    indices_S2 = indices_S2[np.isfinite(S2[indices_S2[:, 0], indices_S2[:, 1]])]

    # Convertir las entradas relevantes en arrays
    S1_finite = S1[indices_S1[:, 0], indices_S1[:, 1]]
    S2_finite = S2[indices_S2[:, 0], indices_S2[:, 1]]
    
    # Calcular el kernel con producto cartesiano
    K = np.sum([k_walk(d1, d2) for d1 in S1_finite for d2 in S2_finite])
    
    return K

def shortest_path_kernel2(S1, S2, k_walk):
    K = 0
    n = len(S1)
    m = len(S2)
    for i in range(n):
        for j in range(i, n):
            for ii in range(m):
                for jj in range(ii, m):
                    if np.isfinite(S1[i, j]) and np.isfinite(S2[ii, jj]):
                        K += k_walk(S1[i, j], S2[ii, jj])
    return K

def shortest_path_kernel(S1, S2, k_walk):
    try:
        return shortest_path_kernel1(S1, S2, k_walk)
    except Exception as e:
        print(f"Error: {e}, trying another approach")
        sys.stdout.flush()
        return shortest_path_kernel2(S1, S2, k_walk)

# kernel walk functions
def dirac_kernel(a, b):
    if a == b:
        return 1
    else:
        return 0
    
def gaussian_kernel(a,b, sigma = 10):
    return np.exp(-((a-b)**2)/(2*sigma**2))

def count_trips_mibici(data_user, threshold = 5, complement = False):
    viajes_user = data_user.groupby([data_user[['Origen_Id', 'Destino_Id']].min(axis=1), data_user[['Origen_Id', 'Destino_Id']].max(axis=1)]).size().reset_index(name='counts')
    viajes_user.columns = ['Est_A', 'Est_B', 'counts']
    if not complement:
        viajes_user = viajes_user[viajes_user['counts'] >= threshold]
    else:
        viajes_user = viajes_user[viajes_user['counts'] < threshold]
    if viajes_user.empty:
        return None
    total = viajes_user['counts'].sum()
    viajes_user['prob'] = viajes_user['counts']/total
    viajes_user = viajes_user.sort_values(by = 'prob', ascending = False).reset_index(drop=True)
    return viajes_user

def compute_matrix(counter_user, normalized = False, self_loops = False):
    if not self_loops:
        counter_user = counter_user[counter_user['Est_A'] != counter_user['Est_B']]
    vertex = list(set(counter_user['Est_A'].unique().tolist() + counter_user['Est_B'].unique().tolist()))
    matrix = np.zeros((len(vertex), len(vertex)))
    for i in range(len(counter_user)):
        current_trip = counter_user.iloc[i]
        count = current_trip["counts"]
        estA = current_trip["Est_A"]
        estB = current_trip["Est_B"]

        matrix[vertex.index(estA)][vertex.index(estB)] = count
        matrix[vertex.index(estB)][vertex.index(estA)] = count
    if normalized:
        D = np.sum(matrix, axis = 1)
        D = np.diag(D)
        D = np.linalg.inv(np.sqrt(D))
        matrix = D @ matrix @ D
    return matrix

# function to compute the gram matrix
def gram_matrix(data, k_function, normalized = False, save = False, directory = None):
    """This function computes the gram matrix of the data using the kernel function k_function
    Parameters:
    data: list of matrices
    k_function: kernel function which takes two matrices as input
    normalized: boolean, if True the gram matrix is normalized
    save: boolean, if True the gram matrix is saved in the current directory
    directory: string, directory where the gram matrix is saved
    Returns:
    gram: gram matrix of the data
    """
    n = len(data)
    gram = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            gram[i, j] = k_function(data[i], data[j])
            gram[j, i] = gram[i, j]
    if normalized:
        D = np.diag(np.diag(gram))
        D = np.linalg.inv(np.sqrt(D))
        gram = D @ gram @ D
    if save:
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(directory + 'gram_matrix.npy', gram)
    return gram

dir = "/home/est_posgrado_angel.mendez/spk/p1/"

s = 44

# Load data
data_2019 = pd.read_csv('/home/est_posgrado_angel.mendez/spk/2019.csv')
month = "01"

dates = [f"2019-{month}-{str(i).zfill(2)}" for i in range(1, 16)]

data = []

for date in dates:
    current_data = data_2019[data_2019['Inicio_del_viaje'].str.startswith(date)]
    current_counter = count_trips_mibici(current_data)
    current_matrix = compute_matrix(current_counter, self_loops=True)
    current_s = floyd_warshall(current_matrix)
    data.append(current_s)
    del current_data, current_counter, current_matrix, current_s
    gc.collect()

# functions to compute the kernel

k_fnc1 = lambda x, y: shortest_path_kernel(x, y, dirac_kernel)
k_fnc2 = lambda x, y: shortest_path_kernel(x, y, lambda a, b: gaussian_kernel(a, b, sigma = s))

# to compute the gram matrix with the first kernel
print("Computing the gram matrix with the dirac kernel")
sys.stdout.flush()
start = time.time()
gram_matrix(data, k_fnc1, normalized = True, save = True, directory = f'{dir}gram_matrix1/')
end = time.time()
print(f"Time elapsed with the dirac kernel: {end - start}")
sys.stdout.flush()

gc.collect()

# to compute the gram matrix with the second kernel
print("Computing the gram matrix with the gaussian kernel")
sys.stdout.flush()
start = time.time()
gram_matrix(data, k_fnc2, normalized = True, save = True, directory = f'{dir}gram_matrix2/')
end = time.time()
print(f"Time elapsed with the gaussian kernel: {end - start}")
sys.stdout.flush()

gc.collect()