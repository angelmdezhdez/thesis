#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import gc
import sys
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

    # Calcular el kernel con producto cartesiano sin crear listas grandes
    K = 0
    for d1 in S1_finite:
        for d2 in S2_finite:
            K += k_walk(d1, d2)

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
    return 1 if a == b else 0

# function to compute the gram matrix

def gram_matrix(data, k_function, normalized=False, save=False, directory=None):
    """This function computes the Gram matrix of the data using the kernel function k_function.
    
    Parameters:
    data: list of matrices
    k_function: kernel function which takes two matrices as input
    normalized: boolean, if True the Gram matrix is normalized
    save: boolean, if True the Gram matrix is saved in the specified directory
    directory: string, directory where the Gram matrix is saved
    
    Returns:
    gram: Gram matrix of the data
    """
    n = len(data)
    gram = np.zeros((n, n))

    # Compute only the upper triangle of the Gram matrix
    for i in range(n):
        for j in range(i, n):
            gram[i, j] = k_function(data[i], data[j])
            if i != j:
                gram[j, i] = gram[i, j]

    # Normalize the Gram matrix if required
    if normalized:
        diag = np.sqrt(np.diag(gram))
        gram = gram / np.outer(diag, diag)

    # Save the Gram matrix to a file if required
    if save:
        if directory is None:
            raise ValueError("Directory must be specified if save is True.")
            sys.stdout.flush()
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, 'gram_matrix.npy'), gram)

    return gram

def count_trips_ecobici(data_user, threshold = 5, complement = False):
    viajes_user = data_user.groupby([data_user[['Ciclo_Estacion_Retiro', 'Ciclo_Estacion_Arribo']].min(axis=1), data_user[['Ciclo_Estacion_Retiro', 'Ciclo_Estacion_Arribo']].max(axis=1)]).size().reset_index(name='counts')
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


dir = '/home/est_posgrado_angel.mendez/spk2/'

data_2019 = pd.read_csv(dir + '2019.csv')

dates = [f"2019-01-{str(i).zfill(2)}" for i in range(1, 16)]

data = []

for date in dates:
    current_data = data_2019[data_2019['Fecha_Retiro'].str.startswith(date)]
    current_counter = count_trips_ecobici(current_data)
    current_matrix = compute_matrix(current_counter, self_loops=True)
    current_s = floyd_warshall(current_matrix)
    data.append(current_s)
    del current_data, current_counter, current_matrix, current_s
    gc.collect()

kernel_fnc = lambda x, y: shortest_path_kernel(x, y, dirac_kernel)

print("Computing the gram matrix with the dirac kernel without normalization")
sys.stdout.flush()
start = time.time()
gram_matrix(data, kernel_fnc, save = True, directory = f'{dir}gram_matrix1/')
end = time.time()
print(f"Time elapsed with the dirac kernel: {end - start}")
sys.stdout.flush()

gc.collect()

print("Computing the gram matrix with the dirac kernel with normalization")
sys.stdout.flush()
start = time.time()
gram_matrix(data, kernel_fnc, normalized = True, save = True, directory = f'{dir}gram_matrix2/')
end = time.time()
print(f"Time elapsed with the dirac kernel: {end - start}")
sys.stdout.flush()

gc.collect()