# libraries
import os
import time
import multiprocessing
import gc
import sys
import numpy as np
import pandas as pd
sys.stdout.flush()

##############################################################################################################

# Floyd-Warshall algorithm
def floyd_warshall(AdjMatrix):
    n = len(AdjMatrix)
    cost = np.copy(AdjMatrix)
    cost[cost == 0] = np.inf
    np.fill_diagonal(cost, 0)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                cost[i, j] = min(cost[i, j], cost[i, k] + cost[k, j])
    return cost

##############################################################################################################

def leer_matriz(nombre_archivo):
    matriz = []
    with open(nombre_archivo, 'r') as archivo:
        archivo.readline()
        archivo.readline()
        for linea in archivo:
            fila = [float(valor) for valor in linea.strip().split()]
            matriz.append(fila)
    return matriz

def encontrar_estacion(est, matriz):
    for i in range(len(matriz)):
        if matriz[i][0] == est:
            return matriz[i][1], matriz[i][2]
    return None, None

##############################################################################################################

# kernel walk functions
def dirac_kernel(a, b):
    # Acepta scalars o arrays para comparación
    return np.asarray(a) == np.asarray(b)
    
def gaussian_kernel1(a,b, sigma = 0.1620289):
    return np.exp(-((a-b)**2)*sigma)

def gaussian_kernel2(a,b, sigma = 38.9827449):
    a = np.array(a)
    b = np.array(b)
    return np.exp(-(np.linalg.norm(a-b)**2)*sigma)

##############################################################################################################

# generate index for parallel processing
def index_gen(n):
    for i in range(n):
        for j in range(n):
            yield i, j


# task for parallel processing
def task1(args):
    index, S1, S2, V1, V2, fnc1, fnc2, est = args
    i, j = index

    # Verificar si el valor en S1 es finito
    if not np.isfinite(S1[i, j]):
        return [], [], []

    # Precomputar las estaciones iniciales y finales de V2
    init2_all = np.array([encontrar_estacion(v, est) for v in V2])
    end2_all = np.array([encontrar_estacion(v, est) for v in V2])

    # Filtrar estaciones válidas en S2
    valid_indices = np.where(np.isfinite(S2))
    k_indices, l_indices = valid_indices

    # Obtener valores correspondientes en S2 y estaciones
    valid_S2 = S2[k_indices, l_indices]
    valid_init2 = init2_all[k_indices]
    valid_end2 = end2_all[l_indices]

    # Encontrar estaciones iniciales y finales para el nodo actual
    init1 = encontrar_estacion(V1[i], est)
    end1 = encontrar_estacion(V1[j], est)

    # Calcular fnc1 para valores válidos
    fnc1_values = fnc1(S1[i, j], valid_S2)

    # Calcular fnc2 para todas las combinaciones
    fnc2_init_values = np.array([fnc2(init1, init2) for init2 in valid_init2])
    fnc2_end_values = np.array([fnc2(end1, end2) for end2 in valid_end2])

    # Sumar el producto de las tres funciones
    #s = np.sum(fnc1_values * fnc2_init_values * fnc2_end_values)
    return fnc1_values, fnc2_init_values, fnc2_end_values


def task2(args):
    index, S1, S2, V1, V2, fnc1, fnc2, est = args
    i, j = index
    s = 0
    n = len(S2)
    init1 = encontrar_estacion(V1[i], est)
    end1 = encontrar_estacion(V1[j], est)
    k1 = []
    k2 = []
    k3 = []
    if np.isfinite(S1[i, j]):
        for k in range(n):
            for l in range(n):
                if np.isfinite(S2[k, l]):
                    init2 = encontrar_estacion(V2[k], est)
                    end2 = encontrar_estacion(V2[l], est)
                    k1.append(fnc1(S1[i, j], S2[k, l]))
                    k2.append(fnc2(init1, init2))
                    k3.append(fnc2(end1, end2))
                    #s += fnc1(S1[i, j], S2[k, l])*fnc2(init1, init2)*fnc2(end1, end2)
        return k1, k2, k3
    else:
        return [], [], []
    
def task(args):
    try:
        return task1(args)
    except Exception as e:
        print('Error with args:', args[0], 'Error:', e)
        print('Trying task2...')
        return task2(args)  


# shortest path kernel parallel implementation
def sp_kernel(S1, S2, V1, V2, ker_func1 = None, ker_func2 = None, est = None):
    n = len(S1)
    m = len(S2)
    k = 0
    pool = multiprocessing.Pool(processes=os.cpu_count())
    try:
        tasks = [(index, S1, S2, V1, V2, ker_func1, ker_func2, est) for index in index_gen(n)]
        results_test = pool.imap(task, tasks, chunksize = 100)
        #k = sum(results_test)
    except Exception as e:
        raise e
    finally:
        pool.close()
        pool.join()
    #return k
    k1_list = []
    k2_list = []
    k3_list = []
    for result in results_test:
        k1, k2, k3 = result
        k1_list.extend(k1)
        k2_list.extend(k2)
        k3_list.extend(k3)
    return k1_list, k2_list, k3_list


##############################################################################################################

def count_trips_from_matrix(adj_matrix, stations, threshold=5, complement=False):
    # Ensure the matrix is square and compatible with the stations list
    n = len(stations)
    if adj_matrix.shape != (n, n):
        raise ValueError("The adjacency matrix dimensions must match the length of the stations vector.")

    # Flatten the upper triangle of the matrix (excluding diagonal)
    trip_data = []
    for i in range(n):
        for j in range(i + 1, n):
            weight = adj_matrix[i, j]
            if weight > 0:  # Ignore zero weights
                trip_data.append((stations[i], stations[j], weight))
    
    # Create the DataFrame
    trips_df = pd.DataFrame(trip_data, columns=['Est_A', 'Est_B', 'counts'])

    # Apply the threshold/complement filtering
    if not complement:
        trips_df = trips_df[trips_df['counts'] >= threshold]
    else:
        trips_df = trips_df[trips_df['counts'] < threshold]

    if trips_df.empty:
        return None

    # Compute the probabilities
    total_counts = trips_df['counts'].sum()
    trips_df['prob'] = trips_df['counts'] / total_counts

    # Sort by probability in descending order and reset the index
    trips_df = trips_df.sort_values(by='prob', ascending=False).reset_index(drop=True)

    return trips_df

# compute the counts of trips between stations
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

##############################################################################################################

# compute the matrix of trips between stations

def log_prob_matrix(counter_user, normalized = False, self_loops = False):
    if not self_loops:
        counter_user = counter_user[counter_user['Est_A'] != counter_user['Est_B']]
    vertex = list(set(counter_user['Est_A'].unique().tolist() + counter_user['Est_B'].unique().tolist()))
    matrix = np.zeros((len(vertex), len(vertex)))
    for i in range(len(counter_user)):
        current_trip = counter_user.iloc[i]
        count = -np.log(current_trip["prob"])
        estA = current_trip["Est_A"]
        estB = current_trip["Est_B"]

        matrix[vertex.index(estA)][vertex.index(estB)] = count
        matrix[vertex.index(estB)][vertex.index(estA)] = count
    if normalized:
        D = np.sum(matrix, axis = 1)
        D = np.diag(D)
        D = np.linalg.inv(np.sqrt(D))
        matrix = D @ matrix @ D
    return matrix, vertex

##############################################################################################################

# main
if __name__ == "__main__":
    # Graphs
    nodes_G1 = np.array([157, 158, 249, 275, 276])
    adj_matrix_G1 = np.array([
        [0, 6, 5, 0, 0],
        [6, 0, 0, 41, 32],
        [5, 0, 0, 0, 0],
        [0, 41, 0, 0, 0],
        [0, 32, 0, 0, 0]
    ])

    # Graph 2
    nodes_G2 = np.array([4, 70, 71, 85, 278])
    adj_matrix_G2 = np.array([
        [0, 34, 6, 0, 45],
        [34, 0, 0, 0, 0],
        [6, 0, 0, 85, 0],
        [0, 0, 85, 0, 0],
        [45, 0, 0, 0, 0]
    ])

    # Graph 3
    nodes_G3 = np.array([34, 35, 36, 54, 268])
    adj_matrix_G3 = np.array([
        [0, 44, 0, 11, 21],
        [44, 0, 0, 0, 0],
        [0, 0, 0, 8, 47],
        [11, 0, 8, 0, 5],
        [21, 0, 47, 5, 0]
    ])

    # Graph 4
    nodes_G4 = np.array([34, 35, 54, 268])
    adj_matrix_G4 = np.array([
        [0, 37, 24, 0],
        [37, 0, 45, 0],
        [24, 45, 0, 5],
        [0, 0, 5, 0]
    ])

    # Graph 5
    nodes_G5 = np.array([50, 51, 61, 69, 75])
    adj_matrix_G5 = np.array([
        [0, 20, 0, 40, 0],
        [20, 0, 35, 0, 0],
        [0, 35, 0, 0, 15],
        [40, 0, 0, 0, 0],
        [0, 0, 15, 0, 0]
    ])

    # Graph 6
    nodes_G6 = np.array([149, 226, 228, 232, 233])
    adj_matrix_G6 = np.array([
        [0, 0, 0, 10, 0],
        [0, 0, 0, 0, 15],
        [0, 0, 0, 35, 0],
        [10, 0, 35, 0, 0],
        [0, 15, 0, 0, 0]
    ])

    # Graph 7
    nodes_G7 = np.array([173, 178, 247, 271, 272])
    adj_matrix_G7 = np.array([
        [0, 50, 8, 49, 6],
        [50, 0, 0, 0, 0],
        [8, 0, 0, 0, 0],
        [49, 0, 0, 0, 0],
        [6, 0, 0, 0, 0]
    ])

    # Graph 8
    nodes_G8 = np.array([11, 26, 30, 52, 156])
    adj_matrix_G8 = np.array([
        [0, 45, 20, 0, 10],
        [45, 0, 0, 40, 0],
        [20, 0, 0, 0, 0],
        [0, 40, 0, 0, 0],
        [10, 0, 0, 0, 0]
    ])

    nodes = [nodes_G1, nodes_G2, nodes_G3, nodes_G4, nodes_G5, nodes_G6, nodes_G7, nodes_G8]
    adj_matrices = [adj_matrix_G1, adj_matrix_G2, adj_matrix_G3, adj_matrix_G4, adj_matrix_G5, adj_matrix_G6, adj_matrix_G7, adj_matrix_G8]
    estations = leer_matriz('est_2019.txt')

    dframes = []
    for i in range(len(nodes)):
        dframes.append(count_trips_from_matrix(adj_matrices[i], nodes[i]))

    data1 = []
    data2 = []

    for df in dframes:
        if df is not None:
            current_matrix, current_vertex = log_prob_matrix(df)
            current_s = floyd_warshall(current_matrix)
            data1.append(current_s)
            data2.append(current_vertex)

    print(f'There are {os.cpu_count()} CPUs available.')
    sys.stdout.flush()
    print('Computing...')
    sys.stdout.flush()
    # problems with stations
    for i in range(len(data2)):
        if data2[i] is not None:
            if np.array_equal(np.asarray(data2[i]), np.asarray(nodes[i])):
                print(f'Error with stations in graph {i+1}')
                print('Stations given:', np.asarray(nodes[i]), 'Shape:', np.asarray(nodes[i]).shape)
                print('Stations found:', np.asarray(data2[i]), 'Shape:', np.asarray(data2[i]).shape)
                print('Difference:', set(nodes[i]).difference(data2[i]))
                sys.stdout.flush()
                #break


    start = time.time()
    kernel = lambda x, y, w, z: sp_kernel(x, y, w, z, gaussian_kernel1, gaussian_kernel2, estations)
    n = len(data1)

    for i in range(n):
        for j in range(i, n):
            print(f'Computing ({i}, {j}) entry.', end='\r')
            sys.stdout.flush()
            aux = kernel(data1[i], data1[j], data2[i], data2[j])
            np.save(f'results_test/K_l_{i}_{j}.npy', aux[0])
            np.save(f'results_test/K_i_{i}_{j}.npy', aux[1])
            np.save(f'results_test/K_e_{i}_{j}.npy', aux[2])

    end = time.time()
    print(f'Time elapsed: {end-start} seconds.')


