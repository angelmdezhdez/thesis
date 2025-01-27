# libraries
import os
import time
import multiprocessing
import gc
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    s = 0

    # Verificar si el valor en S1 es finito
    if not np.isfinite(S1[i, j]):
        return 0

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
    s = np.sum(fnc1_values * fnc2_init_values * fnc2_end_values)
    return s


def task2(args):
    index, S1, S2, V1, V2, fnc1, fnc2, est = args
    i, j = index
    s = 0
    n = len(S2)
    init1 = encontrar_estacion(V1[i], est)
    end1 = encontrar_estacion(V1[j], est)
    if np.isfinite(S1[i, j]):
        for k in range(n):
            for l in range(n):
                if np.isfinite(S2[k, l]):
                    init2 = encontrar_estacion(V2[k], est)
                    end2 = encontrar_estacion(V2[l], est)
                    s += fnc1(S1[i, j], S2[k, l])*fnc2(init1, init2)*fnc2(end1, end2)
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
def sp_kernel(S1, S2, V1, V2, ker_func1 = None, ker_func2 = None, est = None):
    n = len(S1)
    m = len(S2)
    k = 0
    pool = multiprocessing.Pool(processes=os.cpu_count())
    try:
        tasks = [(index, S1, S2, V1, V2, ker_func1, ker_func2, est) for index in index_gen(n)]
        results = pool.imap(task, tasks, chunksize = 100)
        k = sum(results)
    except Exception as e:
        raise e
    finally:
        pool.close()
        pool.join()
    return k

##############################################################################################################

# gram matrix computation
def gram_matrix(data1, data2, k_function, normalized=False, save=False, directory=None):
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
    n = len(data1)
    gram = np.zeros((n, n))

    # Compute only the upper triangle of the Gram matrix
    for i in range(n):
        for j in range(i, n):
            gram[i, j] = k_function(data1[i], data1[j], data2[i], data2[j])
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
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, 'gram_matrix.npy'), gram)

    return gram


def gram_matrix_time(data1, data2, k_function, normalized=False, save=False, directory=None):
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
    n = len(data1)
    gram = np.zeros((n, n))
    total_calculations = n * (n + 1) // 2  
    start_time = time.time()
    calculations_done = 0

    # Compute only the upper triangle of the Gram matrix
    for i in range(n):
        for j in range(i, n):
            gram[i, j] = k_function(data1[i], data1[j], data2[i], data2[j])
            if i != j:
                gram[j, i] = gram[i, j]
            # Update progress
            calculations_done += 1
            elapsed_time = time.time() - start_time
            remaining_time = (elapsed_time / calculations_done) * (total_calculations - calculations_done)
            print(f"Computing ({i}, {j}) entry - Estimated remaining time: {remaining_time:.2f} seconds", end="\r")

    # Normalize the Gram matrix if required
    if normalized:
        diag = np.sqrt(np.diag(gram))
        gram = gram / np.outer(diag, diag)

    # Save the Gram matrix to a file if required
    if save:
        if directory is None:
            raise ValueError("Directory must be specified if save is True.")
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, 'gram_matrix.npy'), gram)

    print("\nCompleted.") 
    return gram


def gram_matrix_generalized(data1, data2, k_function, save=False, directory=None):
    """This function computes the Gram matrix of the data using the kernel function k_function.
    
    Parameters:
    data1,2: list of matrices 
    k_function: kernel function which takes two matrices as input
    save: boolean, if True the Gram matrix is saved in the specified directory
    directory: string, directory where the Gram matrix is saved
    
    Returns:
    gram: Gram matrix of the data
    """
    n = len(data1)
    m = len(data2)
    if n != m:
        raise ValueError("Both data sets must have the same length.")
    gram = np.zeros((n, m))

    # Compute only the upper triangle of the Gram matrix
    for i in range(n):
        for j in range(i, m):
            gram[i, j] = k_function(data1[i], data2[j])
            if i != j:
                gram[j, i] = gram[i, j]

    # Save the Gram matrix to a file if required
    if save:
        if directory is None:
            raise ValueError("Directory must be specified if save is True.")
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, 'gram_matrix.npy'), gram)

    return gram

def gram_matrix_time_generalized(data1, data2, k_function, save=False, directory=None):
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
    n = len(data1)
    m = len(data2)
    if n != m:
        raise ValueError("Both data sets must have the same length.")
    gram = np.zeros((n, n))
    total_calculations = n * (n + 1) // 2  
    start_time = time.time()
    calculations_done = 0

    # Compute only the upper triangle of the Gram matrix
    for i in range(n):
        for j in range(i, n):
            gram[i, j] = k_function(data1[i], data2[j])
            if i != j:
                gram[j, i] = gram[i, j]
            # Update progress
            calculations_done += 1
            elapsed_time = time.time() - start_time
            remaining_time = (elapsed_time / calculations_done) * (total_calculations - calculations_done)
            print(f"Computing ({i}, {j}) entry - Estimated remaining time: {remaining_time:.2f} seconds", end="\r")

    # Save the Gram matrix to a file if required
    if save:
        if directory is None:
            raise ValueError("Directory must be specified if save is True.")
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, 'gram_matrix.npy'), gram)

    print("\nCompleted.") 
    return gram


##############################################################################################################

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
    return matrix, vertex

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

def normalize_matrix(matrix):
    diag = np.sqrt(np.diag(matrix))
    m = matrix / np.outer(diag, diag)
    return m

##############################################################################################################

def plot_heatmap(matrix, title, labels, with_values = False, save = False, directory = None):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xticks(np.arange(matrix.shape[1]), labels, rotation=90)
    plt.yticks(np.arange(matrix.shape[0]), labels)
    plt.xticks(np.arange(matrix.shape[1]))
    plt.yticks(np.arange(matrix.shape[0]))
    if with_values:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, f'{matrix[i, j]:.1f}', ha='center', va='center', color='white')
    plt.tight_layout()
    if save:
        os.makedirs(directory, exist_ok=True)
        plt.savefig(os.path.join(directory, title + '.pdf'), bbox_inches='tight')
    else:
        plt.show()
    plt.close()

##############################################################################################################

# main
if __name__ == "__main__":
    dir = '/home/est_posgrado_angel.mendez/Prueba_SPK3/'

    system = 'mibici'

    main_data = pd.read_csv(dir + system + '_2019.csv')
    main_data = main_data[main_data['Genero'] == 'F']
    estations = leer_matriz(dir + 'est_2019.txt')

    #dates1 = ['2019-01-01', '2019-01-02', '2019-01-03', '2019-02-03', '2019-02-04', '2019-02-05', '2019-02-06', '2019-02-07', '2019-03-16', '2019-03-17', '2019-03-18', '2019-03-19', '2019-03-20']
    #dates2 = ['2019-04-29', '2019-04-30', '2019-05-01', '2019-05-02', '2019-05-03', '2019-09-14', '2019-09-15', '2019-09-16', '2019-09-17', '2019-09-18', '2019-11-16', '2019-11-17', '2019-11-18', '2019-11-19', '2019-11-20']
    dates = [f'2019-{j:02d}-{i:02d}' for j in range(3,4) for i in range(1, 32)]
    dates_true = []

    data1 = []
    data2 = []


    for date in dates:
        print('Processing date:', date)
        sys.stdout.flush()
        current_data = main_data[main_data['Inicio_del_viaje'].str.startswith(date)]
        current_counter = count_trips_mibici(current_data)
        if current_counter is not None:
            current_matrix, current_vector = log_prob_matrix(current_counter)
            current_s = floyd_warshall(current_matrix)
            data1.append(current_s)
            data2.append(current_vector)
            dates_true.append(date)



    print('Computing Gram matrix...')
    sys.stdout.flush()
    start = time.time()
    kernel = lambda x, y, w, z: sp_kernel(x, y, w, z, gaussian_kernel1, gaussian_kernel2, estations)
    print(f'There are {os.cpu_count()} CPUs available.')
    sys.stdout.flush()
    K = gram_matrix_time(data1, data2, kernel, normalized=False)
    print('Time:', time.time() - start)
    sys.stdout.flush()
    #plot_heatmap(K1, 'Gram matrix_F', dates_true, save=True, directory=dir+'exp/plots/')
    np.save(dir + 'K.npy', K)
    print('Gram matrix saved.')
    sys.stdout.flush()
    gc.collect()