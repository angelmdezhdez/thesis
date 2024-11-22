# libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.sparse import csr_matrix, csc_matrix, kron, eye, diags

def vector_rf(W, d, f_vec, p_h, node, random_walks=100, h=100):
    '''
    This funtion computes the feature vector of a node using GGRF
    Args:
        W: Adjacency matrix
        d: Degree vector
        f_vec: Function to compute modulation of the random walk
        p_h: Probability of stopping the random walk
        node: Node of interest
        random_walks: Number of random walks
        h: Default value
    Returns:
        phi: Feature vector of the node
    '''
    n = h
    phi = np.zeros(len(d), dtype=np.float32)
    m = random_walks
    f_m = f_vec(n).astype(np.float32)

    for w in range(m):
        load = np.float64(1.0)  # Temporary to avoid overflow
        current_node = node
        terminated = False
        walk_lenght = 0
        register = [current_node]
        counter = 0

        while not terminated:
            if walk_lenght == n:
                n = 2 * n
                f_m = f_vec(n).astype(np.float32)

            phi[current_node] += np.float32(load * f_m[walk_lenght])  # Convert again to float32

            walk_lenght += 1
            neighbors = W[current_node].indices
            new_node = np.random.choice(neighbors)
            aux = []
            while new_node in register:
                aux.append(new_node)
                new_node = np.random.choice(neighbors)
                if len(aux) == len(neighbors):
                    break
            if len(aux) == len(neighbors):
                new_node = np.random.choice(neighbors)

            # Calcular usando float64 y convertir el resultado a float32
            load = np.float64(load * (d[current_node].item() / (1 - p_h)) * W[current_node, new_node])

            current_node = new_node
            register.append(current_node)
            counter += 1

            terminated = (np.random.uniform(0, 0.5) < p_h)
            if counter == 150:
                break

    return phi / np.float32(m)


def compute_f_vector(f_alpha, n):
    '''
    Calcula la función de modulación para una función alpha dada y n
    '''
    alpha = f_alpha(n).astype(np.float32)
    f = np.zeros(n, dtype=np.float32)

    f[0] = np.sqrt(alpha[0])
    aux = 2 * f[0]
    f[1] = alpha[1] / aux
    f[2] = (alpha[2] - f[1]**2) / aux

    for i in range(3, n):
        suma = sum(f[i-p] * f[p] for p in range(1, i))
        f[i] = (alpha[i] - suma) / aux

    return f

def kernel_graph_random_features(Wx, dx, f_vec, Px, Qx, p_h, random_walks=100):
    '''
    Calcula el valor del kernel usando el método de random features
    '''
    K1 = np.zeros(Wx.shape, dtype=np.float32)
    K2 = np.zeros(Wx.shape, dtype=np.float32)

    for i in range(len(dx)):
        phi1 = vector_rf(Wx, dx, f_vec, p_h, i, random_walks).astype(np.float32)
        phi2 = vector_rf(Wx, dx, f_vec, p_h, i, random_walks).astype(np.float32)
        for j in range(len(dx)):
            K1[i][j] = phi1[j]
            K2[i][j] = phi2[j]

    K = K1 @ K2.T
    return np.dot(Qx, np.dot(K, Px).astype(np.float32))

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

def plot_counter(counter_user, est, title, save = False, dir = None):
    vertex = list(set(counter_user['Est_A'].unique().tolist() + counter_user['Est_B'].unique().tolist()))
    opacity = np.linspace(0.1, 0.5, len(counter_user))
    #print(vertex)
    plt.figure(figsize=(10, 6))
    for i in vertex:
        esta = encontrar_estacion(i, est)
        #print(esta)
        plt.scatter(esta[1], esta[0], color='blue')
        plt.text(esta[1] + 0.00001, esta[0] + 0.00001, str(i), fontsize=7, ha='left', va='bottom')
    for i in range(len(counter_user)):
        current_trip = counter_user.iloc[i]
        prob = current_trip["prob"]
        estA = current_trip["Est_A"]
        estB = current_trip["Est_B"]
        if estA == estB:
            plt.scatter(encontrar_estacion(estA, est)[1], encontrar_estacion(estA, est)[0], color='red', marker='*', s=100)
        else:
            aux = np.array([encontrar_estacion(estA, est), encontrar_estacion(estB, est)])
            plt.plot(aux[:,1], aux[:,0], color='black', alpha=opacity[i])
    plt.grid()
    plt.title(f'{title}')def count_trips_mibici(data_user, threshold = 5, complement = False):
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

def plot_counter(counter_user, est, title, save = False, dir = None):
    vertex = list(set(counter_user['Est_A'].unique().tolist() + counter_user['Est_B'].unique().tolist()))
    opacity = np.linspace(0.1, 0.5, len(counter_user))
    #print(vertex)
    plt.figure(figsize=(10, 6))
    for i in vertex:
        esta = encontrar_estacion(i, est)
        #print(esta)
        plt.scatter(esta[1], esta[0], color='blue')
        plt.text(esta[1] + 0.00001, esta[0] + 0.00001, str(i), fontsize=7, ha='left', va='bottom')
    for i in range(len(counter_user)):
        current_trip = counter_user.iloc[i]
        prob = current_trip["prob"]
        estA = current_trip["Est_A"]
        estB = current_trip["Est_B"]
        if estA == estB:
            plt.scatter(encontrar_estacion(estA, est)[1], encontrar_estacion(estA, est)[0], color='red', marker='*', s=100)
        else:
            aux = np.array([encontrar_estacion(estA, est), encontrar_estacion(estB, est)])
            plt.plot(aux[:,1], aux[:,0], color='black', alpha=opacity[i])
    plt.grid()
    plt.title(f'{title}')
    if save:
        directory = f'{dir}/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f'{directory}/{title}.png')
        plt.close()
    else:
        plt.show()

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
        matrix = np.sqrt(D) @ matrix @ np.sqrt(D)
    return matrix
    if save:
        directory = f'{dir}/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f'{directory}/{title}.png')
        plt.close()
    else:
        plt.show()

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
        matrix = np.sqrt(D) @ matrix @ np.sqrt(D)
    return matrix

data_2019 = pd.read_csv('/home/est_posgrado_angel.mendez/random_walks2/2019.csv')
estaciones = leer_matriz('/home/est_posgrado_angel.mendez/random_walks2/est_2019.txt')

i1 = np.array([i for i in range(1, 16)])
i2 = np.array([i for i in range(16, 31)])

matrix = np.zeros((len(i1), len(i2)))

for i in range(len(i1)):
    for j in range(len(i2)):
        data1 = data_2019[data_2019['Inicio_del_viaje'].str.startswith(f'2019-01-{i1[i]:02d}')]
        data2 = data_2019[data_2019['Inicio_del_viaje'].str.startswith(f'2019-01-{i2[j]:02d}')]
        counter1 = count_trips_mibici(data1)
        counter2 = count_trips_mibici(data2)
        if counter1 is not None and counter2 is not None:
            m1 = compute_matrix(counter1, normalized=True)
            m2 = compute_matrix(counter2, normalized=True)
            m1_s = csr_matrix(m1.astype(np.float32))  
            m2_s = csr_matrix(m2.astype(np.float32)) 
            try:
                mx = kron(m1_s, m2_s, format='csr')
                dx = np.sum(mx, axis = 1).astype(np.float32)
                px = (np.ones(len(dx), dtype=np.float32) / len(dx)).astype(np.float32)  
                qx = (np.ones(len(dx), dtype=np.float32) / len(dx)).astype(np.float32)
                f_alpha = lambda n: alpha_laplace(0.18, n, 1).astype(np.float32)
                f_vec = lambda n: compute_f_vector(f_alpha, n)
                p_h = np.float32(0.2)
                matrix[i][j] = kernel_graph_random_features(mx, dx, f_vec, px, qx, p_h)
            except:
                print(f'Error en [{i1[i]}-{i2[j]}]')
                matrix[i][j] = -1
        else:
            matrix[i][j] = 0

np.save(f'matrix_[{i1[0]}-{i1[-1]}][{i2[0]}-{i2[-1]}].npy', matrix)