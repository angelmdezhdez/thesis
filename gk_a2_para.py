from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, kron
import sys
sys.stdout.flush()

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

def alpha_laplace(s, n, d = 1):
    '''
    This function computes the alpha function for a Laplacian kernel
    Args:
        s: Laplacian kernel parameter for regularization
        n: Number of values to compute
        d: Default value (power of the degree)
    Returns:
        alpha: Alpha function of length n
    '''
    alpha = np.ones(n)
    aux1 = 0
    aux2 = 1
    # Recurrent formula
    q = 1 / (1 + s**(-2))
    #q = 1

    for i in range(1, n):
        alpha[i] = ((d + aux1) / aux2) * q * alpha[i-1]
        aux1 += 1
        aux2 += 1

    return alpha

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



def calculate_kernel(pair):
    i, j, i1, i2, data_2019 = pair
    # Extraer los datos correspondientes
    print(f'Calculando para i = {i1[i]}, j = {i2[j]}')
    sys.stdout.flush()
    data1 = data_2019[data_2019['Inicio_del_viaje'].str.startswith(f'2019-01-{i1[i]:02d}')]
    data2 = data_2019[data_2019['Inicio_del_viaje'].str.startswith(f'2019-01-{i2[j]:02d}')]
    
    if data1.empty or data2.empty:
        return i, j, 0  # Resultado vacío
    
    counter1 = count_trips_mibici(data1)
    counter2 = count_trips_mibici(data2)
    
    if counter1 is None or counter2 is None:
        return i, j, 0
    
    # Computar las matrices normalizadas
    m1 = compute_matrix(counter1, normalized=True)
    m2 = compute_matrix(counter2, normalized=True)
    m1_s = csr_matrix(m1.astype(np.float32))
    m2_s = csr_matrix(m2.astype(np.float32))
    
    try:
        # Producto de Kronecker y cálculo del kernel
        mx = kron(m1_s, m2_s, format='csr')
        dx = np.sum(mx, axis=1).astype(np.float32)
        px = np.ones(len(dx), dtype=np.float32) / len(dx)
        qx = np.ones(len(dx), dtype=np.float32) / len(dx)
        f_alpha = lambda n: alpha_laplace(0.18, n, 1).astype(np.float32)
        f_vec = lambda n: compute_f_vector(f_alpha, n)
        p_h = np.float32(0.2)
        kernel_value = kernel_graph_random_features(mx, dx, f_vec, px, qx, p_h)
        return i, j, kernel_value
    except Exception as e:
        print(f'Error en [{i1[i]}-{i2[j]}], {e}')
        return i, j, -1  # Error en el cálculo

# Crear una lista de pares (i, j) y datos necesarios
data_2019 = pd.read_csv('/home/est_posgrado_angel.mendez/random_walks2/2019.csv')
i1 = np.array([i for i in range(1, 16)])
i2 = np.array([i for i in range(16, 31)])
pairs = [(i, j, i1, i2, data_2019) for i in range(len(i1)) for j in range(len(i2))]

# Paralelizar el cálculo usando multiprocessing
if __name__ == '__main__':

    num_cpus = cpu_count()
    matrix = np.zeros((len(i1), len(i2)))
    with Pool(num_cpus) as pool:
        results = pool.map(calculate_kernel, pairs)
    
    # Rellenar la matriz con los resultados
    for i, j, value in results:
        matrix[i][j] = value

    # Guardar la matriz resultante
    np.save(f'matrix_[{i1[0]}-{i1[-1]}][{i2[0]}-{i2[-1]}].npy', matrix)
