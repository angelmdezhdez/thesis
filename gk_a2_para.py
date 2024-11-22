from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, kron

def calculate_kernel(pair):
    i, j, i1, i2, data_2019 = pair
    # Extraer los datos correspondientes
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
    except:
        return i, j, -1  # Error en el cálculo

# Crear una lista de pares (i, j) y datos necesarios
data_2019 = pd.read_csv('/home/est_posgrado_angel.mendez/random_walks2/2019.csv')
i1 = np.array([i for i in range(1, 16)])
i2 = np.array([i for i in range(16, 31)])
pairs = [(i, j, i1, i2, data_2019) for i in range(len(i1)) for j in range(len(i2))]

# Paralelizar el cálculo usando multiprocessing
if __name__ == '__main__':
    matrix = np.zeros((len(i1), len(i2)))
    with Pool() as pool:
        results = pool.map(calculate_kernel, pairs)
    
    # Rellenar la matriz con los resultados
    for i, j, value in results:
        matrix[i][j] = value

    # Guardar la matriz resultante
    np.save(f'matrix_[{i1[0]}-{i1[-1]}][{i2[0]}-{i2[-1]}].npy', matrix)
