# libraries
import numpy as np
import pandas as pd
import os
import sys
sys.stdout.flush()
import time

########################################################################################

# funtion that computes the ij entry of the kronecker product of A and B

def kronecker(A,B,i,j):
    return A[i//B.shape[0],j//B.shape[1]]*B[i%B.shape[0],j%B.shape[1]]

# function that finds neigbors in the kronecker product of A and B

def neighbors_kron(A,B,i):
    n = []
    for j in range(A.shape[1]*B.shape[1]):
        if kronecker(A,B,i,j) != 0:
            n.append(j)
    return n

########################################################################################

# funtion that uses GGRF
def vector_rf_kron(W1, W2, d, f_vec, p_h, node, random_walks = 100, h = 100):
    '''
    This funtion computes the feature vector of a node using GGRF
    Args:
        W1: Adjacency matrix of the first graph
        W2: Adjacency matrix of the second graph
        d: Degree vector
        f_vec: Function to compute modulation of the random walk
        p_h: Probability of stopping the random walk
        node: Node of interest
        random_walks: Number of random walks
        h: Default value
    Returns:
        phi: Feature vector of the node
    '''
    # Initial values
    n = h
    phi = np.zeros(len(d))
    m = random_walks
    f_m = f_vec(n)

    for w in range(m):
        # Initial values for the random walk
        load = 1
        current_node = node
        terminated = False
        walk_lenght = 0
        
        # Register of the nodes visited
        register = [current_node]
        counter = 0
        while terminated == False:
            
            # In case we require more values of f
            if walk_lenght == n:
                #print("Requerí mas valores de f")
                n = 2 * n
                f_m = f_vec(n)

            # Update the feature vector
            phi[current_node] += load * f_m[walk_lenght]
            # Update the walk length
            walk_lenght += 1

            # Select the next node searching in the neighbors
            neighbors = neighbors_kron(W1,W2,current_node)
            new_node = np.random.choice(neighbors)
            aux = []
            # If the node is already in the register, we search for a new one
            while new_node in register:
                aux.append(new_node)
                new_node = np.random.choice(neighbors)
                if len(aux) == len(neighbors):
                    break
            # If we tried all the neighbors, we select a random one
            if len(aux) == len(neighbors):
                new_node = np.random.choice(neighbors)

            # Update the load
            load = load * (d[current_node] / (1 - p_h))* kronecker(W1,W2,current_node,new_node)

            # Update the current node
            current_node = new_node

            # Update the register
            register.append(current_node)
            counter += 1

            # Check if the random walk is terminated
            terminated = (np.random.uniform(0,0.5) < p_h)
            if counter == 150:
                break

    return phi / m

########################################################################################

# modulation function
def compute_f_vector(f_alpha, n):
    '''
    This function computes the modulation function for a given alpha function and n
    according to the GGRF paper
    Args:
        f_alpha: Alpha function
        n: Number of values to compute
    Returns:
        f: Modulation function of length n
    '''
    alpha = f_alpha(n)
    f = np.zeros(n)

    # Initial values
    f[0] = np.sqrt(alpha[0])
    aux = 2 * f[0]

    f[1] = alpha[1] / aux

    f[2] = (alpha[2] - f[1]**2) / aux

    # Compute the rest of the values
    for i in range(3, n):
        suma = sum(f[i-p] * f[p] for p in range(1, i))
        f[i] = (alpha[i] - suma) / aux

    return f

########################################################################################

# coefficients for a Laplacian kernel

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

########################################################################################

def kernel_graph_random_features(W1, W2, dx, f_vec, Px, Qx, p_h, random_walks = 100):
    '''
    This function computes the kernel value using the random features method
    Args:
        W1: Adjacency matrix of the first graph
        W2: Adjacency matrix of the second graph
        dx: Degree vector
        f_vec: Function to compute modulation of the random walk
        Px: Probability vector with arriving probabilities
        Qx: Probability vector with leaving probabilities
        p_h: Probability of stopping the random walk
        random_walks: Number of random walks
    Returns:
        K: Kernel value
    '''
    # Define the matrices to store the feature vectors
    K1 = []
    K2 = []

    # Iteration over the nodes
    for i in range(len(dx)):
        # Compute the feature vector for the node i
        phi1 = vector_rf_kron(W1, W2, dx, f_vec, p_h, i, random_walks)
        K1.append(phi1)
        phi1 = vector_rf_kron(W1, W2, dx, f_vec, p_h, i, random_walks)
        K2.append(phi1)

    # Compute the estimation
    K = np.dot(K1, np.transpose(K2))

    return np.dot(Qx, np.dot(K, Px))

########################################################################################

# function to compute the kernel matrix

def kernel_graph_random_features1(W1, W2, dx, f_vec, Px, Qx, p_h, random_walks = 100):
    '''
    This function computes the kernel value using the random features method with changes to
    save memory
    Args:
        W1: Adjacency matrix of the first graph
        W2: Adjacency matrix of the second graph
        dx: Degree vector
        f_vec: Function to compute modulation of the random walk
        Px: Probability vector with arriving probabilities
        Qx: Probability vector with leaving probabilities
        p_h: Probability of stopping the random walk
        random_walks: Number of random walks
    Returns:
        K: Kernel value
    '''
    # Initial value
    K = 0
    vectors = []
    # Compute and save the first vector
    phi_0 = vector_rf_kron(W1, W2, dx, f_vec, p_h, 0, random_walks)
    aux = np.zeros(len(phi_0))
    for i in range(W1.shape[1]*W2.shape[1]):
        phi_i = vector_rf_kron(W1, W2, dx, f_vec, p_h, i, random_walks)
        vectors.append(phi_i)
        aux[i] = np.dot(phi_0, phi_i)
    
    # Add the first term to the kernel
    K += Qx[0] * np.dot(Px, aux)

    # Compute the rest of the terms
    for i in range(1, W1.shape[1]*W2.shape[1]):
        phi_i = vector_rf_kron(W1, W2, dx, f_vec, p_h, i, random_walks)
        aux = np.zeros(len(phi_i))
        for j in range(W1.shape[1]*W2.shape[1]):
            aux[j] = np.dot(phi_i, vectors[j])
        # Add the term to the kernel
        K += Qx[i] * np.dot(Px, aux)

    return K

########################################################################################

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

########################################################################################

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

########################################################################################

# function to compute the degree vector of a kronecker product

def degree_kron(W1, W2):
    return np.kron(W1.sum(axis = 1), W2.sum(axis = 1))

########################################################################################

data_2019 = pd.read_csv('/home/est_posgrado_angel.mendez/random_walks3/2019.csv')

s = 0.18

i1 = np.array([i for i in range(1, 9)])

matrix = np.ones((len(i1), len(i1)))

for i in range(len(i1)):
    for j in range(i+1, len(i1)):
        data1 = data_2019[data_2019['Inicio_del_viaje'].str.startswith(f'2019-01-{i1[i]:02d}')]
        data2 = data_2019[data_2019['Inicio_del_viaje'].str.startswith(f'2019-01-{i1[j]:02d}')]
        print(f'Calculando para i = {i1[i]}, j = {i1[j]}')
        start = time.time()
        sys.stdout.flush()
        counter1 = count_trips_mibici(data1)
        counter2 = count_trips_mibici(data2)
        if counter1 is not None and counter2 is not None:
            m1 = compute_matrix(counter1, normalized=True)
            m2 = compute_matrix(counter2, normalized=True)
            try:
                dx = degree_kron(m1, m2)
                px = np.ones(len(dx)) / len(dx)
                qx = np.ones(len(dx)) / len(dx)
                f_alpha = lambda n: alpha_laplace(s, n, 1)
                f_vec = lambda n: compute_f_vector(f_alpha, n)
                p_h = 0.2
                matrix[i][j] = kernel_graph_random_features(m1, m2, dx, f_vec, px, qx, p_h, random_walks = 80)
                matrix[j][i] = matrix[i][j]
            except Exception as e:
                print(f'Error en [{i1[i]}-{i1[j]}], {e}')
                matrix[i][j] = kernel_graph_random_features1(m1, m2, dx, f_vec, px, qx, p_h, random_walks = 80)
                matrix[j][i] = matrix[i][j]
        else:
            matrix[i][j] = 0
        end = time.time()
        print(f'Tiempo: {end - start}. Resultado: {matrix[i][j]}')
        sys.stdout.flush()

        np.save(f'matrix_[{i1[0]}-{i1[-1]}]_{s}.npy', matrix)

