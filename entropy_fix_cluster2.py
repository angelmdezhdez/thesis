import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
sys.stdout.flush()

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

data_2019 = pd.read_csv('2019.csv')
data = data_2019[data_2019['Inicio_del_viaje'].str.startswith('2019-01')]
del data_2019
estaciones = leer_matriz('est_2019.txt')

users = np.load('users_to_test.npy')

def plot_user(counter_user, est, user, save = False, dir = None):
    vertex = list(set(counter_user['Est_A'].unique().tolist() + counter_user['Est_B'].unique().tolist()))
    #print(vertex)
    plt.figure(figsize=(10, 6))
    for i in vertex:
        esta = encontrar_estacion(i, est)
        #print(esta)
        plt.scatter(esta[1], esta[0], color='blue')
        plt.text(esta[1] + 0.00001, esta[0] + 0.00001, str(i), fontsize=7, ha='left', va='bottom')
    for i in range(len(counter_user)):
        current_trip = counter_user.iloc[i]
        estA = current_trip["Est_A"]
        estB = current_trip["Est_B"]
        if estA == estB:
            plt.scatter(encontrar_estacion(estA, est)[1], encontrar_estacion(estA, est)[0], color='red', marker='*', s=100)
        else:
            aux = np.array([encontrar_estacion(estA, est), encontrar_estacion(estB, est)])
            plt.plot(aux[:,1], aux[:,0], color='black', alpha=0.1)
    plt.grid()
    plt.title(f'Usuario {user}')
    if save:
        directory = f'{dir}/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f'{directory}/user_{user}.png')
        plt.close()
    else:
        plt.show()

def compute_entropy_normalized(counter_user):
    '''
    args:
    counter_user: DataFrame with columns Est_A, Est_B, counts, prob given by count_trips_mibici by a specific user
    total_counter: DataFrame with columns Est_A, Est_B, counts, prob given by count_trips_mibici by all users
    return:
    entropy: float with the entropy
    '''
    entropy = 0
    N = len(counter_user)
    if N == 0:
        return None
    else:
        for i in range(N):
            prob = counter_user.iloc[i]['prob']
            entropy -= prob * np.log(prob)
        if N > 1:
            entropy /= np.log(N)
        return entropy
    

for user in users:
    print(user)
    sys.stdout.flush()
    counter_user1 = count_trips_mibici(data[data['Usuario_Id'] == user])
    if counter_user1 is None:
        continue
    counter_user2 = count_trips_mibici(data[data['Usuario_Id'] == user], complement = True)
    if counter_user2 is None:
        continue
    entropy1 = compute_entropy_normalized(counter_user1)
    entropy2 = compute_entropy_normalized(counter_user2)
    if entropy1 is None or entropy2 is None:
        continue
    if entropy1 > 0 and entropy2 > 0:
        plot_user(counter_user1, estaciones, user, save = True, dir = 'users_with_threshold')
        plot_user(counter_user2, estaciones, user, save = True, dir = 'users_without_threshold')
