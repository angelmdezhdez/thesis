# librerías
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

# Load data
data_2019 = pd.read_csv('2019.csv')
data = data_2019[data_2019['Inicio_del_viaje'].str.startswith('2019-01') | data_2019['Inicio_del_viaje'].str.startswith('2019-02') | data_2019['Inicio_del_viaje'].str.startswith('2019-03')]
del data_2019
estaciones = leer_matriz('est_2019.txt')
data['Inicio_del_viaje'] = pd.to_datetime(data['Inicio_del_viaje'])
data['Fin_del_viaje'] = pd.to_datetime(data['Fin_del_viaje'])
data['Dia'] = data['Inicio_del_viaje'].dt.day_name()

users_counts = data['Usuario_Id'].value_counts()
users = users_counts.index.tolist()

weekends = ['Saturday', 'Sunday']
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

weeks = [
    ('2019-01-01', '2019-01-06'),
    ('2019-01-07', '2019-01-13'),
    ('2019-01-14', '2019-01-20'),
    ('2019-01-21', '2019-01-27'),
    ('2019-01-28', '2019-02-03'),
    ('2019-02-04', '2019-02-10'),
    ('2019-02-11', '2019-02-17'),
    ('2019-02-18', '2019-02-24'),
    ('2019-02-25', '2019-03-03'),
    ('2019-03-04', '2019-03-10'),
    ('2019-03-11', '2019-03-17'),
    ('2019-03-18', '2019-03-24'),
    ('2019-03-25', '2019-03-31')
]

directorio1 = 'entropy_per_week_with_threshold'
directorio2 = 'entropy_per_week_without_threshold'

if not os.path.exists(directorio1):
    os.makedirs(directorio1)
if not os.path.exists(directorio2):
    os.makedirs(directorio2)

for week in weeks:
    print('Semana del ' + week[0] + ' al ' + week[1])
    sys.stdout.flush()
    week_data = data[(data['Inicio_del_viaje'] >= week[0]) & (data['Inicio_del_viaje'] <= week[1])]
    entropy_weekends_with_threshold = []
    entropy_weekdays_with_threshold = []
    entropy_weekends_without_threshold = []
    entropy_weekdays_without_threshold = []
    for user in users:
        user_data = week_data[week_data['Usuario_Id'] == user]
        if user_data.empty:
            continue
        user_data_weekends = user_data[user_data['Dia'].isin(weekends)]
        user_data_weekdays = user_data[user_data['Dia'].isin(weekdays)]
        counter_user_weekends = count_trips_mibici(user_data_weekends, threshold = 5)
        counter_user_weekdays = count_trips_mibici(user_data_weekdays, threshold = 5)
        counter_user_weekends_without_threshold = count_trips_mibici(user_data_weekends, threshold = 1)
        counter_user_weekdays_without_threshold = count_trips_mibici(user_data_weekdays, threshold = 1)
        if counter_user_weekends is not None:
            entropy_weekends_with_threshold.append(compute_entropy_normalized(counter_user_weekends))
        if counter_user_weekdays is not None:
            entropy_weekdays_with_threshold.append(compute_entropy_normalized(counter_user_weekdays))
        if counter_user_weekends_without_threshold is not None:
            entropy_weekends_without_threshold.append(compute_entropy_normalized(counter_user_weekends_without_threshold))
        if counter_user_weekdays_without_threshold is not None:
            entropy_weekdays_without_threshold.append(compute_entropy_normalized(counter_user_weekdays_without_threshold))

     # Gráfico con umbral
    plt.figure(figsize=(12, 5))  # Tamaño ajustado de la figura
    plt.suptitle('Semana del ' + week[0] + ' al ' + week[1])
    plt.subplots_adjust(wspace=0.4)  # Espacio entre subfiguras
    
    plt.subplot(1, 2, 1)
    plt.hist(entropy_weekdays_with_threshold, bins=15, edgecolor='black', linewidth=1.2, range=(0, 1))
    plt.xlabel('Entropía')
    plt.ylabel('Frecuencia')
    plt.title('Entre semana con umbral de 5')
    plt.ylim(0, max(plt.hist(entropy_weekdays_with_threshold, bins=15, range=(0, 1))[0]) + 5)  # Ajuste del límite superior del eje y

    plt.subplot(1, 2, 2)
    plt.hist(entropy_weekends_with_threshold, bins=15, edgecolor='black', linewidth=1.2, range=(0, 1))
    plt.xlabel('Entropía')
    plt.ylabel('Frecuencia')
    plt.title('Fin de semana con umbral de 5')
    plt.ylim(0, max(plt.hist(entropy_weekends_with_threshold, bins=15, range=(0, 1))[0]) + 5)  # Ajuste del límite superior del eje y

    plt.tight_layout()
    plt.savefig(directorio1 + '/week_' + week[0] + '_' + week[1] + '.png')
    plt.close()
    
    # Gráfico sin umbral
    plt.figure(figsize=(12, 5))
    plt.suptitle('Semana del ' + week[0] + ' al ' + week[1])
    plt.subplots_adjust(wspace=0.4)
    
    plt.subplot(1, 2, 1)
    plt.hist(entropy_weekdays_without_threshold, bins=15, edgecolor='black', linewidth=1.2, range=(0, 1))
    plt.xlabel('Entropía')
    plt.ylabel('Frecuencia')
    plt.title('Entre semana sin umbral')
    plt.ylim(0, max(plt.hist(entropy_weekdays_without_threshold, bins=15, range=(0, 1))[0]) + 5)

    plt.subplot(1, 2, 2)
    plt.hist(entropy_weekends_without_threshold, bins=15, edgecolor='black', linewidth=1.2, range=(0, 1))
    plt.xlabel('Entropía')
    plt.ylabel('Frecuencia')
    plt.title('Fin de semana sin umbral')
    plt.ylim(0, max(plt.hist(entropy_weekends_without_threshold, bins=15, range=(0, 1))[0]) + 5)

    plt.tight_layout()
    plt.savefig(directorio2 + '/week_' + week[0] + '_' + week[1] + '.png')
    plt.close()