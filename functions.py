import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
import warnings
warnings.filterwarnings("ignore")

# Función para leer matrices
def leer_matriz(nombre_archivo):
    matriz = []
    with open(nombre_archivo, 'r') as archivo:
        archivo.readline()
        archivo.readline()
        for linea in archivo:
            fila = [float(valor) for valor in linea.strip().split()]
            matriz.append(fila)
    return matriz

def count_trips_mibici(data_user):
    viajes_user = data_user.groupby([data_user[['Origen_Id', 'Destino_Id']].min(axis=1), data_user[['Origen_Id', 'Destino_Id']].max(axis=1)]).size().reset_index(name='counts')
    viajes_user.columns = ['Est_A', 'Est_B', 'counts']
    total = viajes_user['counts'].sum()
    viajes_user['prob'] = viajes_user['counts']/total
    return viajes_user

# Función para encontrar la estación en la matriz con posiciones físicas 
def encontrar_estacion(est, matriz):
    for i in range(len(matriz)):
        if matriz[i][0] == est:
            return matriz[i][1], matriz[i][2]
    return None, None


# Función para generar un vector de colores
def generar_colores(n):
    # Obtener un mapa de colores 'hsv' con n colores distintos
    colores = plt.cm.get_cmap('coolwarm', n)
    # Generar el vector de colores
    return [colores(i) for i in range(n)]


# Función para graficar las trayectorias de un usuario
def user_flow(u, d, esta, estaciones, users_counts, name_dir, threshold=5, zoom = True, flow_consider = True):
    #conteo_trayectorias = defaultdict(int)
    latitudes = []
    longitudes = []
    genre = d['Genero'].iloc[0]
    year = d['Año_de_nacimiento'].iloc[0]
    plt.figure(figsize=(12, 12))
    for i in range(len(esta)):
        lat, lon = encontrar_estacion(esta[i], estaciones)
        plt.scatter(lon, lat, color='black')

    # Agrupar los datos por Origen_Id y Destino_Id y contar la cantidad de viajes
    viajes_por_estaciones = d.groupby([d[['Origen_Id', 'Destino_Id']].min(axis=1), d[['Origen_Id', 'Destino_Id']].max(axis=1)]).size().reset_index(name='counts')
    viajes_por_estaciones.columns = ['Estacion_A', 'Estacion_B', 'counts']

    total_viajes = viajes_por_estaciones['counts'].sum()
    viajes_por_estaciones['probabilidad'] = viajes_por_estaciones['counts'] / total_viajes

    if flow_consider:
        viajes_considerados = viajes_por_estaciones[viajes_por_estaciones['counts'] >= threshold]
    else:
        viajes_considerados = viajes_por_estaciones[viajes_por_estaciones['counts'] < threshold]

    # Ordenar los datos por el conteo de viajes en orden descendente
    viajes_considerados = viajes_considerados.sort_values(by='counts', ascending=False)
    total_viajes_considerados = viajes_considerados['counts'].sum()

    entropia = -np.sum(viajes_por_estaciones['probabilidad'] * np.log(viajes_por_estaciones['probabilidad']))
    
    colors = generar_colores(len(viajes_considerados))
        
    num_viajes_considerados = len(viajes_considerados)
    if num_viajes_considerados > 0:
        colors = generar_colores(num_viajes_considerados)
        viajes_considerados = viajes_considerados.reset_index(drop=True)
    
        # Dibujar las trayectorias entre estaciones
        for i, viaje in viajes_considerados.iterrows():
            est_A, est_B = viaje['Estacion_A'], viaje['Estacion_B']
            
            # Coordenadas de las estaciones
            lat_A, lon_A = encontrar_estacion(est_A, estaciones)
            lat_B, lon_B = encontrar_estacion(est_B, estaciones)
            
            # Agregar las coordenadas a las listas
            latitudes.extend([lat_A, lat_B])
            longitudes.extend([lon_A, lon_B])
            
            # Calcular el grosor de la línea basado en la probabilidad
            linewidth = viaje['probabilidad'] * 10
            linewidth = min(max(linewidth, 1), 5)
            
            # Dibujar la línea que conecta las estaciones
            if est_A == est_B:
                # Dibujar un círculo alrededor de la estación
                plt.scatter(lon_A, lat_A, color=colors[i], s=100, label=f'Probabilidad: {viaje["probabilidad"]:.4f}')
            else:
                # Dibujar la línea que conecta las estaciones
                plt.plot([lon_A, lon_B], [lat_A, lat_B], color=colors[i], linewidth=linewidth, label=f'Probabilidad: {viaje["probabilidad"]:.4f}')
    else:
        print(f"No hay viajes considerados para el usuario {u} con el umbral de {threshold}")

    # Ajusta los límites del gráfico para centrar en la media
    if len(latitudes) > 0:
        if zoom:
            lon_max = np.max(longitudes)
            lon_min = np.min(longitudes)
            lat_max = np.max(latitudes)
            lat_min = np.min(latitudes)
            aux_lon = np.abs(lon_max - lon_min) / 10
            aux_lat = np.abs(lat_max - lat_min) / 10
            plt.xlim(lon_min - aux_lon, lon_max + aux_lon)
            plt.ylim(lat_min - aux_lat, lat_max + aux_lat)

        plt.title(f'Usuario {u}, #viajes totales: {users_counts[u]}, #viajes considerados: {total_viajes_considerados}, género {genre}, nacido en {year}, entropia: {entropia:.2f}')
        # Agrupar elementos en el legend si hay más de 42
        plt.grid()
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(handles) > 42:
            grouped_handles = handles[:41]
            grouped_labels = labels[:41]
            grouped_handles.append(plt.Line2D([0], [0], color='black', lw=2))
            grouped_labels.append(f'Y {len(handles) - 41} más...')
            plt.legend(grouped_handles, grouped_labels, loc='best')
        else:
            plt.legend(loc='best')
        #plt.legend(loc='best')
        directory = name_dir if zoom else f'{name_dir}_nozoom'
        if not os.path.exists(directory):
            os.makedirs(directory)
        if zoom:
            plt.savefig(f'{name_dir}/usuario_{u}_trayectorias.png')
        else:
            plt.savefig(f'{name_dir}_nozoom/usuario_{u}_trayectorias.png')
        plt.close()
        plt.clf()