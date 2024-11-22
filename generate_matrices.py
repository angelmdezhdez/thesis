import numpy as np
import os

def leer_matriz(nombre_archivo):
    matriz = []
    with open(nombre_archivo, 'r') as archivo:
        archivo.readline()
        archivo.readline()
        for linea in archivo:
            fila = [float(valor) for valor in linea.strip().split()]
            matriz.append(fila)
    return matriz

years = [2019, 2020, 2021, 2022, 2023, 2024]
dir1 = '/home/user/Desktop/Datos/Adj_mibici/matrices_'
dir1_1 = '/home/user/Desktop/Datos/Adj_mibici_npy/matrices_'

for year in years:
    dir_temp = dir1 + str(year) + '/'
    dir_temp2 = dir1_1 + str(year) + '/'
    if not os.path.exists(dir_temp):
        os.makedirs(dir_temp)
    if not os.path.exists(dir_temp2):
        os.makedirs(dir_temp2)
    for file in os.listdir(dir_temp):
        if file.endswith('.txt'):
            print(file)
            matriz = np.array(leer_matriz(dir_temp + file))
            np.save(dir_temp2 + file[:-4], matriz)