import subprocess
from itertools import product
import os
import time

start = time.time()

# Parámetros a variar
dim_list = [22, 45, 67, 90]
regs_list = ['reg', 'noreg']
input_list = [f'/Users/antoniomendez/Desktop/Tesis/thesis/results_dictionary_learning_mibici/90n_{i}a_{j}' for i in dim_list for j in regs_list]



# Parámetros fijos
base_command = [
    "python3", "kmeans_learned_nodes.py",
    "-int", "[2,60]",
    "-st", "/Users/antoniomendez/Desktop/Tesis/Datos/Adj_mibici/matrices_estaciones/est_2024.npy",
    "-cell", "/Users/antoniomendez/Desktop/Tesis/thesis/station_cells/station_cells_mibici_2024_16.pkl",
    "-nodes", "/Users/antoniomendez/Desktop/Tesis/thesis/station_cells/nodes_mibici_16.npy",
    "-index", "16",
    '-sys', 'mibici',
    "-part", "16"
]

# Ejecutar combinaciones
for dim, reg, inp in product(dim_list, regs_list, input_list):
    cmd = base_command + [
        "-input", inp,
        "-out", f"results_kmeans_nodes/mibici_test_kmeans_90_{dim}_{reg}"
    ]
    print("Ejecutando:", " ".join(cmd))
    subprocess.run(cmd)

mensaje = f"Terminé en {(time.time() - start)/60:.2f} minutos"
canal = "aamh_091099_ntfy"

comando = f'curl -d "{mensaje}" ntfy.sh/{canal}'
os.system(comando)