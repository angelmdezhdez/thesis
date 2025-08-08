import subprocess
import os
import time
import os.path as osp

start = time.time()

# Parámetros a variar
dim_list = [12, 23, 38, 46]
regs_list = ['reg', 'noreg']
input_list = [
    f'/Users/antoniomendez/Desktop/Tesis/thesis/results_dictionary_learning_eco/{dim_list[-1]}n_{dim}a_{reg}'
    for dim in dim_list for reg in regs_list
]

# Parámetros fijos
base_command = [
    "python3", "kmeans_learned_nodes.py",
    "-int", f"[2,{int(dim_list[-1]/2)}]",
    "-st", "/Users/antoniomendez/Desktop/Tesis/Datos/Adj_eco/matrices_estaciones/est_2024.npy",
    "-cell", "/Users/antoniomendez/Desktop/Tesis/thesis/station_cells/station_cells_ecobici_2024_6.pkl",
    "-nodes", "/Users/antoniomendez/Desktop/Tesis/thesis/station_cells/nodes_eco_6.npy",
    "-index", "18",
    "-sys", "ecobici",
    "-part", "6"
]

# Ejecutar una sola vez por input
for inp in input_list:
    # Extraer solo el nombre de la carpeta final de 'inp'
    folder_name = osp.basename(inp)

    # Crear nombre de carpeta de salida usando el identificador del input
    output_folder = f"results_kmeans_nodes/{folder_name}"

    cmd = base_command + [
        "-input", inp,
        "-out", output_folder
    ]

    print("Ejecutando:", " ".join(cmd))
    subprocess.run(cmd)

# Tiempo total y notificación
mensaje = f"Terminé en {(time.time() - start)/60:.2f} minutos"
canal = "aamh_091099_ntfy"
os.system(f'curl -d "{mensaje}" ntfy.sh/{canal}')