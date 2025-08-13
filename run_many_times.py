import subprocess
import os
import time
import os.path as osp

start = time.time()

# Parámetros a variar
# eco
dim_matrix = [[4, 7, 11, 15],
            [6, 12, 18, 22],
            [12, 23, 38, 46],
            [20, 40, 60, 80]]

partitions = [3, 4, 6, 8]

# MiBici
#dim_matrix = [[3, 6, 9, 12],
#            [6, 12, 18, 22],
#            [11, 22, 34, 45],
#            [19, 38, 57, 77]]
#
#partitions = [4, 6, 10, 14]

for i, dim_list in enumerate(dim_matrix):
    part = partitions[i]

    regs_list = ['reg', 'noreg']
    input_list = [
        f'/Users/antoniomendez/Desktop/Tesis/thesis/results_dictionary_learning_eco/{dim_list[-1]}n_{dim}a_{reg}'
        for dim in dim_list for reg in regs_list
    ]

    # Parámetros fijos
    base_command = [
        "python3", "aglo_learned_nodes.py",
        "-int", f"[2,{int(dim_list[-1]/2)}]",
        "-st", "/Users/antoniomendez/Desktop/Tesis/Datos/Adj_eco/matrices_estaciones/est_2024.npy",
        "-cell", f"/Users/antoniomendez/Desktop/Tesis/thesis/station_cells/station_cells_ecobici_2024_{part}.pkl",
        "-nodes", f"/Users/antoniomendez/Desktop/Tesis/thesis/station_cells/nodes_eco_{part}.npy",
        "-index", "18",
        "-sys", "ecobici",
        "-part", f"{part}"
    ]

    # Ejecutar una sola vez por input
    for inp in input_list:
        # Extraer solo el nombre de la carpeta final de 'inp'
        folder_name = osp.basename(inp)

        # Crear nombre de carpeta de salida usando el identificador del input
        output_folder = f"results_aglo_nodes/results_eco/{folder_name}"

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