import subprocess
import os

for i in range(63, 71):
    experiments = [
        ['python3', 'aglo_learned_nodes.py', '-int', '[2,23]', '-st', '/Users/antoniomendez/Desktop/Tesis/Datos/Adj_eco/matrices_estaciones/est_2024.npy', '-cell', 'station_cells/station_cells_ecobici_2024_6.pkl', '-nodes', 'station_cells/nodes_eco_6.npy', '-index', str(i), '-sys', 'ecobici', '-part', '6', '-input', 'results_dict_dist_predict_2024/results_noreg', '-out', f'results_dist_sequence_days/day_{i}_noreg'],
        ['python3', 'aglo_learned_nodes.py', '-int', '[2,23]', '-st', '/Users/antoniomendez/Desktop/Tesis/Datos/Adj_eco/matrices_estaciones/est_2024.npy', '-cell', 'station_cells/station_cells_ecobici_2024_6.pkl', '-nodes', 'station_cells/nodes_eco_6.npy', '-index', str(i), '-sys', 'ecobici', '-part', '6', '-input', 'results_dict_dist_predict_2024/results_reg', '-out', f'results_dist_sequence_days/day_{i}_reg']
    ]

    for cmd in experiments:
        print('Running command:', ' '.join(cmd))
        subprocess.run(cmd)

os.system(f'curl -d "Projects Finished!!!" ntfy.sh/aamh_091099_ntfy')