import subprocess

experiments = [
    ['python3', 'aglo_learned_nodes.py', '-int', '[2,23]', '-st', '/Users/antoniomendez/Desktop/Tesis/Datos/Adj_eco/matrices_estaciones/est_2024.npy', '-cell', 'station_cells/station_cells_ecobici_2024_6.pkl', '-nodes', 'station_cells/nodes_eco_6.npy', '-index', '84', '-sys', 'ecobici', '-part', '6', '-input', 'results_dict_predict_2024/results_results_noreg', '-out', 'results_sequence_days/day_84_noreg'],
    ['python3', 'aglo_learned_nodes.py', '-int', '[2,23]', '-st', '/Users/antoniomendez/Desktop/Tesis/Datos/Adj_eco/matrices_estaciones/est_2024.npy', '-cell', 'station_cells/station_cells_ecobici_2024_6.pkl', '-nodes', 'station_cells/nodes_eco_6.npy', '-index', '84', '-sys', 'ecobici', '-part', '6', '-input', 'results_dict_predict_2024/results_results_reg', '-out', 'results_sequence_days/day_84_reg'],
    ['python3', 'aglo_learned_nodes.py', '-int', '[2,23]', '-st', '/Users/antoniomendez/Desktop/Tesis/Datos/Adj_eco/matrices_estaciones/est_2024.npy', '-cell', 'station_cells/station_cells_ecobici_2024_6.pkl', '-nodes', 'station_cells/nodes_eco_6.npy', '-index', '84', '-sys', 'ecobici', '-part', '6', '-input', 'results_dict_predict_2024/results_results_reg_sparse', '-out', 'results_sequence_days/day_84_reg_sparse']
]

for cmd in experiments:
    subprocess.run(cmd)