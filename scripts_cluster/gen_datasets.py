import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import time
import folium
import os
import sys
import pickle
import random
import abstract_flows.arrow as arrow
import abstract_flows.grid as grid
import abstract_flows.flows as flows
sys.stdout.flush()


random.seed(42)
np.random.seed(42)

def create_flow_matrix(df: pd.DataFrame):
    """
    Create a square numpy matrix representing flows between grid cells.

    Parameters:
    - df: pd.DataFrame with columns ['i_A', 'j_A', 'i_B', 'j_B', 'flow_count']

    Returns:
    - matrix: np.ndarray of shape (n, n), where n is the number of unique grid cells
    - nodes: list of tuples (i, j), mapping matrix indices to grid cells
    """
    # Extract all unique grid cells
    cells_A = list(zip(df['i_A'], df['j_A']))
    cells_B = list(zip(df['i_B'], df['j_B']))
    all_cells = set(cells_A + cells_B)

    # Map each unique cell to an index
    nodes = sorted(all_cells)  # Optional: sort for consistency
    cell_to_index = {cell: idx for idx, cell in enumerate(nodes)}
    n = len(nodes)

    # Initialize the flow matrix
    matrix = np.zeros((n, n), dtype=float)

    # Populate the matrix
    for _, row in df.iterrows():
        src = (row['i_A'], row['j_A'])
        dst = (row['i_B'], row['j_B'])
        weight = row['flow_count']
        i = cell_to_index[src]
        j = cell_to_index[dst]
        matrix[i, j] = weight

    return matrix, nodes

grid_size = [4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
split_percent = 0.8
years = [2024, 2023, 2022, 2021, 2020, 2019, 2018]

main_data_dir = '/home/est_posgrado_angel.mendez/data_tesis/'

print("Starting dataset generation...MiBici")
sys.stdout.flush()

for gs in grid_size:
    print(f"Generating dataset for grid size {gs}...")
    sys.stdout.flush()
    
    flows_matrices_train = []
    nodes_list_train = []
    flows_matrices_test = []
    nodes_list_test = []

    flows_matrices = []
    nodes_list = []

    max_num_nodes = 0
    max_num_index = 0

    for year in years:
        print(f"Processing year {year}...")
        sys.stdout.flush()
        
        try:
            data_mibici = pd.read_csv(main_data_dir + f'mibici/mibici_{year}.csv')
            estaciones_mibici = np.load(main_data_dir + f'mibici/est_{year}.npy')
        except Exception as e:
            print(f"Error loading data for year {year}: {e}")
            continue

        dates = data_mibici['Inicio'].str[:10]
        dates = dates.unique()

        num_split = int(len(dates) * split_percent)
        split_indexes = np.random.choice(len(dates), num_split, replace=False)

        dates_train = dates[split_indexes]

        for i, date in enumerate(dates):
            print(f"Processing date {date}")
            sys.stdout.flush()
            current_data = data_mibici[data_mibici['Inicio'].str.contains(date)]
            current_counter = flows.count_trips_mibici(current_data, threshold=1, directed=True)
            current_stations = pd.unique(np.concatenate((current_counter['Est_A'].unique(), current_counter['Est_B'].unique())))
            current_grid = grid.Grid(gs, gs, 'mibici')
            current_station_cells = flows.stations_and_cells(current_grid.geodataframe(), current_stations, estaciones_mibici)
            current_graph_df = flows.abstract_flows(current_counter, current_grid.geodataframe(), current_station_cells, estaciones_mibici, threshold=1)

            mat, nod = create_flow_matrix(current_graph_df)

            mat = mat/mat.sum(axis=0, keepdims=True)

            if len(nod) > max_num_nodes:
                max_num_nodes = len(nod)
                max_num_index = i

            flows_matrices.append(mat)
            nodes_list.append(nod)


    ref_mat = flows_matrices[max_num_index]
    ref_nodes = nodes_list[max_num_index]

    out_dir = f'mibici_dataset_{gs}_split{int(split_percent*100)}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ref_index = {node: idx for idx, node in enumerate(ref_nodes)}
    dim = len(ref_nodes)

    for mat, nod, date in zip(flows_matrices, nodes_list, dates):
        aligned_mat = np.zeros((dim, dim), dtype=float)
        node_to_index = {node: idx for idx, node in enumerate(nod)}

        for node_i in nod:
            for node_j in nod:
                if node_i in ref_index and node_j in ref_index:
                    i = ref_index[node_i]
                    j = ref_index[node_j]
                    aligned_mat[i, j] = mat[node_to_index[node_i], node_to_index[node_j]]
        
        if date in dates_train:
            flows_matrices_train.append(aligned_mat)
            nodes_list_train.append(ref_nodes)
        else:
            flows_matrices_test.append(aligned_mat)
            nodes_list_test.append(ref_nodes)

    # Shufle the training and test sets
    dates_test = [date for date in dates if date not in dates_train]

    train_data = list(zip(flows_matrices_train, nodes_list_train, dates_train))
    test_data = list(zip(flows_matrices_test, nodes_list_test, dates_test))
    random.shuffle(train_data)
    random.shuffle(test_data)

    flows_matrices_train, nodes_list_train, dates_train = zip(*train_data)
    flows_matrices_test, nodes_list_test, dates_test = zip(*test_data)

    # Save the datasets

    np.save(out_dir + '/flows_train.npy', np.array(flows_matrices_train))
    np.save(out_dir + '/nodes_train.npy', np.array(nodes_list_train))
    np.save(out_dir + '/dates_train.npy', np.array(dates_train))
    np.save(out_dir + '/flows_test.npy', np.array(flows_matrices_test))
    np.save(out_dir + '/nodes_test.npy', np.array(nodes_list_test))
    np.save(out_dir + '/dates_test.npy', np.array(dates_test))
    print(f"Dataset for grid size {gs} saved in {out_dir}")
    sys.stdout.flush()

print("Dataset generation completed for MiBici.")
sys.stdout.flush()
#########################################################################
print("Starting dataset generation...Ecobici")
sys.stdout.flush()

for gs in grid_size:
    print(f"Generating dataset for grid size {gs}...")
    sys.stdout.flush()
    
    flows_matrices_train = []
    nodes_list_train = []
    flows_matrices_test = []
    nodes_list_test = []

    flows_matrices = []
    nodes_list = []

    max_num_nodes = 0
    max_num_index = 0

    for year in years:
        print(f"Processing year {year}...")
        sys.stdout.flush()
        
        try:
            data_ecobici = pd.read_csv(main_data_dir + f'ecobici/ecobici_{year}.csv')
            estaciones_ecobici = np.load(main_data_dir + f'ecobici/est_{year}.npy')
        except Exception as e:
            print(f"Error loading data for year {year}: {e}")
            continue

        dates = data_ecobici['Inicio'].str[:10]
        dates = dates.unique()

        num_split = int(len(dates) * split_percent)
        split_indexes = np.random.choice(len(dates), num_split, replace=False)

        dates_train = dates[split_indexes]

        for i, date in enumerate(dates):
            print(f"Processing date {date}")
            sys.stdout.flush()
            current_data = data_ecobici[data_ecobici['Inicio'].str.contains(date)]
            current_counter = flows.count_trips_ecobici(current_data, threshold=1, directed=True)
            current_stations = pd.unique(np.concatenate((current_counter['Est_A'].unique(), current_counter['Est_B'].unique())))
            current_grid = grid.Grid(int(1.8*gs), gs, 'ecobici')
            current_station_cells = flows.stations_and_cells(current_grid.geodataframe(), current_stations, estaciones_ecobici)
            current_graph_df = flows.abstract_flows(current_counter, current_grid.geodataframe(), current_station_cells, estaciones_ecobici, threshold=1)

            mat, nod = create_flow_matrix(current_graph_df)

            mat = mat/mat.sum(axis=0, keepdims=True)

            if len(nod) > max_num_nodes:
                max_num_nodes = len(nod)
                max_num_index = i

            flows_matrices.append(mat)
            nodes_list.append(nod)


    ref_mat = flows_matrices[max_num_index]
    ref_nodes = nodes_list[max_num_index]

    out_dir = f'ecobici_dataset_{gs}_split{int(split_percent*100)}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    ref_index = {node: idx for idx, node in enumerate(ref_nodes)}
    dim = len(ref_nodes)
    for mat, nod, date in zip(flows_matrices, nodes_list, dates):
        aligned_mat = np.zeros((dim, dim), dtype=float)
        node_to_index = {node: idx for idx, node in enumerate(nod)}

        for node_i in nod:
            for node_j in nod:
                if node_i in ref_index and node_j in ref_index:
                    i = ref_index[node_i]
                    j = ref_index[node_j]
                    aligned_mat[i, j] = mat[node_to_index[node_i], node_to_index[node_j]]
        
        if date in dates_train:
            flows_matrices_train.append(aligned_mat)
            nodes_list_train.append(ref_nodes)
        else:
            flows_matrices_test.append(aligned_mat)
            nodes_list_test.append(ref_nodes)
    # Shufle the training and test sets
    dates_test = [date for date in dates if date not in dates_train]
    train_data = list(zip(flows_matrices_train, nodes_list_train, dates_train))
    test_data = list(zip(flows_matrices_test, nodes_list_test, dates_test))
    random.shuffle(train_data)
    random.shuffle(test_data)
    flows_matrices_train, nodes_list_train, dates_train = zip(*train_data)
    flows_matrices_test, nodes_list_test, dates_test = zip(*test_data)
    # Save the datasets
    np.save(out_dir + '/flows_train.npy', np.array(flows_matrices_train))
    np.save(out_dir + '/nodes_train.npy', np.array(nodes_list_train))
    np.save(out_dir + '/dates_train.npy', np.array(dates_train))
    np.save(out_dir + '/flows_test.npy', np.array(flows_matrices_test))
    np.save(out_dir + '/nodes_test.npy', np.array(nodes_list_test))
    np.save(out_dir + '/dates_test.npy', np.array(dates_test))
    print(f"Dataset for grid size {gs} saved in {out_dir}")
    sys.stdout.flush()
print("Dataset generation completed for Ecobici.")
sys.stdout.flush()
