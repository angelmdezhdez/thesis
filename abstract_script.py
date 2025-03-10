#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import folium
import os
import sys
sys.stdout.flush()
sys.path.append(os.path.abspath('abstract_flows'))
import arrow
import grid
import flows

##################################################################################################
# load data

dir = '/home/est_posgrado_angel.mendez/af_test2/'

system = 'ecobici'

dates = [f'2019-01-{str(i).zfill(2)}' for i in range(1, 16)]

len_mesh = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
thresholds = [1, 2, 3, 4, 5]

count_data = []

if system == 'ecobici':
    # read files
    stations = np.load(dir + 'data_eco/est_2019.npy')
    full_data = pd.read_csv(dir + 'data_eco/ecobici_2019.csv')

    # get count data
    for date in dates:
        print(f'Counting trips for {date}')
        sys.stdout.flush()
        current_data = full_data[full_data['Fecha_Retiro'].str.contains(date)]
        current_counter = flows.count_trips_ecobici(current_data, threshold=1, directed=True)
        count_data.append(current_counter)
else:
    # read files
    stations = np.load(dir + 'data_mibici/est_2019.npy')
    full_data = pd.read_csv(dir + 'data_mibici/mibici_2019.csv')

    # get count data
    for date in dates:
        print(f'Counting trips for {date}')
        sys.stdout.flush()
        current_data = full_data[full_data['Inicio_del_viaje'].str.contains(date)]
        current_counter = flows.count_trips_mibici(current_data, threshold=1, directed=True)
        count_data.append(current_counter)

if system == 'ecobici':
    for date in dates:
        current_counter = count_data[dates.index(date)]
        current_stations = pd.unique(np.concatenate((current_counter['Est_A'].unique(), current_counter['Est_B'].unique())))
        for mesh in len_mesh:
            print(f'Calculating flows for {date} with mesh {mesh}')
            sys.stdout.flush()
            for t in thresholds:
                print(f'Calculating flows for {date} with mesh {mesh} and threshold {t}')
                sys.stdout.flush()
                try:
                    current_grid = grid.Grid(int(1.8*mesh), mesh, 'ecobici')
                    current_map = current_grid.map_around()
                    current_station_cells = flows.stations_and_cells(current_grid.geodataframe(), current_stations, stations)
                    current_graph_df = flows.abstract_flows(current_counter, current_grid.geodataframe(), current_station_cells, stations, threshold=t)
                    current_map = flows.plot_flows_dataframe(current_graph_df, current_grid.geodataframe(), current_map, title=f'Flows Ecobici: {date} - {int(mesh*1.8)}x{mesh} - threshold {t}')
                    dir_name = f'{dir}abstracted_flows_eco/{date}/'
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    current_map.save(f'{dir_name}ecobici_{date}_{int(mesh*1.8)}x{mesh}_threshold_{t}.html')
                except Exception as e:
                    print(f'Error: {e}')
                    sys.stdout.flush()
                    continue

else:
    for date in dates:
        current_counter = count_data[dates.index(date)]
        current_stations = pd.unique(np.concatenate((current_counter['Est_A'].unique(), current_counter['Est_B'].unique())))
        for mesh in len_mesh:
            print(f'Calculating flows for {date} with mesh {mesh}')
            sys.stdout.flush()
            for t in thresholds:
                try:
                    current_grid = grid.Grid(mesh, mesh, 'mibici')
                    current_map = current_grid.map_around()
                    current_station_cells = flows.stations_and_cells(current_grid.geodataframe(), current_stations, stations)
                    current_graph_df = flows.abstract_flows(current_counter, current_grid.geodataframe(), current_station_cells, stations, threshold=t)
                    current_map = flows.plot_flows_dataframe(current_graph_df, current_grid.geodataframe(), current_map, title=f'Flows Ecobici: {date} - {mesh}x{mesh} - threshold {t}')
                    dir_name = f'{dir}abstracted_flows_mibici/{date}/'
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                    current_map.save(f'{dir_name}mibici_{date}_{mesh}x{mesh}_threshold_{t}.html')
                except Exception as e:
                    print(f'Error: {e}')
                    sys.stdout.flush()
                    continue