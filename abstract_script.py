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

dir = '/home/est_posgrado_angel.mendez/af_test/'

system = 'ecobici'

dates = [f'2019-01-{str(i).zfill(2)}' for i in range(1, 16)]

len_mesh = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

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
        current_counter = flows.count_trips_ecobici(current_data, threshold=1)
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
        current_counter = flows.count_trips_mibici(current_data, threshold=1)
        count_data.append(current_counter)

if system == 'ecobici':
    for date in dates:
        current_counter = count_data[dates.index(date)]
        current_stations = np.unique(np.concatenate(current_counter[['Est_A', 'Est_B']].values))
        for mesh in len_mesh:
            print(f'Calculating flows for {date} with mesh {mesh}')
            sys.stdout.flush()
            try:
                current_grid = grid.Grid(int(1.8*mesh), mesh, 'ecobici')
                current_map = current_grid.map_around()
                current_station_cells = flows.stations_and_cells(current_grid.geodataframe(), current_stations, stations)
                current_graph = flows.abstract_flows(current_counter, current_grid.geodataframe(), current_station_cells, stations)
                current_map = flows.plot_abstracted_graph(current_graph, current_map, title=f'Flows Ecobici: {date} - {int(mesh*1.8)}x{mesh}')
                dir_name = f'{dir}abstracted_flows/{date}/'
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                current_map.save(f'{dir_name}ecobici_{date}_{int(mesh*1.8)}x{mesh}.html')
            except Exception as e:
                print(f'Error: {e}')
                sys.stdout.flush()
                continue

else:
    for date in dates:
        current_counter = count_data[dates.index(date)]
        current_stations = np.unique(np.concatenate(current_counter[['Est_A', 'Est_B']].values))
        for mesh in len_mesh:
            print(f'Calculating flows for {date} with mesh {mesh}')
            sys.stdout.flush()
            try:
                current_grid = grid.Grid(mesh, mesh, 'mibici')
                current_map = current_grid.map_around()
                current_station_cells = flows.stations_and_cells(current_grid.geodataframe(), current_stations, stations)
                current_graph = flows.abstract_flows(current_counter, current_grid.geodataframe(), current_station_cells, stations)
                current_map = flows.plot_abstracted_graph(current_graph, current_map, title=f'Flows Mibici: {date} - {mesh}x{mesh}')
                dir_name = f'{dir}abstracted_flows/{date}/'
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                current_map.save(f'{dir_name}mibici_{date}_{mesh}x{mesh}.html')
            except Exception as e:
                print(f'Error: {e}')
                sys.stdout.flush()
                continue
                


