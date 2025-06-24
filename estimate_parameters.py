#libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import folium
import os
import sys
import abstract_flows.arrow as arrow
import abstract_flows.grid as grid
import abstract_flows.flows as flows
sys.stdout.flush()

###############################################################################################
# functions
# Floyd-Warshall algorithm
def floyd_warshall(AdjMatrix):
    n = len(AdjMatrix)
    cost = np.copy(AdjMatrix).astype(float)
    cost[cost == 0] = np.inf
    np.fill_diagonal(cost, 0)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                cost[i, j] = min(cost[i, j], cost[i, k] + cost[k, j])
    return cost

# Print parameters
def parameters_estimator(sample: list)->None:
    ordered_sample = sorted(sample)
    # for mean
    mean = np.mean(ordered_sample)
    print(f"Mean: {mean}")
    sys.stdout.flush()
    # for median
    median = np.median(ordered_sample)
    print(f"Median: {median}")
    sys.stdout.flush()
    # for variance
    variance = np.var(ordered_sample)
    print(f"Variance: {variance}")
    sys.stdout.flush()

def list_hours(init_hour, end_hour, time_interval):
    init_hour = datetime.datetime.strptime(init_hour, '%H:%M:%S')
    end_hour = datetime.datetime.strptime(end_hour, '%H:%M:%S')
    time_interval = datetime.timedelta(minutes=time_interval)

    lh = []
    
    if init_hour > end_hour:
        while init_hour.strftime('%H:%M:%S') != '00:00:00':  
            lh.append(init_hour.strftime('%H:%M:%S'))
            init_hour += time_interval
            if init_hour.strftime('%H:%M:%S') == '00:00:00':  
                break

        init_hour = datetime.datetime.strptime('00:00:00', '%H:%M:%S') 

    while init_hour <= end_hour:
        lh.append(init_hour.strftime('%H:%M:%S'))
        init_hour += time_interval
    
    lh.append(end_hour.strftime('%H:%M:%S'))

    return lh

###################################################################################################

dir = "/home/est_posgrado_angel.mendez/est_parameters/data/"

# list of time intervals
time_intervals = [60, 120, 240, 360, 540, 1080]

system = "mibici"

if system == "mibici":
    data = pd.read_csv(dir + f"{system}_2019.csv")
    dates = pd.to_datetime(data['Inicio_del_viaje'], format='%Y-%m-%d %H:%M:%S').dt.date
    dates = pd.Series(dates).astype(str).unique().tolist()
    dates = sorted(dates)

else:
    data = pd.read_csv(dir + f"{system}_2019.csv")
    dates = data['Fecha_Retiro'].unique
    dates = sorted(dates)

dates_selected = np.random.choice(dates, 100, replace=False)
stations = np.load(dir + "est_2019.npy")

for time_interval in time_intervals:
    list_hours_ = list_hours('06:00:00', '23:59:59', time_interval)
    print(f'System: {system}')
    sys.stdout.flush()
    print(f"Time interval: {time_interval} minutes")
    sys.stdout.flush()
    distances = []
    for date in dates_selected:
        for hour in list_hours_[:-1]:
            if system == "mibici":
                current_data = data[(data['Inicio_del_viaje'] >= date + ' ' + hour) & (data['Inicio_del_viaje'] < date + ' ' + list_hours_[list_hours_.index(hour)+1])]
                current_counter = flows.count_trips_mibici(current_data, threshold=1, directed=False)
                dims = [5,5]
                current_grid = grid.Grid(dims[0], dims[1], system)
            else:
                current_data = data[(data['Fecha_Retiro'] == date) & (data['Hora_Retiro'] >= hour) & (data['Hora_Retiro'] < list_hours_[list_hours_.index(hour)+1])]
                current_counter = flows.count_trips_ecobici(current_data, threshold=1, directed=False)
                dims = [9,5]
                current_grid = grid.Grid(dims[0], dims[1], system)

            current_stations = pd.unique(np.concatenate((current_counter['Est_A'].unique(), current_counter['Est_B'].unique())))
            current_station_cells = flows.stations_and_cells(current_grid.geodataframe(), current_stations, stations)
            current_graph_df = flows.abstract_flows(current_counter, current_grid.geodataframe(), current_station_cells, stations, threshold=1)
            current_weighted_adj_matrix = flows.create_adjacency_matrix(current_graph_df, dims, directed=False)
            current_weighted_adj_matrix = current_weighted_adj_matrix / np.sum(current_weighted_adj_matrix)
            for i in range(len(current_weighted_adj_matrix)):
                for j in range(len(current_weighted_adj_matrix)):
                    if i == j:
                        current_weighted_adj_matrix[i][j] = 0
                    elif current_weighted_adj_matrix[i][j] == 0:
                        current_weighted_adj_matrix[i][j] = 0
                    else:
                        current_weighted_adj_matrix[i][j] = -np.log(current_weighted_adj_matrix[i][j])
            sp_matrix = floyd_warshall(current_weighted_adj_matrix)

            for i in range(len(sp_matrix)):
                for j in range(len(sp_matrix)):
                    for ii in range(len(sp_matrix)):
                        for jj in range(len(sp_matrix)):
                            if i != j and ii != jj:
                                if sp_matrix[i][j] != np.inf and sp_matrix[ii][jj] != np.inf:
                                    distances.append(np.abs(sp_matrix[i][j] - sp_matrix[ii][jj])**2)
    distances = sorted(distances)
    parameters_estimator(distances)

            

