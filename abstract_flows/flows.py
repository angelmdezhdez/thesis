import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import folium 
import os
import sys
from collections import defaultdict
import base64
import io
import arrow

def find_location_cell(data_grid, station_array):
    #[lat, lon] = station
    station = [station_array[0], station_array[1]]
    if station[0] < 0:
        station = [station[1], station[0]]

    for i in range(len(data_grid)):
        current = data_grid.iloc[i]
        if station[0] >= np.min([current['lat1'], current['lat2']]) and station[0] <= np.max([current['lat1'], current['lat2']]):
            if station[1] >= np.min([current['lon1'], current['lon2']]) and station[1] <= np.max([current['lon1'], current['lon2']]):
                return data_grid.iloc[i]['i'], data_grid.iloc[i]['j']
    return None

def find_station(station_id, station_matrix):
    for i in range(len(station_matrix)):
        if station_matrix[i,0] == station_id:
            # [lat, lon]
            return station_matrix[i,1], station_matrix[i,2]
    return None

def stations_and_cells(data_grid, stations_ids, stations_matrix):
    stations_cells = dict()
    for id in stations_ids:
        try:
            station = find_station(id, stations_matrix)
            cell = find_location_cell(data_grid, station)
        except:
            cell = None
        if cell != None:
            stations_cells[id] = cell
    return stations_cells

def count_trips_ecobici(data_user, threshold = 5, complement = False):
    viajes_user = data_user.groupby([data_user[['Ciclo_Estacion_Retiro', 'Ciclo_Estacion_Arribo']].min(axis=1), data_user[['Ciclo_Estacion_Retiro', 'Ciclo_Estacion_Arribo']].max(axis=1)]).size().reset_index(name='counts')
    viajes_user.columns = ['Est_A', 'Est_B', 'counts']
    if not complement:
        viajes_user = viajes_user[viajes_user['counts'] >= threshold]
    else:
        viajes_user = viajes_user[viajes_user['counts'] < threshold]
    if viajes_user.empty:
        return None
    total = viajes_user['counts'].sum()
    viajes_user['prob'] = viajes_user['counts']/total
    viajes_user = viajes_user.sort_values(by = 'prob', ascending = False).reset_index(drop=True)
    return viajes_user

def count_trips_mibici(data_user, threshold = 5, complement = False):
    viajes_user = data_user.groupby([data_user[['Origen_Id', 'Destino_Id']].min(axis=1), data_user[['Origen_Id', 'Destino_Id']].max(axis=1)]).size().reset_index(name='counts')
    viajes_user.columns = ['Est_A', 'Est_B', 'counts']
    if not complement:
        viajes_user = viajes_user[viajes_user['counts'] >= threshold]
    else:
        viajes_user = viajes_user[viajes_user['counts'] < threshold]
    if viajes_user.empty:
        return None
    total = viajes_user['counts'].sum()
    viajes_user['prob'] = viajes_user['counts']/total
    viajes_user = viajes_user.sort_values(by = 'prob', ascending = False).reset_index(drop=True)
    return viajes_user

def abstract_flows(trips_counted, cells_data, station_cells, station_matrix):
    '''
    trips_counted: dataframe with columns Est_A, Est_B, counts, prob
    station_cells: dictionary with station_id as key and cell as value
    station_matrix: numpy array with station_id, lat, lon
    '''
    cell_flows = defaultdict(lambda: defaultdict(int))

    # agrupar los viajes por celda
    for _, row in trips_counted.iterrows():
        try:
            cell_A = station_cells[row['Est_A']]
            cell_B = station_cells[row['Est_B']]
            cell_flows[cell_A][cell_B] += row['counts']
        except:
            continue

    # calcular los centros de masa de las celdas
    cell_mass_centers = {}
    
    for cell_A in cell_flows:
        total_trips = sum(cell_flows[cell_A].values())
        lat_sum, lon_sum = 0, 0

        # recolectar las estaciones en la celda A
        stations_in_cell_A = [s for s, cell in station_cells.items() if cell == cell_A]
        if stations_in_cell_A:
            for station in stations_in_cell_A:
                lat, lon = find_station(station, station_matrix)
                # Sumar las coordenadas de las estaciones ponderadas por el número de viajes
                lat_sum += lat * trips_counted[trips_counted['Est_A'] == station]['counts'].sum()
                lon_sum += lon * trips_counted[trips_counted['Est_A'] == station]['counts'].sum()

            # Calculamos el centro de masa
            if total_trips > 0:
                center_lat = lat_sum / total_trips
                center_lon = lon_sum / total_trips
                cell_mass_centers[cell_A] = (center_lat, center_lon)
            else:
                # Si no hay viajes, asignamos un valor predeterminado
                cell_mass_centers[cell_A] = (None, None)
        else:
            # Si no hay estaciones en la celda, asignamos un valor predeterminado
            cell_mass_centers[cell_A] = (None, None)

        if find_location_cell(cells_data, cell_mass_centers[cell_A]) != cell_A:
            cell_mass_centers[cell_A] = (np.mean([cells_data.iloc[cell_A[0]]['lat1'], cells_data.iloc[cell_A[0]]['lat2']]), np.mean([cells_data.iloc[cell_A[1]]['lon1'], cells_data.iloc[cell_A[1]]['lon2']]))

    # construir el diccionario de flujos abstractos
    abstracted_flows = {}
    for cell_A in cell_flows:
        abstracted_flows[cell_A] = {}
        for cell_B in cell_flows[cell_A]:
            # verificar si la celda B tiene estaciones
            mass_center_B = cell_mass_centers.get(cell_B, (None, None))
            abstracted_flows[cell_A][cell_B] = {
                'flow_count': cell_flows[cell_A][cell_B],
                'mass_center_A': cell_mass_centers.get(cell_A, (None, None)),
                'mass_center_B': mass_center_B
            }
    return abstracted_flows



def plot_abstracted_graph(abstracted_graph, folium_map, range_weights=[1, 6], title=None):
    # Step 1: Find max and min flow_count
    flow_counts = [
        flow_data['flow_count']
        for cell_A in abstracted_graph
        for cell_B, flow_data in abstracted_graph[cell_A].items()
    ]
    flow_counts = list(set(flow_counts))
    max_flow = max(flow_counts) if flow_counts else 1
    min_flow = min(flow_counts) if flow_counts else 0

    # Step 2: Create linspace for thickness and colors
    min_weight = range_weights[0]
    max_weight = range_weights[1]
    weights = np.linspace(min_weight, max_weight, num=len(flow_counts)) if flow_counts else [min_weight]

    exp = 0.5
    colors = plt.cm.inferno(np.linspace(0, 1, num=len(weights))**exp) if flow_counts else [plt.cm.inferno(0)]

    sorted_flows = sorted(flow_counts)
    flow_to_weight = {flow: weight for flow, weight in zip(sorted_flows, weights)}
    flow_to_color = {flow: color for flow, color in zip(sorted_flows, colors)}

    # Step 3: Find the most relevant node (highest sum of flows and most connections)
    node_relevance = {}
    self_connected_nodes = set()
    for cell_A in abstracted_graph:
        total_weight = sum(flow_data['flow_count'] for flow_data in abstracted_graph[cell_A].values())
        connections = len(abstracted_graph[cell_A])
        node_relevance[cell_A] = total_weight + connections  # Combined score
        if cell_A in abstracted_graph[cell_A]:
            self_connected_nodes.add(cell_A)

    most_relevant_node = max(node_relevance, key=node_relevance.get, default=None)

    # Step 4: Draw edges and mass centers
    for cell_A in abstracted_graph:
        for cell_B, flow_data in abstracted_graph[cell_A].items():
            mass_center_A = flow_data['mass_center_A']
            mass_center_B = flow_data['mass_center_B']
            flow_count = flow_data['flow_count']

            if mass_center_A != (None, None) and mass_center_B != (None, None) and mass_center_A != mass_center_B:
                lat1, lon1 = mass_center_A
                lat2, lon2 = mass_center_B

                weight = flow_to_weight.get(flow_count, min_weight)
                color = flow_to_color.get(flow_count, plt.cm.viridis(0))

                color_hex = '#{:02x}{:02x}{:02x}'.format(
                    int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                )

                # Draw the arc
                arrow.draw_arrow(
                    folium_map,
                    lat1, lon1, lat2, lon2,
                    color=color_hex,
                    weight=weight,
                    tip=6,
                    text=f'Flujos: {int(flow_count)}',
                    radius_fac=1.0
                )

            # Determine the color of the node
            if cell_A in self_connected_nodes:
                node_color = 'green'
                add = '\n(Conexión interna)'
                if cell_A == most_relevant_node:
                    node_color = 'red'
            else:
                node_color = 'black'
                add = ''
                if cell_A == most_relevant_node:
                    node_color = 'red'

            folium.CircleMarker(
                location=[mass_center_A[0], mass_center_A[1]],
                radius=7 if cell_A == most_relevant_node else 5,  
                color=node_color,
                fill=True,
                fill_color=node_color,
                fill_opacity=1.0,
                popup=f'Centro de masa: {cell_A}' + add
            ).add_to(folium_map)

    # Step 5: Add title if provided
    if title:
        title_html = f"""
        <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
        """
        folium_map.get_root().html.add_child(folium.Element(title_html))

    # Step 6: Add color bar
    fig, ax = plt.subplots(figsize=(4, 1))
    cmap = plt.cm.inferno
    norm = plt.Normalize(vmin=min_flow, vmax=max_flow)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='horizontal')
    cb.set_label('Flow Count')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    colorbar_html = f'<img src="data:image/png;base64,{encoded}" style="position: absolute; bottom: 10px; left: 10px; width: 200px;">'
    folium_map.get_root().html.add_child(folium.Element(colorbar_html))

    return folium_map
