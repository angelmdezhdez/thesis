import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import folium 
import os
import sys
from collections import defaultdict
import base64
import io
from . import arrow
from . import grid

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

def count_trips_ecobici(data_user, threshold=5, complement=False, directed=False):
    if directed:
        # For directed trips, group by the exact origin and destination
        viajes_user = data_user.groupby(['Ciclo_Estacion_Retiro', 'Ciclo_Estacion_Arribo']).size().reset_index(name='counts')
        viajes_user.columns = ['Est_A', 'Est_B', 'counts']
    else:
        # For undirected trips, group by min and max of origin and destination
        viajes_user = data_user.groupby([data_user[['Ciclo_Estacion_Retiro', 'Ciclo_Estacion_Arribo']].min(axis=1), 
                                      data_user[['Ciclo_Estacion_Retiro', 'Ciclo_Estacion_Arribo']].max(axis=1)]).size().reset_index(name='counts')
        viajes_user.columns = ['Est_A', 'Est_B', 'counts']

    # Apply threshold filtering
    if not complement:
        viajes_user = viajes_user[viajes_user['counts'] >= threshold]
    else:
        viajes_user = viajes_user[viajes_user['counts'] < threshold]

    if viajes_user.empty:
        return None

    # Calculate probabilities
    total = viajes_user['counts'].sum()
    viajes_user['prob'] = viajes_user['counts'] / total

    # Sort by probability
    viajes_user = viajes_user.sort_values(by='prob', ascending=False).reset_index(drop=True)
    return viajes_user


def count_trips_mibici(data_user, threshold=5, complement=False, directed=False):
    if directed:
        # For directed trips, group by the exact origin and destination
        viajes_user = data_user.groupby(['Origen_Id', 'Destino_Id']).size().reset_index(name='counts')
        viajes_user.columns = ['Est_A', 'Est_B', 'counts']
    else:
        # For undirected trips, group by min and max of origin and destination
        viajes_user = data_user.groupby([data_user[['Origen_Id', 'Destino_Id']].min(axis=1), 
                                      data_user[['Origen_Id', 'Destino_Id']].max(axis=1)]).size().reset_index(name='counts')
        viajes_user.columns = ['Est_A', 'Est_B', 'counts']

    # Apply threshold filtering
    if not complement:
        viajes_user = viajes_user[viajes_user['counts'] >= threshold]
    else:
        viajes_user = viajes_user[viajes_user['counts'] < threshold]

    if viajes_user.empty:
        return None

    # Calculate probabilities
    total = viajes_user['counts'].sum()
    viajes_user['prob'] = viajes_user['counts'] / total

    # Sort by probability
    viajes_user = viajes_user.sort_values(by='prob', ascending=False).reset_index(drop=True)
    return viajes_user


def abstract_flows(trips_counted, cells_data, station_cells, station_matrix, threshold=1):
    '''
    Parameters:
        trips_counted (DataFrame): DataFrame with columns Est_A, Est_B, counts, prob.
        cells_data (DataFrame): DataFrame with cell information (e.g., lat1, lat2, lon1, lon2).
        station_cells (dict): Dictionary with station_id as key and cell as value.
        station_matrix (numpy.ndarray): Array with station_id, lat, lon.
        threshold (int): Minimum flow count to include in the DataFrame (default is 0).

    Returns:
        flows_df (DataFrame): DataFrame with columns i_A, j_A, i_B, j_B, flow_count, mass_center_A, and mass_center_B.
    '''
    cell_flows = defaultdict(lambda: defaultdict(int))

    # Group trips by cell
    for _, row in trips_counted.iterrows():
        try:
            cell_A = station_cells[row['Est_A']]
            cell_B = station_cells[row['Est_B']]
            cell_flows[cell_A][cell_B] += row['counts']
        except KeyError:
            continue

    # Calculate mass centers of the cells
    cell_mass_centers = {}
    
    for cell_A in cell_flows:
        total_trips = sum(cell_flows[cell_A].values())
        lat_sum, lon_sum = 0, 0

        # Collect stations in cell_A
        stations_in_cell_A = [s for s, cell in station_cells.items() if cell == cell_A]
        if stations_in_cell_A:
            for station in stations_in_cell_A:
                lat, lon = find_station(station, station_matrix)
                # Sum coordinates weighted by the number of trips
                lat_sum += lat * trips_counted[trips_counted['Est_A'] == station]['counts'].sum()
                lon_sum += lon * trips_counted[trips_counted['Est_A'] == station]['counts'].sum()

            # Calculate mass center
            if total_trips > 0:
                center_lat = lat_sum / total_trips
                center_lon = lon_sum / total_trips
                cell_mass_centers[cell_A] = (center_lat, center_lon)
            else:
                # If no trips, assign a default value
                cell_mass_centers[cell_A] = (None, None)
        else:
            # If no stations in the cell, assign a default value
            cell_mass_centers[cell_A] = (None, None)

        # Ensure mass center is within the cell
        if find_location_cell(cells_data, cell_mass_centers[cell_A]) != cell_A:
            cell_mass_centers[cell_A] = (
                np.mean([cells_data.iloc[cell_A[0]]['lat1'], cells_data.iloc[cell_A[0]]['lat2']]),
                np.mean([cells_data.iloc[cell_A[1]]['lon1'], cells_data.iloc[cell_A[1]]['lon2']])
            )

    # Build the flows DataFrame
    flows_data = []
    for cell_A in cell_flows:
        for cell_B, flow_count in cell_flows[cell_A].items():
            if flow_count >= threshold:  # Apply threshold
                mass_center_A = cell_mass_centers.get(cell_A, (None, None))
                mass_center_B = cell_mass_centers.get(cell_B, (None, None))
                flows_data.append([
                    cell_A[0], cell_A[1],  # i_A, j_A
                    cell_B[0], cell_B[1],  # i_B, j_B
                    flow_count,            # flow_count
                    mass_center_A,         # mass_center_A
                    mass_center_B          # mass_center_B
                ])

    # Create DataFrame
    flows_df = pd.DataFrame(
        flows_data,
        columns=['i_A', 'j_A', 'i_B', 'j_B', 'flow_count', 'mass_center_A', 'mass_center_B']
    )

    return flows_df



def plot_flows_dataframe(flows_df, cell_data, folium_map, range_weights=[1, 6], title=None):
    '''
    Parameters:
        flows_df (DataFrame): DataFrame with columns i_A, j_A, i_B, j_B, flow_count, mass_center_A, and mass_center_B.
        cell_data (DataFrame): DataFrame with columns i, j, lat1, lon1, lat2, lon2.
        folium_map (folium.Map): Folium map object.
        range_weights (list): Range of weights for edge thickness.
        title (str): Title of the map.
    '''
    # Step 1: Find max and min flow_count
    flow_counts = flows_df['flow_count'].unique()
    max_flow = flow_counts.max() if len(flow_counts) > 0 else 1
    min_flow = flow_counts.min() if len(flow_counts) > 0 else 0

    # Step 2: Create linspace for thickness and colors
    min_weight = range_weights[0]
    max_weight = range_weights[1]
    weights = np.linspace(min_weight, max_weight, num=len(flow_counts)) if len(flow_counts) > 0 else [min_weight]

    exp = 0.5
    colors = plt.cm.inferno(np.linspace(0, 1, num=len(weights))**exp) if len(flow_counts) > 0 else [plt.cm.inferno(0)]

    sorted_flows = sorted(flow_counts)
    flow_to_weight = {flow: weight for flow, weight in zip(sorted_flows, weights)}
    flow_to_color = {flow: color for flow, color in zip(sorted_flows, colors)}

    # Step 3: Find the most relevant node and self-connected nodes
    node_relevance = defaultdict(int)
    self_connected_nodes = set()

    # Calculate node relevance and identify self-connected nodes
    for _, row in flows_df.iterrows():
        cell_A = (row['i_A'], row['j_A'])
        cell_B = (row['i_B'], row['j_B'])
        flow_count = row['flow_count']

        node_relevance[cell_A] += flow_count
        node_relevance[cell_B] += flow_count

        if cell_A == cell_B:
            self_connected_nodes.add(cell_A)


    most_relevant_node = max(node_relevance, key=node_relevance.get, default=None)

    # Step 4: Compute trips starting and ending in each cell
    trips_start = defaultdict(int)
    trips_end = defaultdict(int)

    for _, row in flows_df.iterrows():
        cell_A = (row['i_A'], row['j_A'])
        cell_B = (row['i_B'], row['j_B'])
        flow_count = row['flow_count']

        trips_start[cell_A] += flow_count
        trips_end[cell_B] += flow_count

    # Step 5: Plot rectangles for trips starting and ending in each cell
    for _, row in cell_data.iterrows():
        cell = (row['i'], row['j'])
        lat1, lon1 = row['lat1'], row['lon1']
        lat2, lon2 = row['lat2'], row['lon2']

        if cell in trips_start or cell in trips_end:
            total_trips = trips_start.get(cell, 0) + trips_end.get(cell, 0)
            if total_trips > 0:
                start_percent = trips_start.get(cell, 0) / total_trips
                end_percent = trips_end.get(cell, 0) / total_trips

                # Calculate bounds for the left (end) and right (start) parts of the rectangle
                mid_lat = lat1 + (lat2 - lat1) * end_percent

                # Draw the left part (end trips) in dark gray
                folium.Rectangle(
                    bounds=[(lat1, lon1), (mid_lat, lon2)],
                    color='black',  # Dark gray
                    fill=True,
                    fill_color='#555555',
                    fill_opacity=0.25,
                    popup=f'Celda: {cell}\nViajes que terminan: {end_percent:.2%}'
                ).add_to(folium_map)

                # Draw the right part (start trips) in light gray
                folium.Rectangle(
                    bounds=[(mid_lat, lon1), (lat2, lon2)],
                    color='black',  # Light gray
                    fill=True,
                    fill_color='#AAAAAA',
                    fill_opacity=0.25,
                    popup=f'Celda: {cell}\nViajes que inician: {start_percent:.2%}'
                ).add_to(folium_map)

    # Step 6: Draw edges and mass centers
    for _, row in flows_df.iterrows():
        cell_A = (row['i_A'], row['j_A'])
        cell_B = (row['i_B'], row['j_B'])
        flow_count = row['flow_count']
        mass_center_A = row['mass_center_A']
        mass_center_B = row['mass_center_B']

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

    # Step 7: Plot nodes with appropriate colors
    max_relevance = max(node_relevance.values()) if node_relevance else 1
    green_intensity = np.linspace(100, 255, num=len(node_relevance)) if len(node_relevance) > 0 else [0]
    for cell, relevance in node_relevance.items():
        # Get mass center from flows_df
        try:
            mass_center = flows_df[(flows_df['i_A'] == cell[0]) & (flows_df['j_A'] == cell[1])]['mass_center_A'].iloc[0]
        except:
            mass_center = (None, None)
            
        if mass_center != (None, None):
            lat, lon = mass_center

            # Determine node color
            if cell == most_relevant_node:
                node_color = 'red'
            elif cell in self_connected_nodes:
                # Gradient of green based on relevance
                current_green = int(green_intensity[sorted(node_relevance, key=node_relevance.get).index(cell)])
                node_color = f'#00{current_green:02x}00'  # Green gradient
            else:
                node_color = 'black'

            # Plot the node
            folium.CircleMarker(
                location=[lat, lon],
                radius=7 if cell == most_relevant_node else 5,
                color=node_color,
                fill=True,
                fill_color=node_color,
                fill_opacity=1.0,
                popup=f'Celda: {cell}\nRelevancia: {relevance:.2f}'
            ).add_to(folium_map)

    # Step 8: Add title if provided
    if title:
        title_html = f"""
        <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
        """
        folium_map.get_root().html.add_child(folium.Element(title_html))

    # Step 9: Add color bar
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


def create_adjacency_matrix(flows_df, dims, directed=True):
    '''
    Parameters:
        flows_df (DataFrame): DataFrame with columns i_A, j_A, i_B, j_B, flow_count.
        dims (tuple): Dimensions of the grid (rows, columns).
        directed (bool): If True, the graph is directed; if False, the graph is undirected.

    Returns:
        adjacency_matrix (numpy.ndarray): Adjacency matrix of the graph with size (dims[0]*dims[1], (dims[0]*dims[1])).
        nodes (list): List of nodes (i, j) in the same order as the adjacency matrix.
    '''
    # Initialize the adjacency matrix with zeros
    n = dims[0] * dims[1]
    adjacency_matrix = np.zeros((n, n), dtype=int)
    
    # Create a mapping from (i, j) to a unique index
    def node_to_index(i, j):
        return i * dims[1] + j
    
    # Fill the adjacency matrix
    for _, row in flows_df.iterrows():
        i_A, j_A = row['i_A'], row['j_A']
        i_B, j_B = row['i_B'], row['j_B']
        flow_count = row['flow_count']
        
        # Convert (i, j) to indices
        idx_A = node_to_index(i_A, j_A)
        idx_B = node_to_index(i_B, j_B)
        
        # Update the adjacency matrix
        adjacency_matrix[idx_A, idx_B] += flow_count
        
        if not directed:
            adjacency_matrix[idx_B, idx_A] += flow_count
    
    # Generate the list of nodes in the same order as the adjacency matrix
    nodes = [(i, j) for i in range(dims[0]) for j in range(dims[1])]
    
    return adjacency_matrix, nodes