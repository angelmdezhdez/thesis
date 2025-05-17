import numpy as np
import networkx as nx
from pathlib import Path

# Parámetros
n = 10      # número de nodos
k = 3       # número de elementos base del diccionario
T = 50      # número total de matrices de flujo
base_count = 5  # número de matrices base

# Crear directorio
Path("synthetic_data").mkdir(exist_ok=True)

# Matrices base normalizadas (simulan flujos de probabilidad)
np.random.seed(0)
base_flows = [np.random.rand(n, n) for _ in range(base_count)]
base_flows = [bf / bf.sum() for bf in base_flows]

# Generar flujos sintéticos
flows = []
for _ in range(T):
    idx = np.random.choice(base_count)
    noise = np.random.normal(scale=0.01, size=(n, n))
    noisy_flow = base_flows[idx] + noise
    noisy_flow[noisy_flow < 0] = 0
    noisy_flow = noisy_flow / noisy_flow.sum()
    flows.append(noisy_flow)

flows = np.stack(flows)  # (T, n, n)
np.save("synthetic_data/flows.npy", flows)

# Grafo de adyacencia y Laplaciana
G = nx.erdos_renyi_graph(n, p=0.4, seed=42)
A = nx.to_numpy_array(G)
L = np.diag(A.sum(axis=1)) - A  # Laplaciana no normalizada
np.save("synthetic_data/laplacian.npy", L)
