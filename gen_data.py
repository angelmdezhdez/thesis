import numpy as np
from pathlib import Path

adj_nodes = np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                      [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                      [0, 0, 0, 0, 0, 0, 1, 1, 1, 0]])

#adj_nodes = np.array([[0, 1, 0, 1],
#                      [1, 0, 0, 1],
#                      [0, 0, 0, 1],
#                      [1, 1, 1, 0]])

D = np.diag(np.sum(adj_nodes, axis=1))

L = np.eye(adj_nodes.shape[0]) - (np.linalg.inv(D))**0.5 @ adj_nodes @ np.linalg.inv(D)**0.5

Dict = np.array([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.3, 0.0, 0.0, 0.1, 1.0, 0.5, 0.0, 0.0, 0.0],
                 [0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.4, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0],
                 [0.2, 0.0, 0.3, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.1, 0.2, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0],
                 [0.1, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 1.0],
                 [0.0, 0.0, 0.3, 0.0, 0.2, 0.0, 0.0, 0.5, 0.0, 0.0],
                 [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
                 ])


#f1 = np.array([[5, 0, 0, 0, 0, 0, 5, 0, 0, 0],
#               [0, 6, 0, 6, 0, 0, 0, 0, 0, 0],
#               [0, 8, 7, 6, 0, 0, 9, 0, 0, 0],
#               [0, 0, 6, 0, 6, 0, 0, 0, 0, 0],
#               [0, 0, 5, 5, 6, 0, 0, 0, 0, 0],
#               [5, 0, 0, 0, 0, 5, 0, 0, 0, 0],
#               [0, 0, 9, 0, 0, 0, 0, 0, 0, 10],
#               [0, 0, 0, 0, 0, 0, 0, 5, 6, 0],
#               [0, 0, 0, 0, 0, 0, 5, 6, 5, 0],
#               [0, 0, 0, 0, 0, 0, 9, 6, 0, 5]])
#
#f2 = np.array([[0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
#               [6, 6, 9, 0, 0, 6, 0, 0, 0, 0],
#               [0, 8, 8, 6, 7, 0, 9, 0, 0, 0],
#               [0, 0, 6, 0, 6, 0, 0, 0, 0, 0],
#               [0, 0, 5, 5, 6, 0, 0, 0, 0, 0],
#               [6, 6, 0, 0, 0, 5, 9, 0, 0, 0],
#               [0, 0, 9, 0, 0, 8, 9, 0, 0, 9],
#               [0, 0, 0, 0, 0, 0, 0, 0, 6, 7],
#               [0, 0, 0, 0, 0, 0, 0, 6, 5, 6],
#               [0, 0, 0, 0, 0, 0, 9, 6, 5, 5]])
#               
#f3 = np.array([[5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 6, 6, 0, 0, 0, 0, 0, 0, 0],
#               [0, 5, 5, 0, 7, 0, 9, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 6, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 5, 6, 0, 0, 0],
#               [0, 0, 9, 0, 0, 7, 8, 0, 5, 6],
#               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 0, 0, 0, 0, 0, 6, 0, 5, 0],
#               [0, 0, 0, 0, 0, 0, 5, 0, 0, 0]])
#
#f4 = np.array([[5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#               [0, 6, 8, 0, 0, 0, 9, 0, 0, 0],
#               [0, 9, 5, 5, 7, 8, 9, 0, 0, 0],
#               [0, 0, 7, 5, 6, 0, 0, 0, 0, 0],
#               [0, 0, 6, 7, 0, 0, 0, 0, 0, 0],
#               [0, 0, 8, 0, 0, 5, 8, 0, 0, 0],
#               [0, 8, 8, 0, 0, 9, 8, 0, 6, 6],
#               [0, 0, 0, 0, 0, 0, 0, 6, 0, 0],
#               [0, 0, 0, 0, 0, 0, 5, 0, 5, 6],
#               [0, 0, 0, 0, 0, 0, 5, 0, 7, 5]])

#f1 = np.array([[9, 0, 0, 0],
#               [0, 10, 0, 0],
#               [0, 0, 0, 8],
#               [0, 0, 8, 0]])
#
#f2 = np.array([[8, 0, 0, 0],
#               [0, 0, 0, 9],
#               [0, 0, 7, 0],
#               [0, 9, 0, 0]])
#
#f3 = np.array([[0, 0, 0, 10],
#               [0, 8, 0, 0],
#               [0, 0, 7, 0],
#               [9, 0, 0, 0]])

n = adj_nodes.shape[0]
T = 30 # no de flujos por flujo base
n_base = 4 # no de flujos base
noise_level = 0.01 # nivel de ruido

Path("synthetic_data2").mkdir(exist_ok=True)

np.random.seed(0)
# Generar flujos base
base_flows = []

for i in range(n_base):
    flow = np.zeros((n, n))
    cols = np.random.choice(Dict.shape[1], n, replace=True)
    for j, jj in enumerate(cols):
        for k in range(n):
            flow[k, j] = Dict[k, jj]

    base_flows.append(flow)

flows = []
labels = []
for i, f in enumerate(base_flows):
    for _ in range(T):
        noise = np.random.rand(n, n) * noise_level
        noise = noise * (np.random.rand(n, n) < noise_level/10)
        noisy_flow = f + noise
        col_sums = noisy_flow.sum(axis=0, keepdims=True)
        col_sums[col_sums == 0] = 1
        noisy_flow = noisy_flow / col_sums
        flows.append(noisy_flow)
        labels.append(i)


flows = np.array(flows)
labels = np.array(labels)

indexes = np.random.permutation(len(flows))

flows = flows[indexes]
labels = labels[indexes]

np.save("synthetic_data2/flows.npy", flows)

np.save("synthetic_data2/laplacian.npy", L)

np.save("synthetic_data2/labels.npy", labels)

np.save("synthetic_data2/dictionary.npy", Dict)  

for i, f in enumerate(base_flows):
    np.save(f'synthetic_data2/f{i+1}.npy', f)

print("Flujos sintéticos generados y guardados en 'synthetic_data2/flows.npy'")
print("Laplaciano guardado en 'synthetic_data2/laplacian.npy'")