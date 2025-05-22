import numpy as np
from pathlib import Path

#adj_nodes = np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
#                      [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#                      [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
#                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                      [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
#                      [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#                      [0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
#                      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
#                      [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
#                      [0, 0, 0, 0, 0, 0, 1, 1, 1, 0]])

adj_nodes = np.array([[0, 1, 0, 1],
                      [1, 0, 0, 1],
                      [0, 0, 0, 1],
                      [1, 1, 1, 0]])

D = np.diag(np.sum(adj_nodes, axis=1))

L = np.eye(adj_nodes.shape[0]) - (np.linalg.inv(D))**0.5 @ adj_nodes @ np.linalg.inv(D)**0.5


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

f1 = np.array([[9, 0, 0, 0],
               [0, 10, 0, 0],
               [0, 0, 0, 8],
               [0, 0, 8, 0]])

f2 = np.array([[8, 0, 0, 0],
               [0, 0, 0, 9],
               [0, 0, 7, 0],
               [0, 9, 0, 0]])

f3 = np.array([[0, 0, 0, 10],
               [0, 8, 0, 0],
               [0, 0, 7, 0],
               [9, 0, 0, 0]])

n = adj_nodes.shape[0]
T = 10 # no de flujos por flujo base

Path("synthetic_data").mkdir(exist_ok=True)

np.random.seed(0)
# Generar flujos base
base_flows = [f1, f2, f3]

flows = []
labels = []
for i, f in enumerate(base_flows):
    for _ in range(T):
        noise = np.random.randint(0, 2, size=(n, n))
        noise = noise * (np.random.rand(n, n) < 0.2)
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

np.save("synthetic_data/flows.npy", flows)

np.save("synthetic_data/laplacian.npy", L)

np.save("synthetic_data/labels.npy", labels)

for i, f in enumerate(base_flows):
    col_sums = f.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    f = f / col_sums
    np.save(f'synthetic_data/f{i+1}.npy', f)

print("Flujos sintéticos generados y guardados en 'synthetic_data/flows.npy'")
print("Laplaciano guardado en 'synthetic_data/laplacian.npy'")