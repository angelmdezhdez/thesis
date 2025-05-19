import numpy as np
from pathlib import Path

n = 10 #nodos
T = 25 # no de flujos por flujo base

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

L = np.diag(adj_nodes.sum(axis=1)) - adj_nodes 


f1 = np.array([[5, 0, 0, 0, 0, 0, 5, 0, 0, 0],
               [0, 6, 0, 6, 0, 0, 0, 0, 0, 0],
               [0, 8, 7, 6, 0, 0, 9, 0, 0, 0],
               [0, 0, 6, 0, 6, 0, 0, 0, 0, 0],
               [0, 0, 5, 5, 6, 0, 0, 0, 0, 0],
               [5, 0, 0, 0, 0, 5, 0, 0, 0, 0],
               [0, 0, 9, 0, 0, 0, 0, 0, 0, 10],
               [0, 0, 0, 0, 0, 0, 0, 5, 6, 0],
               [0, 0, 0, 0, 0, 0, 5, 6, 5, 0],
               [0, 0, 0, 0, 0, 0, 9, 6, 0, 5]])

f2 = np.array([[0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
               [6, 6, 9, 0, 0, 6, 0, 0, 0, 0],
               [0, 8, 8, 6, 7, 0, 9, 0, 0, 0],
               [0, 0, 6, 0, 6, 0, 0, 0, 0, 0],
               [0, 0, 5, 5, 6, 0, 0, 0, 0, 0],
               [6, 6, 0, 0, 0, 5, 9, 0, 0, 0],
               [0, 0, 9, 0, 0, 8, 9, 0, 0, 9],
               [0, 0, 0, 0, 0, 0, 0, 0, 6, 7],
               [0, 0, 0, 0, 0, 0, 0, 6, 5, 6],
               [0, 0, 0, 0, 0, 0, 9, 6, 5, 5]])
               
f3 = np.array([[5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 6, 6, 0, 0, 0, 0, 0, 0, 0],
               [0, 5, 5, 0, 7, 0, 9, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 6, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 5, 6, 0, 0, 0],
               [0, 0, 9, 0, 0, 7, 8, 0, 5, 6],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 6, 0, 5, 0],
               [0, 0, 0, 0, 0, 0, 5, 0, 0, 0]])

f4 = np.array([[5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 6, 8, 0, 0, 0, 9, 0, 0, 0],
               [0, 9, 5, 5, 7, 8, 9, 0, 0, 0],
               [0, 0, 7, 5, 6, 0, 0, 0, 0, 0],
               [0, 0, 6, 7, 0, 0, 0, 0, 0, 0],
               [0, 0, 8, 0, 0, 5, 8, 0, 0, 0],
               [0, 8, 8, 0, 0, 9, 8, 0, 6, 6],
               [0, 0, 0, 0, 0, 0, 0, 6, 0, 0],
               [0, 0, 0, 0, 0, 0, 5, 0, 5, 6],
               [0, 0, 0, 0, 0, 0, 5, 0, 7, 5]])

Path("synthetic_data").mkdir(exist_ok=True)

np.random.seed(0)
base_flows = [f1, f2, f3, f4]

flows = []
labels = []
for i, f in enumerate(base_flows):
    for _ in range(T):
        noise = np.random.randint(0, 3, size=(n, n))
        noisy_flow = f + noise
        noisy_flow = noisy_flow / noisy_flow.sum()
        flows.append(noisy_flow)
        labels.append(i)


flows = np.array(flows)
labels = np.array(labels)

indexes = np.random.permutation(len(flows))

flows = flows[indexes]
labels = labels[indexes]

np.save("synthetic_data/flows.npy", flows)

np.save("synthetic_data/laplacian.npy", L)

print("Flujos sintéticos generados y guardados en 'synthetic_data/flows.npy'")
print("Laplaciano guardado en 'synthetic_data/laplacian.npy'")