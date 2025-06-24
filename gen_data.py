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

#adj_nodes = np.array([[0, 1, 0, 1, 0, 0],
#                      [1, 0, 1, 0, 1, 0],
#                      [0, 1, 0, 0, 0, 1],
#                      [1, 0, 0, 0, 1, 0],
#                      [0, 1, 0, 1, 0, 1],
#                      [0, 0, 1, 0, 1, 0]])

D = np.diag(np.sum(adj_nodes, axis=1))

L = np.eye(adj_nodes.shape[0]) - (np.linalg.inv(D))**0.5 @ adj_nodes @ np.linalg.inv(D)**0.5

#Dict = np.array([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                 [0.0, 0.3, 0.0, 0.0, 0.1, 1.0, 0.5, 0.0, 0.0, 0.0],
#                 [0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                 [0.0, 0.4, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0],
#                 [0.2, 0.0, 0.3, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
#                 [0.0, 0.0, 0.1, 0.2, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0],
#                 [0.1, 0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 1.0],
#                 [0.0, 0.0, 0.3, 0.0, 0.2, 0.0, 0.0, 0.5, 0.0, 0.0],
#                 [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
#                 [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
#                 ])

#Dict = np.array([[0, 9, 6, 10, 8, 0],
#                 [0, 8, 9, 5, 6, 7],
#                 [0, 0, 5, 6, 0, 10],
#                 [10, 0, 0, 10, 5, 5]])

#Dict = np.array([[10, 0, 0, 5, 0, 0],
#                 [9, 8, 9, 6, 0, 0],
#                 [8, 5, 10, 0, 0, 0],
#                 [7, 0, 9, 5, 10, 8],
#                 [0, 2, 0, 0, 10, 7],
#                 [0, 0, 9, 0, 5, 10]])

Dict = np.array([[10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 6, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 8, 7, 0, 0, 0, 9, 0, 0, 0],
                 [0, 0, 10, 0, 0, 5, 0, 0, 0, 0],
                 [0, 0, 8, 8, 0, 8, 0, 0, 0, 0],
                 [5, 6, 0, 0, 0, 10, 9, 0, 0, 0],
                 [0, 0, 0, 9, 10, 8, 9, 10, 5, 6],
                 [0, 5, 0, 0, 5, 5, 0, 6 , 5,7],
                 [5 ,0 ,0 ,0 ,7 ,0 ,0 ,0 ,5 ,10],
                 [9 ,0 ,0 ,0 ,7 ,0 ,0 ,0 ,10 ,0]])


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
T = 100 # no de flujos por flujo base
n_base = 10 # no de flujos base
noise_level = 0.4 # probabilidad de ruido en cada celda
max_value_noise = 4 # valor máximo del ruido

dir = 'exp_train_dict/synthetic_data_v5'

Path(f"{dir}").mkdir(exist_ok=True)

np.random.seed(0)
# Generar flujos base
base_flows = []

for i in range(n_base):
    flow = np.zeros((n, n))
    cols = np.random.choice(Dict.shape[1], n, replace=False)
    for j, jj in enumerate(cols):
        for k in range(n):
            flow[k, j] = Dict[k, jj]

    base_flows.append(flow)

flows = []
labels = []
for i, f in enumerate(base_flows):
    for _ in range(T):
        noise = np.random.randint(0, max_value_noise + 1, (n, n)).astype(float)
        # Aplicar ruido con probabilidad noise_level
        noise = noise * (np.random.rand(n, n) < noise_level)
        noisy_flow = f + noise
        #col_sums = noisy_flow.sum(axis=0, keepdims=True)
        #col_sums[col_sums == 0] = 1
        noisy_flow = noisy_flow / noisy_flow.sum()
        flows.append(noisy_flow)
        labels.append(i)


flows = np.array(flows)
labels = np.array(labels)

indexes = np.random.permutation(len(flows))

flows = flows[indexes]
labels = labels[indexes]

np.save(f"{dir}/flows.npy", flows)

np.save(f"{dir}/laplacian.npy", L)

np.save(f"{dir}/labels.npy", labels)

Dict = Dict / Dict.sum()

np.save(f"{dir}/dictionary.npy", Dict)  

for i, f in enumerate(base_flows):
    np.save(f'{dir}/f{i+1}.npy', f)

print(f"Flujos sintéticos generados y guardados en '{dir}/flows.npy'")
print(f"Laplaciano guardado en '{dir}/laplacian.npy'")