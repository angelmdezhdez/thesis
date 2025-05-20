import subprocess
from itertools import product

# Parámetros a variar
natoms_list = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
lambda_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
gamma_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
smooth_list = [0, 1]

# Parámetros fijos
base_command = [
    "python3", "dict_arr_learning.py",
    "-system", "experiment",
    "-flows", "flows.npy",
    "-lap", "laplacian.npy",
    "-ep", "10",
    "-reg", "l2",
    "-as", "100",
    "-ds", "100",
    "-lr", "1e-2",
    "-bs", "32"
]

# Ejecutar combinaciones
for natoms, lam, gamma, smooth in product(natoms_list, lambda_list, gamma_list, smooth_list):
    cmd = base_command + [
        '-system', f'_{natoms}_{lam}_{gamma}_{smooth}',
        "-natoms", str(natoms),
        "-lambda", str(lam),
        "-gamma", str(gamma)
    ]
    print("Ejecutando:", " ".join(cmd))
    subprocess.run(cmd)
