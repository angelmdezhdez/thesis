import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# ==============================
# Utilidades y carga de datos
# ==============================

def load_data(flow_file, laplacian_file, device):
    F = np.load(flow_file)  
    L = np.load(laplacian_file)  
    F = torch.tensor(F, dtype=torch.float32, device=device)
    L = torch.tensor(L, dtype=torch.float32, device=device)
    return F, L

# ==============================
# Función de optimización alpha
# ==============================

def optimize_alpha(F, D, lambda_reg, n_iter=100, lr=1e-2):
    T, n, _ = F.shape
    k = D.shape[1]
    alpha = torch.randn(T, k, n, device=F.device, requires_grad=True)
    optimizer = torch.optim.Adam([alpha], lr=lr)

    for _ in range(n_iter):
        optimizer.zero_grad()
        recon = torch.einsum('ik,kjn->ijn', D.T, alpha)  # D @ alpha
        loss_rec = ((F - recon)**2).sum()
        loss_sparse = lambda_reg * alpha.abs().sum()
        loss = loss_rec + loss_sparse
        loss.backward()
        optimizer.step()

    return alpha.detach()

# ==============================
# Función de optimización D
# ==============================

def optimize_dictionary(F, alpha, L, gamma_reg, n_iter=100, lr=1e-2):
    T, n, _ = F.shape
    k = alpha.shape[1]
    D = torch.randn(n, k, device=F.device, requires_grad=True)
    optimizer = torch.optim.Adam([D], lr=lr)

    for _ in range(n_iter):
        optimizer.zero_grad()
        recon = torch.einsum('ik,tkn->tin', D, alpha)  # D @ alpha
        loss_rec = ((F - recon)**2).sum()
        loss_smooth = gamma_reg * torch.trace(D.T @ L @ D)
        loss = loss_rec + loss_smooth
        loss.backward()
        optimizer.step()

    return D.detach()

# ==============================
# Entrenamiento principal
# ==============================

def train_dictionary_learning(flow_file, laplacian_file, k=10, n_epochs=10, lambda_reg=0.01, gamma_reg=0.1, alpha_steps=100, d_steps=100, lr=1e-2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    F, L = load_data(flow_file, laplacian_file, device)
    T, n, _ = F.shape

    # Inicialización aleatoria del diccionario
    D = torch.randn(n, k, device=device)

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        alpha = optimize_alpha(F, D, lambda_reg, n_iter=alpha_steps, lr=lr)
        D = optimize_dictionary(F, alpha, L, gamma_reg, n_iter=d_steps, lr=lr)

    return D, alpha

# ==============================
# Ejecución principal
# ==============================

if __name__ == '__main__':
    flow_path = 'flows.npy'  # Reemplace con su ruta
    lap_path = 'laplacian.npy'  # Reemplace con su ruta
    D, alpha = train_dictionary_learning(flow_path, lap_path)

    # Guardar resultados si se desea
    torch.save(D.cpu(), 'dictionary.pt')
    torch.save(alpha.cpu(), 'weights.pt')



import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ==============================
# Dataset personalizado
# ==============================

class FlowDataset(Dataset):
    def __init__(self, F_tensor):
        self.F = F_tensor

    def __len__(self):
        return self.F.shape[0]

    def __getitem__(self, idx):
        return self.F[idx], idx

# ==============================
# Utilidades y carga de datos
# ==============================

def load_data(flow_file, laplacian_file, device):
    F = np.load(flow_file)  # (T, n, n)
    L = np.load(laplacian_file)  # (n, n)
    F = torch.tensor(F, dtype=torch.float32, device=device)
    L = torch.tensor(L, dtype=torch.float32, device=device)
    return F, L

# ==============================
# Optimiza alpha por batches
# ==============================

def optimize_alpha_batches(D, flow_loader, T, k, lambda_reg, device, n_iter=100, lr=1e-2):
    n = D.shape[0]
    alpha_global = torch.zeros(T, k, n, device=device)  # No requiere grad, se actualiza manualmente

    for _ in range(n_iter):
        for batch, idx in flow_loader:
            batch = batch.to(device)
            batch_size = batch.shape[0]

            # Crear batch_alpha con gradientes
            batch_alpha = torch.randn(batch_size, k, n, device=device, requires_grad=True)
            optimizer = torch.optim.Adam([batch_alpha], lr=lr)

            for _ in range(1):  # Un paso de optimización por batch
                optimizer.zero_grad()
                recon = torch.matmul(D, batch_alpha)  # D: (n, k), batch_alpha: (B, k, n) → recon: (B, n, n)
                loss_rec = torch.norm(batch - recon, p='fro')**2
                loss_sparse = lambda_reg * batch_alpha.abs().sum()
                loss = loss_rec + loss_sparse
                loss.backward()
                optimizer.step()

            # Actualizar alpha_global solo en los índices del batch
            alpha_global[idx] = batch_alpha.detach()

    return alpha_global

# ==============================
# Optimiza diccionario con acumulación de gradientes
# ==============================

def optimize_dictionary_batches(F, alpha, L, flow_loader, gamma_reg, n_iter=100, lr=1e-2):
    T, n, _ = F.shape
    k = alpha.shape[1]
    device = F.device

    # Diccionario inicializado aleatoriamente
    D = torch.randn(n, k, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([D], lr=lr)

    for _ in range(n_iter):
        optimizer.zero_grad()
        total_loss_rec = 0.0

        for batch, idx in flow_loader:
            batch = batch.to(device)
            batch_alpha = alpha[idx]  # (B, k, n)

            # Reconstrucción: D (n, k) @ alpha (B, k, n) → (B, n, n)
            recon = torch.matmul(D, batch_alpha)  # (B, n, n)
            
            # Frobenius norm squared
            loss_rec = torch.norm(batch - recon, p='fro')**2
            loss_rec.backward()  # Acumulación de gradientes
            total_loss_rec += loss_rec.item()

        # Regularización de suavidad espacial (fuera del batch)
        loss_smooth = gamma_reg * torch.trace(D.T @ L @ D)
        loss_smooth.backward()  # Se acumula sobre los gradientes ya existentes

        optimizer.step()

        # (Opcional) Puedes imprimir o almacenar la pérdida total si lo deseas
        # print(f"Epoch {epoch}: Loss_rec={total_loss_rec:.2f}, Loss_smooth={loss_smooth.item():.2f}")

    return D.detach()

# ==============================
# Entrenamiento principal
# ==============================

def train_dictionary_learning(flow_file, laplacian_file, k=10, n_epochs=10, lambda_reg=0.01, gamma_reg=0.1, alpha_steps=100, d_steps=100, lr=1e-2, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    F, L = load_data(flow_file, laplacian_file, device)
    T, n, _ = F.shape

    dataset = FlowDataset(F)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    D = torch.randn(n, k, device=device)

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        alpha = optimize_alpha_batches(D, loader, T, k, lambda_reg, device, n_iter=alpha_steps, lr=lr)
        D = optimize_dictionary_batches(F, alpha, L, loader, gamma_reg, n_iter=d_steps, lr=lr)

    return D, alpha

# ==============================
# Ejecución principal
# ==============================

if __name__ == '__main__':
    flow_path = 'flows.npy'  # Reemplaza con tu ruta
    lap_path = 'laplacian.npy'  # Reemplaza con tu ruta
    D, alpha = train_dictionary_learning(flow_path, lap_path)

    torch.save(D.cpu(), 'dictionary.pt')
    torch.save(alpha.cpu(), 'weights.pt')
