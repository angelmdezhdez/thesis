import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import sys
sys.stdout.flush()

# ==============================
# Personalized Dataset
# ==============================

class FlowDataset(Dataset):
    def __init__(self, F_tensor):
        self.F = F_tensor

    def __len__(self):
        return self.F.shape[0]

    def __getitem__(self, idx):
        return self.F[idx], idx

# ==============================
# Utilities
# ==============================

def load_data(flow_file, laplacian_file, device):
    F = np.load(flow_file)
    L = np.load(laplacian_file)
    F = torch.tensor(F, dtype=torch.float32, device=device)
    L = torch.tensor(L, dtype=torch.float32, device=device)
    return F, L

# ==============================
# alpha optimization (with softmax)
# ==============================

def optimize_alpha_batches(D_logits, flow_loader, T, k, lambda_reg, device, n_iter=100, lr=1e-2):
    n = D_logits.shape[0]
    alpha_logits_global = torch.randn(T, k, n, device=device)

    for _ in range(n_iter):
        for batch, idx in flow_loader:
            batch = batch.to(device)
            B = batch.shape[0]

            alpha_logits = torch.rand(B, k, n, device=device, requires_grad=True)
            optimizer = torch.optim.Adam([alpha_logits], lr=lr)

            optimizer.zero_grad()
            D = torch.softmax(D_logits, dim=0)  # n x k
            alpha = torch.softmax(alpha_logits, dim=2)  # B x k x n

            recon = torch.matmul(D, alpha)  # B x n x n
            loss_rec = ((batch - recon) ** 2).sum()
            loss_sparse = lambda_reg * alpha.sum()
            loss = loss_rec / (2*T) + loss_sparse / T

            loss.backward()
            optimizer.step()

            alpha_logits_global[idx] = alpha_logits.detach()

    return alpha_logits_global

# ==============================
# dictionary optimization with cumulative gradients (with softmax)
# ==============================

def optimize_dictionary_batches(F, alpha_logits, L, flow_loader, gamma_reg, n_iter=100, lr=1e-2, smooth=True):
    T, n, _ = F.shape
    k = alpha_logits.shape[1]
    device = F.device

    D_logits = torch.rand(n, k, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([D_logits], lr=lr)

    for _ in range(n_iter):
        optimizer.zero_grad()
        total_loss_tensor = 0.0
        total_loss_rec = 0.0

        for batch, idx in flow_loader:
            batch = batch.to(device)
            batch_alpha_logits = alpha_logits[idx]  # B x k x n

            D = torch.softmax(D_logits, dim=0)  # n x k
            alpha = torch.softmax(batch_alpha_logits, dim=2)  # B x k x n
            recon = torch.matmul(D, alpha)  # B x n x n

            loss_rec = ((batch - recon) ** 2).sum()
            total_loss_tensor += loss_rec
            total_loss_rec += loss_rec.item()

        total_loss_rec /= (2 * T)

        if smooth:
            D = torch.softmax(D_logits, dim=0)
            loss_smooth = gamma_reg * torch.trace(D.T @ L @ D)
            total_loss_tensor += loss_smooth
        else:
            loss_smooth = torch.tensor(0.0, device=device)

        total_loss_tensor.backward()
        optimizer.step()

    return D_logits.detach(), total_loss_rec + loss_smooth.item()

# ==============================
# train 
# ==============================

def train_dictionary_learning(flow_file, laplacian_file, k=10, n_epochs=10, lambda_reg=0.01, gamma_reg=0.1, alpha_steps=100, d_steps=100, lr=1e-2, batch_size=32, smooth=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    sys.stdout.flush()

    F, L = load_data(flow_file, laplacian_file, device)
    T, n, _ = F.shape

    dataset = FlowDataset(F)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    torch.manual_seed(0)
    D_logits = torch.rand(n, k, device=device)

    best_loss = float('inf')
    patience = 15
    loss_vector = []

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        sys.stdout.flush()

        alpha_logits = optimize_alpha_batches(D_logits, loader, T, k, lambda_reg, device, n_iter=alpha_steps, lr=lr)
        D_logits, loss = optimize_dictionary_batches(F, alpha_logits, L, loader, gamma_reg, n_iter=d_steps, lr=lr, smooth=smooth)

        print(f"Loss: {loss:.4f}")
        sys.stdout.flush()

        loss_vector.append(loss)
        if loss < best_loss:
            best_D_logits = D_logits
            best_alpha_logits = alpha_logits
            print(f'Best D and alpha updated in epoch {epoch+1}')
            sys.stdout.flush()
            best_loss = loss
            patience = 10
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping")
                sys.stdout.flush()
                break

    print("Training finished")
    sys.stdout.flush()

    best_D = torch.softmax(best_D_logits, dim=0)
    best_alpha = torch.softmax(best_alpha_logits, dim=2)
    return best_D, best_alpha, loss_vector

# ==============================
# main
# ==============================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dictionary Learning for Arrival Flows')
    parser.add_argument('-system', '--system_key', type=str, default='experiment', required=True)
    parser.add_argument('-flows', '--flows', type=str, required=True)
    parser.add_argument('-lap', '--laplacian', type=str, required=True)
    parser.add_argument('-natoms', '--number_atoms', type=int, required=True)
    parser.add_argument('-ep', '--epochs', type=int, required=True)
    parser.add_argument('-lambda', '--lambda_reg', type=float, required=True)
    parser.add_argument('-smooth', '--smooth', type=int, default=0, required=True)
    parser.add_argument('-gamma', '--gamma_reg', type=float, required=True)
    parser.add_argument('-as', '--alpha_steps', type=int, required=True)
    parser.add_argument('-ds', '--dict_steps', type=int, required=True)
    parser.add_argument('-lr', '--learning_rate', type=float, required=True)
    parser.add_argument('-bs', '--batch_size', type=int, required=True)
    args = parser.parse_args()

    system = args.system_key
    flow_path = args.flows
    lap_path = args.laplacian
    k = args.number_atoms
    n_epochs = args.epochs
    lambda_reg = args.lambda_reg
    gamma_reg = args.gamma_reg
    alpha_steps = args.alpha_steps
    d_steps = args.dict_steps
    lr = args.learning_rate
    batch_size = args.batch_size
    smooth = bool(args.smooth)

    print(f"System: {system}")
    print(f"Flows: {flow_path}")
    print(f"Laplacian: {lap_path}")
    print(f"Number of atoms: {k}")
    print(f"Epochs: {n_epochs}")
    print(f"Lambda: {lambda_reg}")
    print(f"Smoothness: {smooth}")
    print(f"Gamma: {gamma_reg}")
    print(f"Alpha steps: {alpha_steps}")
    print(f"Dictionary steps: {d_steps}")
    print(f"Learning rate: {lr}")
    print(f"Batch size: {batch_size}\n\n")
    print('Starting training')

    D, alpha, loss = train_dictionary_learning(flow_path, lap_path, 
                                               k=k, 
                                               n_epochs=n_epochs, 
                                               lambda_reg=lambda_reg, 
                                               gamma_reg=gamma_reg, 
                                               alpha_steps=alpha_steps, 
                                               d_steps=d_steps, 
                                               lr=lr, 
                                               batch_size=batch_size, 
                                               smooth=smooth)

    save_dir = f'results_{system}'
    os.makedirs(save_dir, exist_ok=True)

    D_s = D.cpu().numpy().shape
    with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
        f.write(f"system: {system}\n")
        f.write(f'number of nodes: {D_s[0]}\n')
        f.write(f"flows: {flow_path}\n")
        f.write(f"laplacian: {lap_path}\n")
        f.write(f"number of atoms: {k}\n")
        f.write(f"epochs: {n_epochs}\n")
        f.write(f"lambda: {lambda_reg}\n")
        f.write(f"smoothness: {smooth}\n")
        f.write(f"gamma: {gamma_reg}\n")
        f.write(f"alpha steps: {alpha_steps}\n")
        f.write(f"dictionary steps: {d_steps}\n")
        f.write(f"learning rate: {lr}\n")
        f.write(f"batch size: {batch_size}\n")
        f.write(f"final loss: {loss[-1]}\n")

    np.save(os.path.join(save_dir, 'dictionary.npy'), D.cpu().numpy())
    np.save(os.path.join(save_dir, 'weights.npy'), alpha.cpu().numpy())
    np.save(os.path.join(save_dir, 'loss.npy'), np.array(loss))

    print('Finished')
    sys.stdout.flush()

