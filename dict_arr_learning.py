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
    '''This class creates a dataset from the flow tensor.'''
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
    '''This function loads the flow and laplacian data from files.'''
    F = np.load(flow_file)  # (T, n, n)
    L = np.load(laplacian_file)  # (n, n)
    F = torch.tensor(F, dtype=torch.float32, device=device)
    L = torch.tensor(L, dtype=torch.float32, device=device)
    return F, L

# ==============================
# alpha optimization
# ==============================

def optimize_alpha_batches(D, flow_loader, T, k, lambda_reg, device, n_iter=100, lr=1e-2, regularization = 'l2'):
    n = D.shape[0]
    # since we are using batch optimization, this global alpha will be updated by parts
    # so we dont need gradients for it
    alpha_global = torch.zeros(T, k, n, device=device)  

    for _ in range(n_iter):
        for batch, idx in flow_loader:
            batch = batch.to(device)
            batch_size = batch.shape[0]

            # batch_alpha is the alpha for the current batch (this one requires gradients)
            batch_alpha = torch.randn(batch_size, k, n, device=device, requires_grad=True)
            optimizer = torch.optim.Adam([batch_alpha], lr=lr)

            # we give an only step to the optimizer
            optimizer.zero_grad()
            # D: dictionary (n, k), n is the number of nodes, k is the number of elements in the dictionary
            # batch_alpha: (B, k, n), where B is the batch size
            # recon: (B, n, n), batch: (B, n, n)
            recon = torch.matmul(D, batch_alpha) 
            # sum over the batch of frobenius norm squared 
            loss_rec = ((batch - recon) ** 2).sum()
            # regularization
            if regularization == 'l2':
                loss_sparse = lambda_reg * torch.norm(batch_alpha, p='fro')**2
            elif regularization == 'l1':
                loss_sparse = lambda_reg * batch_alpha.abs().sum()
            else:
                loss_sparse = torch.tensor(0.0, device=device)
            # total loss
            loss = loss_rec + loss_sparse

            # optimization step
            loss.backward()
            optimizer.step()

            # update the global alpha with the current batch_alpha
            alpha_global[idx] = batch_alpha.detach()

    return alpha_global

# ==============================
# dictionary optimization with cumulative gradients
# ==============================

def optimize_dictionary_batches(F, alpha, L, flow_loader, gamma_reg, n_iter=100, lr=1e-2, smooth=True):
    T, n, _ = F.shape
    k = alpha.shape[1]
    device = F.device

    # D: dictionary, it requires gradients
    D = torch.randn(n, k, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([D], lr=lr)

    for _ in range(n_iter):
        optimizer.zero_grad()
        total_loss_rec = 0.0

        # acumulate gradients since we are using batch optimization
        # we need to accumulate the loss for each batch
        total_loss_tensor = 0.0#torch.tensor(0.0, device=device)
        total_loss_rec = 0.0#torch.tensor(0.0, device=device)

        for batch, idx in flow_loader:
            batch = batch.to(device)
            # select the alpha for the current batch
            batch_alpha = alpha[idx]
            
            # reconstruction
            recon = torch.matmul(D, batch_alpha)  # (B, n, n)
            # loss reconstruction
            loss_rec = ((batch - recon) ** 2).sum()
            total_loss_tensor += loss_rec
            total_loss_rec += loss_rec.item()
        
        # regularization over the spatial smoothness
        if smooth:
            loss_smooth = gamma_reg * torch.trace(D.T @ L @ D)
            total_loss_tensor += loss_smooth
        else:
            loss_smooth = torch.tensor(0.0, device=device)

        total_loss_tensor.backward()
        optimizer.step()

    return D.detach(), total_loss_rec + loss_smooth.item()

# ==============================
# train 
# ==============================

def train_dictionary_learning(flow_file, laplacian_file, k=10, n_epochs=10, lambda_reg=0.01, gamma_reg=0.1, alpha_steps=100, d_steps=100, lr=1e-2, batch_size=32, regularization='l2', smooth=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    sys.stdout.flush()

    F, L = load_data(flow_file, laplacian_file, device)
    T, n, _ = F.shape

    dataset = FlowDataset(F)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    D = torch.randn(n, k, device=device)

    # early stop
    best_loss = float('inf')
    patience = 15

    loss_vector = []

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        sys.stdout.flush()

        alpha = optimize_alpha_batches(D, loader, T, k, lambda_reg, device, n_iter=alpha_steps, lr=lr, regularization=regularization)
        D, loss = optimize_dictionary_batches(F, alpha, L, loader, gamma_reg, n_iter=d_steps, lr=lr, smooth=smooth)

        if regularization == 'l2':
            loss += lambda_reg * torch.norm(alpha, p='fro')**2
        elif regularization == 'l1':
            loss += lambda_reg * alpha.abs().sum()
        else:
            loss += torch.tensor(0.0, device=device)

        print(f"Loss: {loss:.4f}")
        sys.stdout.flush()

        loss_vector.append(loss)
        if loss < best_loss:
            best_D = D
            best_alpha = alpha
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

    return best_D, best_alpha, loss_vector

# ==============================
# main
# ==============================

if __name__ == '__main__':
    # python3 dict_arr_learning.py -system experiment -flows flows.npy -lap laplacian.npy -natoms 10 -ep 10 -reg l2 -lambda 0.01 -smooth 1 -gamma 0.1 -as 100 -ds 100 -lr 1e-2 -bs 32

    parser = argparse.ArgumentParser(description='Dictionary Learning for Arrival Flows')
    parser.add_argument('-system', '--system_key', type=str, default='experiment', help='system of flows', required = True)
    parser.add_argument('-flows', '--flows', type=str, default='flows.npy', help='Path to the flow tensor file', required = True)
    parser.add_argument('-lap', '--laplacian', type=str, default='laplacian.npy', help='Path to the laplacian file', required = True)
    parser.add_argument('-natoms', '--number_atoms', type=int, default=10, help='Number of dictionary elements', required = True)
    parser.add_argument('-ep', '--epochs', type=int, default=10, help='Number of epochs', required = True)
    parser.add_argument('-reg', '--regularization', type=str, default='l2', help='which regularization', required = True)
    parser.add_argument('-lambda', '--lambda_reg', type=float, default=0.01, help='regularization parameter', required = True)
    parser.add_argument('-smooth', '--smooth', type=int, default=0, help='smoothness 1(True)/0(False)', required = True)
    parser.add_argument('-gamma', '--gamma_reg', type=float, default=0.1, help='smoothness regularization parameter', required = True)
    parser.add_argument('-as', '--alpha_steps', type=int, default=100, help='number of steps for alpha optimization', required = True)
    parser.add_argument('-ds', '--dict_steps', type=int, default=100, help='number of steps for dictionary optimization', required = True)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2, help='learning rate', required = True)
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size', required = True)
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
    regularization = args.regularization
    smooth = bool(args.smooth)

    print(f"System: {system}")
    sys.stdout.flush()
    print(f"Flows: {flow_path}")
    sys.stdout.flush()
    print(f"Laplacian: {lap_path}")
    sys.stdout.flush()
    print(f"Number of atoms: {k}")
    sys.stdout.flush()
    print(f"Epochs: {n_epochs}")
    sys.stdout.flush()
    print(f"Regularization: {regularization}")
    sys.stdout.flush()
    print(f"Lambda: {lambda_reg}")
    sys.stdout.flush()
    print(f"Smoothness: {smooth}")
    sys.stdout.flush()
    print(f"Gamma: {gamma_reg}")
    sys.stdout.flush()
    print(f"Alpha steps: {alpha_steps}")
    sys.stdout.flush()
    print(f"Dictionary steps: {d_steps}")
    sys.stdout.flush()
    print(f"Learning rate: {lr}")
    sys.stdout.flush()
    print(f"Batch size: {batch_size}\n\n")
    sys.stdout.flush()

    print('Starting training')
    sys.stdout.flush()


    D, alpha, loss = train_dictionary_learning(flow_path, lap_path, 
                                               k=k, 
                                               n_epochs=n_epochs, 
                                               lambda_reg=lambda_reg, 
                                               gamma_reg=gamma_reg, 
                                               alpha_steps=alpha_steps, 
                                               d_steps=d_steps, 
                                               lr=lr, 
                                               batch_size=batch_size, 
                                               regularization=regularization, 
                                               smooth=smooth)

    # directory to save the results
    save_dir = f'results_{system}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # save parameters
    D_s = D.cpu().numpy().shape

    file_params = open(os.path.join(save_dir, 'params.txt'), 'w')
    file_params.write(f"system: {system}\n")
    file_params.write(f'number of nodes: {D_s[0]}\n')
    file_params.write(f"flows: {flow_path}\n")
    file_params.write(f"laplacian: {lap_path}\n")
    file_params.write(f"number of atoms: {k}\n")
    file_params.write(f"epochs: {n_epochs}\n")
    file_params.write(f"regularization: {regularization}\n")
    file_params.write(f"lambda: {lambda_reg}\n")
    file_params.write(f"smoothness: {smooth}\n")
    file_params.write(f"gamma: {gamma_reg}\n")
    file_params.write(f"alpha steps: {alpha_steps}\n")
    file_params.write(f"dictionary steps: {d_steps}\n")
    file_params.write(f"learning rate: {lr}\n")
    file_params.write(f"batch size: {batch_size}\n")
    file_params.write(f"final loss: {loss[-1]}\n")
    file_params.close()


    np.save(os.path.join(save_dir, 'dictionary.npy'), D.cpu().numpy())
    np.save(os.path.join(save_dir, 'weights.npy'), alpha.cpu().numpy())
    np.save(os.path.join(save_dir, 'loss.npy'), np.array(loss))

    print('Finished')
    sys.stdout.flush()