import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import time
import os
import sys

torch.set_num_threads(os.cpu_count())

sys.stdout.flush()

# ==============================
# Dataloader
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
# extras
# ==============================

def load_data(flow_file, dictionary_file, device):
    '''This function loads the flow and laplacian data from files.'''
    F = np.load(flow_file)  # (T, n, n)
    D = np.load(dictionary_file)  # (n, k)
    F = torch.tensor(F, dtype=torch.float32, device=device)
    D = torch.tensor(D, dtype=torch.float32, device=device)
    return F, D

# ==============================
# alpha optimization
# ==============================

def optimize_alpha_batches(D, alpha_params, flow_loader, T, k, lambda_reg, device, n_iter=100, lr=1e-2, regularization='l2'):
    best_alpha = torch.zeros_like(alpha_params, device=device)
    best_losses = torch.full((T,), float('inf'), device=device)

    alpha_params = alpha_params.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([alpha_params], lr=lr)

    for _ in range(n_iter):
        optimizer.zero_grad()
        total_loss = 0.0

        for batch, idx in flow_loader:
            batch = batch.to(device)
            batch_alpha = alpha_params[idx]

            recon = torch.matmul(D, batch_alpha)
            #recon = softmax_transform(recon)  # Ensure reconstruction is in [0,1]^d
            loss_rec = ((batch - recon) ** 2).sum()

            if regularization == 'l2':
                loss_reg = lambda_reg * torch.norm(batch_alpha, p='fro')**2
            elif regularization == 'l1':
                loss_reg = lambda_reg * batch_alpha.abs().sum()
            else:
                loss_reg = 0.0

            loss = loss_rec + loss_reg
            total_loss += loss

        total_loss.backward()
        optimizer.step()

        # Projection [0,1]^d
        with torch.no_grad():
            alpha_params.clamp_(0.0, 1.0)

    with torch.no_grad():
        for batch, idx in flow_loader:
            batch = batch.to(device)
            batch_alpha = alpha_params[idx]
            recon = torch.matmul(D, batch_alpha)
            #recon = softmax_transform(recon)
            sample_losses = ((batch - recon) ** 2).sum(dim=(1, 2))
            if regularization == 'l2':
                sample_losses += lambda_reg * torch.norm(batch_alpha, p='fro')**2
            elif regularization == 'l1':
                sample_losses += lambda_reg * batch_alpha.abs().sum(dim=(1, 2))
            else:
                sample_losses += torch.tensor(0.0, device=device)
            for i, sample_idx in enumerate(idx):
                if sample_losses[i] < best_losses[sample_idx]:
                    best_losses[sample_idx] = sample_losses[i]
                    best_alpha[sample_idx] = batch_alpha[i].detach()

    return best_alpha, alpha_params, best_losses

# ==============================
# test
# ==============================

def test_dictionary_learning(flow_file, dictionary_file, lambda_reg=0.01, alpha_steps=100, lr=1e-2, batch_size=32, regularization='l2'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    sys.stdout.flush()

    F, D = load_data(flow_file, dictionary_file, device)
    T, n, _ = F.shape
    k = D.shape[1]

    dataset = FlowDataset(F)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # initialization
    s = 10
    torch.manual_seed(s)
    print(f"Random seed: {s}")
    sys.stdout.flush()
    D_params = D.clone().detach().requires_grad_(False)
    alpha_params = torch.randn(T, k, n, device=device, requires_grad=True)

    loss_vector = []
    best_alpha = None
    k = D.shape[1]
        
        # alpha optimization
    alpha, alpha_params, losses = optimize_alpha_batches(
            D_params, 
            alpha_params, 
            loader, 
            T, k, lambda_reg, 
            device, 
            n_iter=alpha_steps, 
            lr=lr, 
            regularization=regularization
        )

    print("Alpha optimization finished")
    sys.stdout.flush()
    loss_vector.append(losses.cpu().numpy())
    best_alpha = alpha.clone().detach()

    return best_alpha, loss_vector

if __name__ == '__main__':
    # python3 test_dict.py -dir None -system little_experiment -flows synthetic_data/flows.npy -dict synthetic_data/dictionary.npy -reg l1 -lambda 0.001 -as (as*epochs) -lr 1e-4 -bs 4

    parser = argparse.ArgumentParser(description='Dictionary Learning for Arrival Flows')
    parser.add_argument('-dir', '--directory', type=str, default=None, help='Directory to save results', required=False)
    parser.add_argument('-system', '--system_key', type=str, default='experiment', help='system of flows', required=True)
    parser.add_argument('-flows', '--flows', type=str, default='flows.npy', help='Path to the flow tensor file', required=True)
    parser.add_argument('-dict', '--dictionary', type=str, default='dictionary.npy', help='Path to the dictionary file', required=True)
    parser.add_argument('-reg', '--regularization', type=str, default='l2', help='which regularization', required=True)
    parser.add_argument('-lambda', '--lambda_reg', type=float, default=0.01, help='regularization parameter', required=True)
    parser.add_argument('-as', '--alpha_steps', type=int, default=100, help='number of steps for alpha optimization', required=True)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2, help='learning rate', required=True)
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size', required=True)
    args = parser.parse_args()

    directory = args.directory
    system = args.system_key
    flow_path = args.flows
    dictionary_path = args.dictionary
    lambda_reg = args.lambda_reg
    alpha_steps = args.alpha_steps
    lr = args.learning_rate
    batch_size = args.batch_size
    regularization = args.regularization

    print(f"System: {system}")
    sys.stdout.flush()
    print(f"Flows: {flow_path}")
    sys.stdout.flush()
    print(f"Dictionary: {dictionary_path}")
    sys.stdout.flush()
    print(f"Regularization: {regularization}")
    sys.stdout.flush()
    print(f"Lambda: {lambda_reg}")
    sys.stdout.flush()
    print(f"Alpha steps: {alpha_steps}")
    sys.stdout.flush()
    print(f"Learning rate: {lr}")
    sys.stdout.flush()
    print(f"Batch size: {batch_size}\n\n")
    sys.stdout.flush()

    print('Starting testing')
    sys.stdout.flush()
    start = time.time()

    alpha, loss = test_dictionary_learning(flow_path, dictionary_path, 
                                             lambda_reg=lambda_reg, 
                                             alpha_steps=alpha_steps, 
                                             lr=lr, 
                                             batch_size=batch_size, 
                                             regularization=regularization
                                             )
    
    end = time.time()
    print(f'Testing time: {(end - start)/60:.2f} minutes')
    sys.stdout.flush()
    

    if directory is not None:
        save_dir = os.path.join(directory, f'results_{system}')
    else:      
        save_dir = f'results_{system}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_params = open(os.path.join(save_dir, 'params_test.txt'), 'w')
    file_params.write(f"system: {system}\n")
    file_params.write(f"flows: {flow_path}\n")
    file_params.write(f"number epochs that was required: {len(loss)}\n")
    file_params.write(f"regularization: {regularization}\n")
    file_params.write(f"lambda: {lambda_reg}\n")
    file_params.write(f"alpha steps: {alpha_steps}\n")
    file_params.write(f"learning rate: {lr}\n")
    file_params.write(f"batch size: {batch_size}\n")
    file_params.write(f"Mean loss: {np.mean(loss):.3f}\n")
    file_params.write(f"final time: {(end - start)/60:.2f} minutes\n")
    file_params.close()

    np.save(os.path.join(save_dir, 'weights_test.npy'), alpha.cpu().numpy())

    print('Finished')
    sys.stdout.flush()


os.system(f'curl -d "Finishing test with {system}" ntfy.sh/aamh_091099_ntfy')