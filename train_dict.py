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

def load_data(flow_file, laplacian_file, device):
    '''This function loads the flow and laplacian data from files.'''
    F = np.load(flow_file)  # (T, n, n)
    L = np.load(laplacian_file)  # (n, n)
    F = torch.tensor(F, dtype=torch.float32, device=device)
    L = torch.tensor(L, dtype=torch.float32, device=device)
    return F, L

def softmax_transform(x):
    x_ = x.flatten()
    x_softmax = torch.softmax(x_, dim=0)
    x_softmax = x_softmax.reshape(x.shape)
    return x_softmax

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
            for i, sample_idx in enumerate(idx):
                if sample_losses[i] < best_losses[sample_idx]:
                    best_losses[sample_idx] = sample_losses[i]
                    best_alpha[sample_idx] = batch_alpha[i].detach()

    return best_alpha, alpha_params

# ==============================
# dictionary optimization with cumulative gradients
# ==============================

def optimize_dictionary_batches(F, alpha, initial_D_params, L, flow_loader, gamma_reg, n_iter=100, lr=1e-2, smooth=True):
    T, n, _ = F.shape
    k = alpha.shape[1]
    device = F.device

    D_params = initial_D_params.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([D_params], lr=lr)

    best_D = initial_D_params.clone().detach()
    #softmax_transform(D_params.detach().clone(), dim=0)
    best_loss = float('inf')

    for _ in range(n_iter):
        optimizer.zero_grad()

        D = D_params
        #softmax_transform(D_params, dim=0)
        
        total_loss_tensor = 0.0
        total_loss_rec = 0.0

        for batch, idx in flow_loader:
            batch = batch.to(device)
            batch_alpha = alpha[idx]

            recon = torch.matmul(D, batch_alpha)
            #recon = softmax_transform(recon)  
            loss_rec = ((batch - recon) ** 2).sum()
            total_loss_tensor += loss_rec
            total_loss_rec += loss_rec.item()

        total_loss_rec /= (2 * T)

        # spatial smoothness regularization
        if smooth:
            loss_smooth = gamma_reg * torch.trace(D.T @ L @ D)
            total_loss_tensor += loss_smooth
            loss_smooth_val = loss_smooth.item()
        else:
            loss_smooth_val = 0.0

        total_loss = (total_loss_rec + loss_smooth_val)

        # only update if the loss is better than the best loss
        if total_loss < best_loss:
            total_loss_tensor.backward()
            optimizer.step()
            with torch.no_grad():
                D_params.clamp_(0.0, 1.0)
            best_loss = total_loss
            best_D = D_params.clone().detach()
            #best_D = softmax_transform(D_params.detach().clone(), dim=0)

    return best_D, D_params, best_loss

# ==============================
# train 
# ==============================

def train_dictionary_learning(flow_file, laplacian_file, k=10, n_epochs=10, lambda_reg=0.01, gamma_reg=0.1, alpha_steps=100, d_steps=100, lr=1e-2, batch_size=32, regularization='l2', smooth=False, pat=15, tol=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    sys.stdout.flush()

    F, L = load_data(flow_file, laplacian_file, device)
    T, n, _ = F.shape

    dataset = FlowDataset(F)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # initialization
    s = 10
    torch.manual_seed(s)
    print(f"Random seed: {s}")
    sys.stdout.flush()
    D_params = torch.randn(n, k, device=device, requires_grad=True)
    alpha_params = torch.randn(T, k, n, device=device, requires_grad=True)

    best_loss = float('inf')
    patience = pat
    loss_vector = []
    best_D = None
    best_alpha = None

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        sys.stdout.flush()

        # current D
        current_D = D_params.clone().detach()
        #softmax_transform(D_params.detach(), dim=0)
        
        # alpha optimization
        alpha, alpha_params = optimize_alpha_batches(
            current_D, 
            alpha_params, 
            loader, 
            T, k, lambda_reg, 
            device, 
            n_iter=alpha_steps, 
            lr=lr, 
            regularization=regularization
        )

        # dictionary optimization
        D, D_params, loss = optimize_dictionary_batches(
            F, 
            alpha, 
            D_params, 
            L, 
            loader, 
            gamma_reg, 
            n_iter=d_steps, 
            lr=lr, 
            smooth=smooth
        )

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
            best_loss = loss
            print(f'Best D and alpha updated in epoch {epoch+1}')
            sys.stdout.flush()
            
        if abs(loss - best_loss) < tol:
            patience = pat
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping")
                sys.stdout.flush()
                break

    print("Training finished")
    sys.stdout.flush()

    return best_D, best_alpha, loss_vector, best_loss.item()

# ==============================
# main
# ==============================

if __name__ == '__main__':
    # python3 train_dict.py -dir None -system little_experiment -flows synthetic_data/flows.npy -lap synthetic_data/laplacian.npy -natoms 4 -ep 1500 -reg l1 -lambda 0.001 -smooth 0 -gamma 0.001 -as 25 -ds 25 -lr 1e-4 -bs 4 -tol 1e-5 -pat 80

    parser = argparse.ArgumentParser(description='Dictionary Learning for Arrival Flows')
    parser.add_argument('-dir', '--directory', type=str, default=None, help='Directory to save results', required=False)
    parser.add_argument('-system', '--system_key', type=str, default='experiment', help='system of flows', required=True)
    parser.add_argument('-flows', '--flows', type=str, default='flows.npy', help='Path to the flow tensor file', required=True)
    parser.add_argument('-lap', '--laplacian', type=str, default='laplacian.npy', help='Path to the laplacian file', required=True)
    parser.add_argument('-natoms', '--number_atoms', type=int, default=10, help='Number of dictionary elements', required=True)
    parser.add_argument('-ep', '--epochs', type=int, default=10, help='Number of epochs', required=True)
    parser.add_argument('-reg', '--regularization', type=str, default='l2', help='which regularization', required=True)
    parser.add_argument('-lambda', '--lambda_reg', type=float, default=0.01, help='regularization parameter', required=True)
    parser.add_argument('-smooth', '--smooth', type=int, default=0, help='smoothness 1(True)/0(False)', required=True)
    parser.add_argument('-gamma', '--gamma_reg', type=float, default=0.1, help='smoothness regularization parameter', required=True)
    parser.add_argument('-as', '--alpha_steps', type=int, default=100, help='number of steps for alpha optimization', required=True)
    parser.add_argument('-ds', '--dict_steps', type=int, default=100, help='number of steps for dictionary optimization', required=True)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2, help='learning rate', required=True)
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='batch size', required=True)
    parser.add_argument('-tol', '--tolerance', type=float, default=1e-4, help='tolerance for early stopping', required=False)
    parser.add_argument('-pat', '--patience', type=int, default=15, help='patience for early stopping', required=False)
    args = parser.parse_args()

    directory = args.directory
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
    tol = args.tolerance
    patience = args.patience

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
    print(f"Batch size: {batch_size}")
    sys.stdout.flush()
    print(f"Tolerance: {tol}")
    sys.stdout.flush()
    print(f"Patience: {patience}\n\n")
    sys.stdout.flush()

    print('Starting training')
    sys.stdout.flush()
    start = time.time()

    D, alpha, loss, bl = train_dictionary_learning(flow_path, lap_path, 
                                             k=k, 
                                             n_epochs=n_epochs, 
                                             lambda_reg=lambda_reg, 
                                             gamma_reg=gamma_reg, 
                                             alpha_steps=alpha_steps, 
                                             d_steps=d_steps, 
                                             lr=lr, 
                                             batch_size=batch_size, 
                                             regularization=regularization, 
                                             smooth=smooth,
                                             tol=tol,
                                             pat=patience
                                             )
    
    end = time.time()
    print(f'Training time: {(end - start)/60:.2f} minutes')
    sys.stdout.flush()
    print('Mean time per epoch: {:.2f} seconds'.format((end - start) / n_epochs))
    sys.stdout.flush()

    if directory is not None:
        save_dir = os.path.join(directory, f'results_{system}')
    else:      
        save_dir = f'results_{system}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    D_s = D.cpu().numpy().shape

    file_params = open(os.path.join(save_dir, 'params.txt'), 'w')
    file_params.write(f"system: {system}\n")
    file_params.write(f'number of nodes: {D_s[0]}\n')
    file_params.write(f"flows: {flow_path}\n")
    file_params.write(f"laplacian: {lap_path}\n")
    file_params.write(f"number of atoms: {k}\n")
    file_params.write(f"total epochs: {n_epochs}\n")
    file_params.write(f"number epochs that was required: {len(loss)}\n")
    file_params.write(f"regularization: {regularization}\n")
    file_params.write(f"lambda: {lambda_reg}\n")
    file_params.write(f"smoothness: {smooth}\n")
    file_params.write(f"gamma: {gamma_reg}\n")
    file_params.write(f"alpha steps: {alpha_steps}\n")
    file_params.write(f"dictionary steps: {d_steps}\n")
    file_params.write(f"learning rate: {lr}\n")
    file_params.write(f"batch size: {batch_size}\n")
    file_params.write(f"final loss: {loss[-1]}\n")
    file_params.write(f"best loss: {bl}\n")
    file_params.write(f"final time: {(end - start)/60:.2f} minutes\n")
    file_params.write(f"mean time per epoch: {(end - start) / len(loss):.2f} seconds\n")
    file_params.write(f"tolerance: {tol}\n")
    file_params.write(f"patience: {patience}\n")
    file_params.close()

    np.save(os.path.join(save_dir, 'dictionary.npy'), D.cpu().numpy())
    np.save(os.path.join(save_dir, 'weights_train.npy'), alpha.cpu().numpy())
    np.save(os.path.join(save_dir, 'loss.npy'), np.array(loss))

    plt.plot(loss)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over epochs')
    plt.grid()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()

    print('Finished')
    sys.stdout.flush()


os.system(f'curl -d "Finishing training with {system}" ntfy.sh/aamh_091099_ntfy')