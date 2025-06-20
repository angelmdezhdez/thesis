import numpy as np
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Visualize synthetic data experiment results')
    args.add_argument('-I', '--Index', type=int, default=-1, help='index of the flow to visualize')
    args = args.parse_args()

    #dir = 'exp_train_dict/synthetic_data_v5/'
    dir = 'exp_train_dict/mibici_dataset_4/'
    #dir_exp = 'exp_train_dict/results_exp_train_dict/experiment_v5_3/'
    dir_exp = 'exp_train_dict/results_exp_train_dict/experiment_mibici_1/'

    s_flows = np.load(dir + 'flows_train.npy')
    #s_flows = np.load(dir + 'flows.npy')
    print('Shape of synthetic flows:', s_flows.shape)
    #s_dict = np.load(dir + 'dictionary.npy')
    s_dict = np.load(dir + 'laplacian_mibici_4.npy')

    learned_dict = np.load(dir_exp + 'dictionary.npy')
    print('Shape of learned dictionary:', learned_dict.shape)
    learned_weights = np.load(dir_exp + 'weights.npy')

    index = args.Index

    if index != -1:
        ran_index = index
    else:
        ran_index = np.random.randint(0, s_flows.shape[0])

    print('Random index:', ran_index)
    print('Flow selected:\n', s_flows[ran_index])

    reconstructed_flow = np.dot(learned_dict, learned_weights[ran_index])
    print('Reconstructed flow:\n', reconstructed_flow)
    print('Learned weights:\n', learned_weights[ran_index])

    print('Original dictionary:\n', s_dict)
    print('Learned dictionary:\n', learned_dict)

    plt.figure(figsize=(12, 8))
    plt.suptitle(f'Comparison of Original and Learned Dictionaries with Reconstructed Flow of index [{ran_index}]', fontsize=16)
    plt.subplot(2, 2, 1)
    plt.imshow(s_flows[ran_index], cmap='viridis', interpolation='nearest')
    plt.title('Original Flow')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 2)
    plt.imshow(reconstructed_flow, cmap='viridis', interpolation='nearest')
    plt.title('Reconstructed Flow')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 2, 3)
    plt.imshow(s_dict, cmap='viridis', interpolation='nearest')
    #plt.title('Original Dictionary')
    plt.title('Laplacian')
    plt.colorbar()
    plt.xticks([i for i in range(s_dict.shape[1])])
    plt.yticks([])
    plt.subplot(2, 2, 4)
    plt.imshow(learned_dict, cmap='viridis', interpolation='nearest')
    plt.title('Learned Dictionary')
    plt.colorbar()
    plt.xticks([i for i in range(learned_dict.shape[1])])
    plt.yticks([])
    plt.tight_layout()
    plt.show()

    n = 300
    loss = np.load(dir_exp + 'loss.npy')
    plt.figure(figsize=(10, 5))
    plt.plot(range(n, len(loss)), loss[n:], label='Loss')
    plt.title(f'Training Loss since epoch {n}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
