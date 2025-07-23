import os
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.stdout.flush()

original_flows_path = "ecobici_flows_dataset_4_split80/flows_train.npy"

indexes = [3, 416, 843, 1253, 1669, 2050]
total_projects = 672

original_flows = np.load(original_flows_path)

for i in range(total_projects):
    print(f'Processing experiment {i+1}/{total_projects}')
    sys.stdout.flush()
    currect_path = f'results/results_ecobici_experiment_{i}/'

    dictionary = np.load(os.path.join(currect_path, 'dictionary.npy'))
    weights = np.load(os.path.join(currect_path, 'weights.npy'))

    for index in indexes:
        original_flow = original_flows[index]
        reconstructed_flow = dictionary @ weights[index]

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.title(f'Original Flow {index}')
        plt.imshow(original_flow, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title(f'Reconstructed Flow {index} from Experiment {i}')
        plt.imshow(reconstructed_flow, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(currect_path, f'reconstruction_{index}.png'))
        plt.close()