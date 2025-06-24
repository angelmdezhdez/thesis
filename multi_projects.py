import subprocess

experiments = [
    ['python3', 'train_dict.py', '-dir', 'exp_train_dict', '-system', 'exp_train_dict/experiment_mibici_1', '-flows', 'exp_train_dict/mibici_dataset_4/flows_train.npy', '-lap', 'exp_train_dict/mibici_dataset_4/laplacian_mibici_4.npy', '-natoms', '12', '-ep', '1000', '-reg', 'l1', '-lambda', '0.0001', '-smooth', '1', '-gamma', '0.00001', '-as', '45', '-ds', '25', '-lr', '1e-4', '-bs', '32', '-tol', '1e-5', '-pat', '150'],
    ['python3', 'train_dict.py', '-dir', 'exp_train_dict', '-system', 'exp_train_dict/experiment_mibici_2', '-flows', 'exp_train_dict/mibici_dataset_4/flows_train.npy', '-lap', 'exp_train_dict/mibici_dataset_4/laplacian_mibici_4.npy', '-natoms', '16', '-ep', '1000', '-reg', 'l1', '-lambda', '0.0001', '-smooth', '1', '-gamma', '0.00001', '-as', '45', '-ds', '25', '-lr', '1e-4', '-bs', '32', '-tol', '1e-5', '-pat', '150'],
    ['python3', 'train_dict.py', '-dir', 'exp_train_dict', '-system', 'exp_train_dict/experiment_mibici_3', '-flows', 'exp_train_dict/mibici_dataset_4/flows_train.npy', '-lap', 'exp_train_dict/mibici_dataset_4/laplacian_mibici_4.npy', '-natoms', '20', '-ep', '1000', '-reg', 'l1', '-lambda', '0.0001', '-smooth', '1', '-gamma', '0.00001', '-as', '45', '-ds', '25', '-lr', '1e-4', '-bs', '32', '-tol', '1e-5', '-pat', '150'],
    ['python3', 'train_dict.py', '-dir', 'exp_train_dict', '-system', 'exp_train_dict/experiment_mibici_4', '-flows', 'exp_train_dict/mibici_dataset_4/flows_train.npy', '-lap', 'exp_train_dict/mibici_dataset_4/laplacian_mibici_4.npy', '-natoms', '24', '-ep', '1000', '-reg', 'l1', '-lambda', '0.0001', '-smooth', '1', '-gamma', '0.00001', '-as', '45', '-ds', '25', '-lr', '1e-4', '-bs', '32', '-tol', '1e-5', '-pat', '150']
]

for cmd in experiments:
    subprocess.run(cmd)