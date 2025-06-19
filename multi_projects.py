import subprocess

experiments = [
    ['python3', 'train_dict.py', '-dir', 'exp_train_dict', '-system', 'exp_train_dict/experiment_v5_2', '-flows', 'exp_train_dict/synthetic_data_v5/flows.npy', '-lap', 'exp_train_dict/synthetic_data_v5/laplacian.npy', '-natoms', '9', '-ep', '800', '-reg', 'l1', '-lambda', '0.0001', '-smooth', '1', '-gamma', '0.00001', '-as', '45', '-ds', '25', '-lr', '1e-4', '-bs', '128', '-tol', '1e-5', '-pat', '150'],
    ['python3', 'train_dict.py', '-dir', 'exp_train_dict', '-system', 'exp_train_dict/experiment_v5_3', '-flows', 'exp_train_dict/synthetic_data_v5/flows.npy', '-lap', 'exp_train_dict/synthetic_data_v5/laplacian.npy', '-natoms', '10', '-ep', '800', '-reg', 'l1', '-lambda', '0.0001', '-smooth', '1', '-gamma', '0.00001', '-as', '45', '-ds', '25', '-lr', '1e-4', '-bs', '128', '-tol', '1e-5', '-pat', '150'],
    ['python3', 'train_dict.py', '-dir', 'exp_train_dict', '-system', 'exp_train_dict/experiment_v5_4', '-flows', 'exp_train_dict/synthetic_data_v5/flows.npy', '-lap', 'exp_train_dict/synthetic_data_v5/laplacian.npy', '-natoms', '11', '-ep', '800', '-reg', 'l1', '-lambda', '0.0001', '-smooth', '1', '-gamma', '0.00001', '-as', '45', '-ds', '25', '-lr', '1e-4', '-bs', '128', '-tol', '1e-5', '-pat', '150'],
    ['python3', 'train_dict.py', '-dir', 'exp_train_dict', '-system', 'exp_train_dict/experiment_v5_5', '-flows', 'exp_train_dict/synthetic_data_v5/flows.npy', '-lap', 'exp_train_dict/synthetic_data_v5/laplacian.npy', '-natoms', '12', '-ep', '800', '-reg', 'l1', '-lambda', '0.0001', '-smooth', '1', '-gamma', '0.00001', '-as', '45', '-ds', '25', '-lr', '1e-4', '-bs', '128', '-tol', '1e-5', '-pat', '150']
]

for cmd in experiments:
    subprocess.run(cmd)