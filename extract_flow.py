import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description="Extract flow from a file")
parser.add_argument("-in", "--input_file", type=str, help="Path to the input file containing flow data", required=True)
parser.add_argument("-index", "--index_flow", type=int, help="Index of the flow to extract", required=True)
args = parser.parse_args()

tensor = np.load(args.input_file, allow_pickle=True)

dir_to_save = os.path.dirname(args.input_file)
if not os.path.exists(dir_to_save):
    os.makedirs(dir_to_save)

flow = tensor[args.index_flow]

np.save(os.path.join(dir_to_save, f"flow_{args.index_flow}.npy"), flow)