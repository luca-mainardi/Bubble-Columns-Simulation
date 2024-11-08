import argparse
import pickle
import numpy as np
import pandas as pd
import os

def main():
    parser = argparse.ArgumentParser(description='Convert pkl to npz')
    parser.add_argument('--pickle-dir', type=str, help='Directory containing pickle files')
    parser.add_argument('--output-dir', type=str, help='Output file path')
    parser.add_argument('--store-as', type=str, default='csv', help='csv or npz')
    args = parser.parse_args()

    if os.path.exists(args.output_dir) == False:
        os.makedirs(args.output_dir)

    for pkl_filename in os.listdir(args.pickle_dir):
        if pkl_filename.endswith('.pkl'):
            file_name = pkl_filename.split('.')[0]
            pkl_path = os.path.join(args.pickle_dir, pkl_filename)
            df = pd.read_pickle(pkl_path)

            if args.store_as == 'csv':
                output_path = args.output_dir + file_name + '.csv'
                df.to_csv(output_path, index=False)
            else:
                output_path = args.output_dir + file_name + '.npz'
                np.savez(output_path, data=df.values, header=df.columns)

if __name__ == '__main__':
    main()