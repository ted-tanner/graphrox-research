import csv
import os
import sys

import numpy as np
import pickle as pkl


def remove_type_from_dict(value_type, dictionary):
    # Createing a stack here instead of using "the stack" to avoid overflowing the stack
    # for deeply nested dictionaries
    stack = [dictionary]

    while len(stack) != 0:
        current = stack.pop()

        for k, v in current.items():
            if isinstance(v, value_type):
                current[k] = None
            elif isinstance(v, dict):
                stack.append(v)

    return dictionary


if __name__ == '__main__':
    if len(sys.argv) < 3:
        filename = os.path.basename(__file__)
        
        print('Usage: ' + filename + ' [Directory containing pickled data] [Output file]')
        print('\tOptional: --filter-contains [String]')
        sys.exit(1)

    directory = sys.argv[1]
    out_file_path = sys.argv[2]

    if not os.path.exists(directory):
        print('Error: Directory ' + directory + ' does not exist')
        sys.exit(1)

    if not os.path.isdir(directory):
        print('Error: ' + directory + ' is not a directory')
        sys.exit(1)

    fltr = lambda x: True

    if len(sys.argv) >= 5 and '--filter-contains' in sys.argv:
        filter_arg_idx = sys.argv.index('--filter-contains')

        if filter_arg_idx != len(sys.argv) - 1:
            fltr = lambda x: sys.argv[filter_arg_idx + 1] in x

    output = []
    avg_error_per_file = []

    for filename in filter(fltr, os.listdir(directory)):
        file_path = os.path.join(directory, filename)

        with open(file_path, 'rb') as f:
            data = pkl.loads(f.read())

        metadata = { k: data[k] for k in set(list(data.keys())) - {'embeddings'} }
        
        for i, emb in enumerate(data['embeddings']):
            std_emb = emb['standard']['embeddings']
            std_emb_time = emb['standard']['execution_time']

            if 'approximate' in emb:
                approx_emb = emb['approximate']['embeddings']
                approx_emb_time = emb['approximate']['execution_time']
            else:
                approx_emb = emb['compressed']['embeddings']
                approx_emb_time = emb['compressed']['execution_time']

            embeddings_len_y = len(std_emb)
            embeddings_len_x = len(std_emb[i])

            assert embeddings_len_y == len(approx_emb)
            assert embeddings_len_x == len(approx_emb[i])

            curr_error = []

            for y in range(embeddings_len_y):
                for x in range(embeddings_len_x):
                    actual = std_emb[y][x]
                    approx = approx_emb[y][x]

                    if actual != 0.0:
                        error = abs(actual - approx) / abs(actual)
                        curr_error.append(error)

            curr_out = metadata.copy()
            curr_out['algorithm'] = emb['name']
            curr_out['avg_error'] = sum(curr_error) / len(curr_error) if len(curr_error) != 0 else 0
            curr_out['time_diff'] = approx_emb_time / std_emb_time

            output.append(curr_out)
            print(curr_out)
        

    if len(output) != 0:
        csv_keys = output[0].keys()
        with open(out_file_path, 'w', newline='') as outf:
            csv_writer = csv.DictWriter(outf, csv_keys)
            csv_writer.writeheader()
            csv_writer.writerows(output)
    else:
        print('ERROR: No embeddings found')



