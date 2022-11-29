import os
import sys

import pickle as pkl


if __name__ == '__main__':
    if len(sys.argv) != 2:
        filename = os.path.basename(__file__)
        
        print('Usage: ' + filename + ' [Directory containing pickled data]')
        sys.exit(1)

    directory = sys.argv[1]

    if not os.path.exists(directory):
        print('Error: Directory ' + directory + ' does not exist')
        sys.exit(1)

    if not os.path.isdir(directory):
        print('Error: ' + directory + ' is not a directory')
        sys.exit(1)

    if not os.path.exists('out'):
        os.makedirs('out')


    avg_error_per_file = []

    for filename in filter(lambda x: True, os.listdir(directory)):
        file_path = os.path.join(directory, filename)

        with open(file_path, 'rb') as f:
            data = pkl.loads(f.read())

        approx_block_dimension = data['approx_block_dimension']

        embeddings_len_y = len(data['embeddings'][0]['standard']['embeddings'])
        embeddings_len_x = len(data['embeddings'][0]['standard']['embeddings'][0])

        assert embeddings_len_y == len(data['embeddings'][0]['approximate']['embeddings'])
        assert embeddings_len_x == len(data['embeddings'][0]['approximate']['embeddings'][0])

        all_error = []
        
        for y in range(embeddings_len_y):
            for x in range(embeddings_len_x):
                actual = data['embeddings'][0]['standard']['embeddings'][y][x]
                approx = data['embeddings'][0]['approximate']['embeddings'][y][x]

                if actual != 0.0:
                    error = abs(actual - approx) / abs(actual)

                    all_error.append(error)

        avg_error = sum(all_error) / len(all_error)

        print(avg_error)
        
        avg_error_per_file.append(avg_error)

    print()
    print(sum(avg_error_per_file) / len(avg_error_per_file))

        # print(data['embeddings'][0]['standard']['embeddings'][30][30])
        # print(data['embeddings'][0]['approximate']['embeddings'][30][30])


