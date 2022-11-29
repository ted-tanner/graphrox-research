import json
import os
import sys

import graphrox as gx
import networkx as nx
import pickle as pkl

from karateclub import FeatherGraph, LDP, Graph2Vec

import synthetic_embeddings as se


if __name__ == '__main__':
    if len(sys.argv) != 3:
        filename = os.path.basename(__file__)
        
        print('Usage: ' + filename + ' [Edges file path] [Data set name]')
        sys.exit(1)

    edges_file_path = sys.argv[1]
    dataset_name = sys.argv[2]

    try:
        with open(edges_file_path) as edges_file:
            data = json.loads(edges_file.read())
    except FileNotFoundError:
        print('Error: ' + edges_file_path + ' was not found')
        sys.exit(1)
    except json.JSONDecodeError:
        print('Error: ' + edges_file_path + ' contained invalid JSON')
        sys.exit(1)

    if not os.path.exists('out'):
        os.makedirs('out')

    print('Loading graphs...')
    nx_graphs = []
    gx_graphs = []

    for edge_list in data.values():
        nx_graph = nx.Graph()
        gx_graph = gx.Graph(is_undirected=True)

        for edge in edge_list:
            nx_graph.add_edge(edge[0], edge[1])
            gx_graph.add_edge(edge[0], edge[1])

        nx_graphs.append(nx_graph)
        gx_graphs.append(gx_graph)

    print('Generating embeddings for uncompressed graph...')

    std_embeddings = []

    std_embeddings.append({
        'name': 'FeatherGraph',
        'embeddings': se.get_embeddings(nx_graphs, FeatherGraph()),
    })

    std_embeddings.append({
        'name': 'LDP',
        'embeddings': se.get_embeddings(nx_graphs, LDP()),
    })

    std_embeddings.append({
        'name': 'Graph2Vec',
        'embeddings': se.get_embeddings(nx_graphs, Graph2Vec()),
    })
        
    print('Compressing and decompressing graph, generating embeddings...')

    compression_level_set = [2, 4, 8, 16, 32]

    total_iterations = len(compression_level_set)
    current_iteration = 0
    
    for compression_level in compression_level_set:
        compressed_graphs = []
        
        for graph in gx_graphs:
            gx_compressed_graph = graph.compress(compression_level)
            compressed_graphs.append(se.graphrox_to_networkx(gx_compressed_graph))
            
        embeddings = []

        embeddings.append({
            'name': 'FeatherGraph',
            'embeddings': se.get_embeddings(compressed_graphs, FeatherGraph()),
        })
        
        embeddings.append({
            'name': 'LDP',
            'embeddings': se.get_embeddings(compressed_graphs, LDP()),
        })
        
        embeddings.append({
            'name': 'Graph2Vec',
            'embeddings': se.get_embeddings(compressed_graphs, Graph2Vec()),
        })
        
        data = {
            'compression_level': compression_level,
            'graph_count': len(nx_graphs),
            'standard_embeddings': {
                ''
            },
            'compressed_embeddings': embeddings,
        }
        
        filename = f'out/{dataset_name}_emb_c({len(nx_graphs)})_t({compression_level}).pkl'
        with open(filename, 'wb') as out_file:
            pkl.dump(data, out_file)
            
        current_iteration += 1
            
        print('Progress: {}%'.format(int((current_iteration / total_iterations) * 100)))
