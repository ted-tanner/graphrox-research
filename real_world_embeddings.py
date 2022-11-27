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

    block_dimension_set = [2, 3, 4, 5]
    threshold_set = [0.1, 0.2, 0.3, 0.4]

    total_iterations = len(block_dimension_set) * len(threshold_set)
    current_iteration = 0

    # These loops will repeatedly regenerate embeddings for nx_graphs. This can be sped up by
    # eliminating this redundant work, but for now it is desireable to redo the calculation
    # to account for variance in available compute capacity during the run
    print('Approximating and generating embeddings...')
    for block_dimension in block_dimension_set:
        for threshold in threshold_set:
            approx_graphs = []
            
            for graph in gx_graphs:
                gx_approx_graph = graph.approximate(block_dimension, threshold)
                approx_graphs.append(se.graphrox_to_networkx(gx_approx_graph))

            embeddings = []

            emb, approx_emb = se.get_embeddings(nx_graphs, approx_graphs, FeatherGraph())
            embeddings.append({
                'name': 'FeatherGraph',
                'standard': emb,
                'approximate': approx_emb,
            })

            emb, approx_emb = se.get_embeddings(nx_graphs, approx_graphs, LDP())
            embeddings.append({
                'name': 'LDP',
                'standard': emb,
                'approximate': approx_emb,
            })

            emb, approx_emb = se.get_embeddings(nx_graphs, approx_graphs, Graph2Vec())
            embeddings.append({
                'name': 'Graph2Vec',
                'standard': emb,
                'approximate': approx_emb,
            })

            data = {
                'approx_block_dimension': block_dimension,
                'approx_threshold': threshold,
                'graph_count': len(nx_graphs),
                'embeddings': embeddings,
            }

            filename = f'out/{dataset_name}_emb_c({len(nx_graphs)})_b({block_dimension})_t({threshold}).pkl'
            with open(filename, 'wb') as out_file:
                pkl.dump(data, out_file)

            current_iteration += 1

            print('Progress: {}%'.format(int((current_iteration / total_iterations) * 100)))
