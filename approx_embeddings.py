# This is only needed if you get an SSL error. This is not a great
# thing to do as it disables SSL verification entirely and it's not a
# good idea to download things without verifying the authenticity of
# the source.
######################################################################
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
######################################################################

import graphrox as gx
import networkx as nx
import os

from multiprocessing import Manager, Process, Queue

from karateclub import FeatherGraph, IGE, LDP, GL2Vec, Graph2Vec, NetLSD, SF, FGSD
from karateclub import WaveletCharacteristic as WvChr, GeoScattering as GS

from karateclub.dataset import GraphSetReader


def process_embeddings(work_queue, embeddings, graphs, approx_graphs):
    while not work_queue.empty():
        job = work_queue.get()
        
        emb, approx_emb = get_embeddings(graphs, approx_graphs, job['model'])
        embeddings.append({
            'name': job['name'],
            'standard': emb,
            'approximate': approx_emb,
        })

        print('Generated ' + job['name'] + ' embedding')


def networkx_to_graphrox(nx_graph):
    """Converts a NetworkX graph to GraphRox graph"""
    gx_graph = gx.Graph(is_undirected=True)
    
    for from_edge, to_edge in nx_graph.edges():
        gx_graph.add_edge(from_edge, to_edge)

    if nx_graph.number_of_nodes() > gx_graph.vertex_count():
        gx_graph.add_vertex(nx_graph.number_of_nodes() - 1)

    return gx_graph


def graphrox_to_networkx(gx_graph):
    """Converts a NetworkX graph to GraphRox graph"""
    nx_graph = nx.Graph()
    
    for from_edge, to_edge in gx_graph.edge_list():
        nx_graph.add_edge(from_edge, to_edge)

    if gx_graph.vertex_count() > nx_graph.number_of_nodes():
        nx_graph.add_node(nx_graph.vertex_count() - 1)

    return nx_graph


def obtain_graphset():
    """Yields the 'reddit10k' Karate Club dataset as a tuple of GraphRox and NetworkX graphs"""
    reader = GraphSetReader('reddit10k')
    nx_graphs = reader.get_graphs()
    target_vec = reader.get_target()

    gx_graphs = []

    for graph in nx_graphs:
        gx_graphs.append(networkx_to_graphrox(graph))

    return (nx_graphs, gx_graphs)


def approximate_gx_graphs(gx_graphs, block_dimension, threshold):
    """Returns duplicated and approximated versions of the gx_graphs"""
    return [g.duplicate().approximate(block_dimension, threshold) for g in gx_graphs]


def get_embeddings(graphs, approx_graphs, model):
    model.fit(graphs)
    embeddings = model.get_embedding()

    model.fit(approx_graphs)
    approx_embeddings = model.get_embedding()

    return (embeddings, approx_embeddings)


if __name__ == '__main__':
    print('Obtaining graphs...')
    nx_graphs, gx_graphs = obtain_graphset()

    print('Approximating graphs...')
    gx_approx_graphs = approximate_gx_graphs(gx_graphs, 2, 0.25)

    nx_approx_graphs = []

    for graph in gx_approx_graphs:
        nx_approx_graphs.append(graphrox_to_networkx(graph))

    work_queue = Queue()
    manager = Manager()
    embeddings = manager.list()

    work_queue.put({'name': 'FeatherGraph', 'model': FeatherGraph()})
    work_queue.put({'name': 'LDP', 'model': LDP()})
    work_queue.put({'name': 'WaveletCharacteristic', 'model': WvChr()})
    work_queue.put({'name': 'GeoScattering', 'model': GS()})
    work_queue.put({'name': 'GL2Vec', 'model': GL2Vec()})
    work_queue.put({'name': 'Graph2Vec', 'model': Graph2Vec()})
    work_queue.put({'name': 'NetLSD', 'model': NetLSD()})
    work_queue.put({'name': 'SF', 'model': SF()})
    work_queue.put({'name': 'FGSD', 'model': FGSD()})

    print('Generating embeddings...')
    print()

    processes = []

    for i in range(os.cpu_count()):
        process = Process(target=process_embeddings, args=(work_queue,
                                                           embeddings,
                                                           nx_graphs,
                                                           nx_approx_graphs))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    print()
    print(len(embeddings))

    # TODO: Create synthetic datasets with larger graphs
