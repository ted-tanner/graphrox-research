import json
import os
import sys
import time

import graphrox as gx
import networkx as nx
import pickle as pkl

from multiprocessing import Manager, Process, Queue

from karateclub import FeatherGraph, LDP, Graph2Vec
from karateclub.dataset import GraphSetReader


def generate_graphset_portion(work_queue, nx_graphset, n, m):
    while not work_queue.empty():
        count = work_queue.get()
        
        for i in range(count):
            graph = nx.barabasi_albert_graph(n, m, seed=int(time.time() + i))

            nx_graphset.append(graph)
        

def generate_barabasi_albert_graphset(n, m, count):
    """Generates a set of `count` Barabasi-Albert graphs with the given `n` and `m`"""
    work_queue = Queue()
    manager = Manager()
    nx_graphset = manager.list()

    extra = count - (int(count / os.cpu_count()) * os.cpu_count())

    for i in range(os.cpu_count()):
        workload = count if i != 0 else count + extra
        work_queue.put(workload)

    processes = []

    for i in range(os.cpu_count()):
        process = Process(target=generate_graphset_portion, args=(work_queue,
                                                                  nx_graphset,
                                                                  n,
                                                                  m))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    return nx_graphset


def get_embeddings(graphs, approx_graphs, model):
    model.fit(graphs)

    before = time.perf_counter()
    embeddings = model.get_embedding()
    embeddings = {
        'execution_time': time.perf_counter() - before,
        'embeddings': embeddings
    }

    model.fit(approx_graphs)

    before = time.perf_counter()
    approx_embeddings = model.get_embedding()
    approx_embeddings = {
        'execution_time': time.perf_counter() - before,
        'embeddings': approx_embeddings,
    }

    return (embeddings, approx_embeddings)


def process_embeddings(work_queue, embeddings, graphs, approx_graphs):
    while not work_queue.empty():
        job = work_queue.get()

        emb, approx_emb = get_embeddings(graphs, approx_graphs, job['model'])
        embeddings.append({
            'name': job['name'],
            'standard': emb,
            'approximate': approx_emb,
        })

        print('Generated ' + job['name'] + ' embeddings')

        
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
        nx_graph.add_node(gx_graph.vertex_count() - 1)

    # karateclub's model.fit() requires all nodes to be present
    for i in range(gx_graph.vertex_count()):
        nx_graph.add_node(i)

    return nx_graph


def approximate_gx_graphs(gx_graphs, block_dimension, threshold):
    """Returns duplicated and approximated versions of the gx_graphs"""
    return [g.duplicate().approximate(block_dimension, threshold) for g in gx_graphs]


if __name__ == '__main__':
    try:
        with open('emb_config.json', 'r') as conf_file:
            conf = json.loads(conf_file.read())
    except FileNotFoundError:
        print('Error: emb_config.json was not found in the current directory')
        sys.exit(1)
    except json.JSONDecodeError:
        print('Error: emb_config.json contained invalid JSON')
        sys.exit(1)

    try:
        tasks = conf['tasks']
    except KeyError:
        print('Error: Tasks not found emb_config.json')
        sys.exit(1)

    if not isinstance(tasks, list):
        print('Error: Tasks list not found in emb_config.json ')
        sys.exit(1)

    for i, task in enumerate(conf['tasks']):
        task_id = i + 1
        try: 
            n = task['n']
            m = task['m']
            block_dimension = task['block_dimension']
            threshold = task['threshold']
            count = task['count']
        except KeyError:
            print('Error: Configuration for task ' + str(task_id) + ' is invalid')
            sys.exit(1)

        if i != 0:
            print('----------------------------------------------------------------')
            print()

        print('Beginning task ' + str(task_id))
        print()

        print('Generating graphs...')

        nx_graphs = generate_barabasi_albert_graphset(n, m, count)
        gx_graphs = []

        for graph in nx_graphs:
            gx_graphs.append(networkx_to_graphrox(graph))

        print('Approximating graphs...')
        gx_approx_graphs = approximate_gx_graphs(gx_graphs, block_dimension, threshold)

        nx_approx_graphs = []

        for graph in gx_approx_graphs:
            nx_approx_graphs.append(graphrox_to_networkx(graph))

        work_queue = Queue()
        manager = Manager()
        embeddings = manager.list()

        work_queue.put({'name': 'FeatherGraph', 'model': FeatherGraph()})
        work_queue.put({'name': 'LDP', 'model': LDP()})
        work_queue.put({'name': 'Graph2Vec', 'model': Graph2Vec()})

        print('Generating embeddings...')
        print()

        processes = []

        for _ in range(os.cpu_count()):
            if not work_queue.empty():
                process = Process(target=process_embeddings, args=(work_queue,
                                                                   embeddings,
                                                                   nx_graphs,
                                                                   nx_approx_graphs))
                process.start()
                processes.append(process)

        for process in processes:
            process.join()

        data = {
            'gen_n': n,
            'gen_m': m,
            'approx_block_dimension': block_dimension,
            'approx_threshold': threshold,
            'graph_count': count,
            'embeddings': embeddings,
        }

        filename = f'emb_c({count})_n({n})_m({m})_b({block_dimension})_t({threshold}).pkl'
        with open(filename, 'wb') as out_file:
            pkl.dump(data, out_file)

        print()
        print('Completed task ' + str(task_id))
        print()
