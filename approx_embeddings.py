import json
import os
import sys
import time

import graphrox as gx
import networkx as nx
import pickle as pkl

from multiprocessing import Manager, Pool, Queue, Value

from karateclub import FeatherGraph, LDP, Graph2Vec
from karateclub.dataset import GraphSetReader


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
        

def generate_barabasi_albert_graphset(n, m, count):
    """Generates a set of `count` Barabasi-Albert graphs with the given `n` and `m`"""
    graphset = []

    for i in range(count):
        graphset.append(nx.barabasi_albert_graph(n, m, seed=int(time.time() + i)))

    return graphset

        
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


def process_embeddings(params, tasks_completed):
    print('Processing embeddings for task ' + str(params['id']) + '...')

    graphs = params['graphs']
    approx_graphs = params['approx_graphs']

    embeddings = []

    emb, approx_emb = get_embeddings(params['graphs'], params['approx_graphs'], FeatherGraph())
    embeddings.append({
        'name': 'FeatherGraph',
        'standard': emb,
        'approximate': approx_emb,
    })

    emb, approx_emb = get_embeddings(params['graphs'], params['approx_graphs'], LDP())
    embeddings.append({
        'name': 'LDP',
        'standard': emb,
        'approximate': approx_emb,
    })

    emb, approx_emb = get_embeddings(params['graphs'], params['approx_graphs'], Graph2Vec())
    embeddings.append({
        'name': 'Graph2Vec',
        'standard': emb,
        'approximate': approx_emb,
    })

    count = params['count']
    n = params['n']
    m = params['m']
    block_dimension = params['block_dimension']
    threshold = params['threshold']

    data = {
        'gen_n': n,
        'gen_m': m,
        'approx_block_dimension': block_dimension,
        'approx_threshold': threshold,
        'graph_count': count,
        'embeddings': embeddings,
    }

    filename = f'out/emb_c({count})_n({n})_m({m})_b({block_dimension})_t({threshold}).pkl'
    with open(filename, 'wb') as out_file:
        pkl.dump(data, out_file)

    with tasks_completed.get_lock():
        tasks_completed.value += 1
        completed = tasks_completed.value

    task_id = params['id']
    tasks_total = params['tasks_total']
        
    print(f'Completed task {task_id}. {completed}/{tasks_total} complete')


def prepare_graphs(params, emb_queue):
    print('Preparing for task ' + str(params['id']) + '...')

    nx_graphs = generate_barabasi_albert_graphset(params['n'], params['m'], params['count'])
    gx_graphs = []

    for graph in nx_graphs:
        gx_graphs.append(networkx_to_graphrox(graph))

    gx_approx_graphs = [
        g.approximate(params['block_dimension'], params['threshold']) for g in gx_graphs
    ]

    nx_approx_graphs = []

    for graph in gx_approx_graphs:
        nx_approx_graphs.append(graphrox_to_networkx(graph))

    print('Finished preparation for task ' + str(params['id']) + '.')

    params['graphs'] = nx_graphs
    params['approx_graphs'] = nx_approx_graphs
    emb_queue.put(params)


def dispatch_worker(prep_queue, emb_queue, tasks_completed):
    while True:
        if not emb_queue.empty():
            task = emb_queue.get()
            process_embeddings(task, tasks_completed)
        elif not prep_queue.empty():
            task = prep_queue.get()
            prepare_graphs(task, emb_queue)
        else:
            time.sleep(0.25)
    

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

    print('There are a total of ' + str(len(tasks)) + ' tasks.')
    print()

    if not os.path.exists('out'):
        os.makedirs('out')

    prep_queue = Queue()
    emb_queue = Queue()

    tasks_completed = Value('i', 0)

    for i, task in enumerate(conf['tasks']):
        task_id = i + 1

        are_task_params_valid = True

        if 'n' not in task:
            are_task_params_valid = False

        if 'm' not in task:
            are_task_params_valid = False

        if 'block_dimension' not in task:
            are_task_params_valid = False

        if 'threshold' not in task:
            are_task_params_valid = False

        if 'count' not in task:
            are_task_params_valid = False

        if not are_task_params_valid:
            print('Error: Configuration for task ' + str(task_id) + ' is invalid')
            continue

        task['id'] = task_id
        task['tasks_total'] = len(tasks)

        prep_queue.put(task)

    worker_pool = Pool(os.cpu_count(), dispatch_worker, (prep_queue, emb_queue, tasks_completed,))

    while True:
        if tasks_completed.value == len(tasks):
            sys.exit(0)
        else:
            time.sleep(1)

        

        
