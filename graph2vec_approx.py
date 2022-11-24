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

from karateclub import FeatherGraph
from karateclub.dataset import GraphSetReader


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


if __name__ == '__main__':
    nx_graphs, gx_graphs = obtain_graphset()
    gx_approx_graphs = approximate_gx_graphs(gx_graphs, 2, 0.25)

    nx_approx_graphs = []

    for graph in gx_approx_graphs:
        nx_approx_graphs.append(graphrox_to_networkx(graph))

    model = FeatherGraph()
    model.fit(nx_graphs)
    embeddings = model.get_embedding()

    model = FeatherGraph()
    model.fit(nx_approx_graphs)
    approx_embeddings = model.get_embedding()

    print(embeddings)
    print()
    print()
    print(approx_embeddings)

    # TODO: Synthetic datasets with larger graphs
