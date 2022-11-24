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

from karateclub import FeatherGraph, IGE, LDP, GL2Vec, Graph2Vec, NetLSD, SF, FGSD
from karateclub import WaveletCharacteristic as WvChr, GeoScattering as GS

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


def get_embeddings(graphs, approx_graphs, model):
    model.fit(graphs)
    embeddings = model.get_embedding()

    model.fit(approx_graphs)
    approx_embeddings = model.get_embedding()

    return (embeddings, approx_embeddings)


if __name__ == '__main__':
    nx_graphs, gx_graphs = obtain_graphset()
    gx_approx_graphs = approximate_gx_graphs(gx_graphs, 2, 0.25)

    nx_approx_graphs = []

    for graph in gx_approx_graphs:
        nx_approx_graphs.append(graphrox_to_networkx(graph))

    fthr_emb, fthr_approx_emb = get_embeddings(nx_graphs, nx_approx_graphs, FeatherGraph())
    ldb_emb, ldb_approx_emb = get_embeddings(nx_graphs, nx_approx_graphs, LDP())
    wc_emb, wc_approx_emb = get_embeddings(nx_graphs, nx_approx_graphs, WvChr())
    gs_emb, gs_approx_emb = get_embeddings(nx_graphs, nx_approx_graphs, GS())
    gl2v_emb, gl2v_approx_emb = get_embeddings(nx_graphs, nx_approx_graphs, GL2Vec())
    g2vec_emb, g2vec_approx_emb = get_embeddings(nx_graphs, nx_approx_graphs, Graph2Vec())
    nlsd_emb, nlsd_approx_emb = get_embeddings(nx_graphs, nx_approx_graphs, NetLSD())
    sf_emb, sf_approx_emb = get_embeddings(nx_graphs, nx_approx_graphs, SF())
    fgsd_emb, fgsd_approx_emb = get_embeddings(nx_graphs, nx_approx_graphs, FGSD())
    
    ige_emb, ige_approx_emb = get_embeddings(nx_graphs[:249], nx_approx_graphs[:249], IGE())


    # TODO: Synthetic datasets with larger graphs
