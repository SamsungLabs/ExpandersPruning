"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Author(s):
James Stewart (j2.stewart@samsung.com; james1995stewart@gmail.com)
Umberto Michieli (u.michieli@samsung.com)
Mete Ozay (m.ozay@samsung.com)

Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import os
import random
import math

from pathlib import Path

import networkx as nx
import numpy as np

from utils.config import GRAPH_SAVE_DIR

def get_neighbors(graph, vertices):
    """ Returns neighboring vertices of S in G. """

    neighbors = set()
    for vertex in vertices:
        neighbors.update(list(graph.neighbors(vertex)))

    return neighbors

def test_expansion(graph, n_0, n_1, samples=1000):
    """ Estimates expansion of G. """

    left, right = nx.bipartite.sets(graph,
                        top_nodes=[x for x in graph.nodes if graph.nodes[x]["bipartite"] == 0])
    expansion = 0

    for _ in range(samples):
        k_left = np.random.randint(1, math.ceil(n_0/2))
        k_right = np.random.randint(1, math.ceil(n_1/2))
        subset_left = random.sample(left, k_left)
        subset_right = random.sample(right, k_right)
        expansion += (len(get_neighbors(graph, subset_left))/k_left)+ \
                            (len(get_neighbors(graph, subset_right))/k_right)

    return expansion/(2*samples)

def balance_deg_seq(degs_0, n_1):
    """ Evently distributes degs_0 over n_1 vertices. """

    n_0 = len(degs_0)

    if n_0 == n_1:
        return degs_0

    degree_0 = sum(degs_0)
    degs_1 = []
    deg_per_vertex = degree_0 // n_1
    remainder = degree_0 % n_1

    for i in range(n_1):
        extra = 1 if (i < remainder) else 0
        degs_1.append(deg_per_vertex+extra)

    return degs_1

def random_graph(n_0, n_1, deg):
    """ Returns a random graph. """

    graph = nx.Graph()
    graph.add_nodes_from(list(range(n_0)), bipartite=0)
    graph.add_nodes_from([x+n_0 for x in range(n_1)], bipartite=1)

    for i in range(n_0):
        for j in range(n_1):
            if random.random() < deg/n_1:
                graph.add_edge(i, j+n_0)

    return graph

def xnet_graph(n_0, n_1, deg):
    """ Returns an XNet graph. """

    graph = nx.Graph()
    graph.add_nodes_from(list(range(n_0)), bipartite=0)
    graph.add_nodes_from([x+n_0 for x in range(n_1)], bipartite=1)

    for i in range(n_0):
        perm = np.random.permutation(n_1)
        for j in range(min(deg, n_1)):
            graph.add_edge(i, perm[j]+n_0)

    return graph

def rreg_graph(n_0, n_1, deg):
    """ Returns a random regular graph. """

    if n_0 > n_1:
        in_degrees = [deg]*n_0
        out_degrees = balance_deg_seq(in_degrees, n_1)
    else:
        out_degrees = [deg]*n_1
        in_degrees = balance_deg_seq(out_degrees, n_0)

    return nx.bipartite.configuration_model(in_degrees, out_degrees, create_using=nx.Graph())

def generate_random_graph(n_0, n_1, deg):
    """ Returns and saves a random graph.
        Returns cached graph if it exists. """

    save_dir = os.path.join(GRAPH_SAVE_DIR, "random")
    os.makedirs(save_dir, exist_ok=True)
    graph_filepath = Path(os.path.join(save_dir, f"n0={n_0}_n1={n_1}_deg={deg}"))

    if graph_filepath.is_file():
        return nx.read_gpickle(graph_filepath)

    graph = random_graph(n_0, n_1, deg)

    nx.write_gpickle(graph, graph_filepath)

    return graph

def generate_xnet_graph(n_0, n_1, deg):
    """ Returns and saves an XNet graph.
        Returns cached graph if it exists. """

    save_dir = os.path.join(GRAPH_SAVE_DIR, "xnet")
    os.makedirs(save_dir, exist_ok=True)
    graph_filepath = Path(os.path.join(save_dir, f"n0={n_0}_n1={n_1}_deg={deg}"))

    if graph_filepath.is_file():
        return nx.read_gpickle(graph_filepath)

    graph = xnet_graph(n_0, n_1, deg)

    nx.write_gpickle(graph, graph_filepath)

    return graph

def generate_rreg_graph(n_0, n_1, deg, n_graphs=100, n_samples=1000):
    """ Returns and saves an optimal random regular graph.
        Returns cached graph if it exists. """

    save_dir = os.path.join(GRAPH_SAVE_DIR, "rreg")
    os.makedirs(save_dir, exist_ok=True)
    graph_filepath = Path(os.path.join(save_dir, f"n0={n_0}_n1={n_1}_deg={deg}"))

    if graph_filepath.is_file():
        return nx.read_gpickle(graph_filepath)

    max_expansion = 0
    max_expander = nx.Graph()

    for _ in range(n_graphs):
        graph = rreg_graph(n_0, n_1, deg)
        expansion = test_expansion(graph, n_0, n_1, n_samples)

        if expansion > max_expansion:
            max_expansion = expansion
            max_expander = graph

    nx.write_gpickle(max_expander, graph_filepath)

    return max_expander
