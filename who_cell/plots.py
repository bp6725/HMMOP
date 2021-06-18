import sys
sys.path.insert(0,'C:\Repos\pomegranate')
from who_cell.simulation.meta_network_simulator import MetaNetworkSimulator
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import random
import functools
from functools import reduce
import copy

import networkx as nx
import holoviews as hv
from scipy.stats import spearmanr

import pomegranate
from pomegranate import HiddenMarkovModel ,State
from pomegranate.distributions import IndependentComponentsDistribution
from pomegranate.distributions import NormalDistribution,DiscreteDistribution

from who_cell.Utils import PomegranateUtils
from IPython.display import display, HTML,clear_output


class Plots() :

    @staticmethod
    def plot_markov_model_graph(markov_model):
        not_silent = []
        new_G = nx.DiGraph()
        for node in markov_model.graph.nodes:
            new_G.add_node(node.name, is_silent=str("s_" in node.name),
                           leading_to=str([e.name for e in markov_model.graph[node]]))
            if not ("s_" in node.name):
                not_silent.append(node.name)

        for edge in markov_model.graph.edges:
            new_G.add_edge(edge[0].name, edge[1].name,
                           is_silent=(str(("s_" in edge[0].name) or ("s_" in edge[1].name))))

        display(hv.Graph.from_networkx(new_G, nx.layout.bipartite_layout(new_G, nodes=not_silent)) \
                .options(cmap=['red', 'blue'], color_index="is_silent", edge_cmap='viridis', directed=True,
                         arrowhead_length=0.01))

        silent_G = nx.DiGraph()
        for node in markov_model.graph.nodes:
            if ("s_" in node.name):
                silent_G.add_node(node.name, is_silent=str("s_" in node.name),
                                  leading_to=str([e.name for e in markov_model.graph[node]]))

        for edge in markov_model.graph.edges:
            if ("s_" in edge[0].name) and ("s_" in edge[1].name):
                silent_G.add_edge(edge[0].name, edge[1].name)

        display(hv.Graph.from_networkx(silent_G, nx.layout.random_layout) \
                .options(directed=True, arrowhead_length=0.02, cmap=['red']))

        not_silent_G = nx.DiGraph()
        for node in markov_model.graph.nodes:
            if ("s_" not in node.name):
                not_silent_G.add_node(node.name, is_silent=str("s_" in node.name),
                                      leading_to=str([e.name for e in markov_model.graph[node]]))

        for edge in markov_model.graph.edges:
            if ("s_" not in edge[0].name) and ("s_" not in edge[1].name):
                not_silent_G.add_edge(edge[0].name, edge[1].name)

        display(hv.Graph.from_networkx(not_silent_G, nx.layout.random_layout) \
                .options(directed=True, arrowhead_length=0.02, cmap=['blue']))
