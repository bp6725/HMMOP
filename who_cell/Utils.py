from who_cell.simulation.meta_network_simulator import MetaNetworkSimulator

import sys
sys.path.insert(0,'C:\Repos\pomegranate')
import pomegranate
from pomegranate import HiddenMarkovModel ,State
from pomegranate.distributions import IndependentComponentsDistribution
from pomegranate.distributions import NormalDistribution,DiscreteDistribution


import itertools
from tqdm import tqdm

import numpy as np
import networkx as nx



class PomegranateUtils() :

    @staticmethod
    def _return_relevant_multi_distribution(state_vactor, prob_of_dim, n_dim_of_chain, n_of_chains,
                                           dist_option="normal"):
        multi_hot_vector_state = np.zeros((n_of_chains * n_dim_of_chain, 1))
        multi_hot_vector_state[list(state_vactor)] = 1

        if dist_option == "normal":
            list_of_normal_dist = [NormalDistribution(dim[0], 0.1) for dim in multi_hot_vector_state]

        if dist_option == "discrete":
            list_of_normal_dist = [DiscreteDistribution({dim[0]: prob_of_dim, (1 - dim[0]): (1 - prob_of_dim)}) for dim
                                   in multi_hot_vector_state]
        return IndependentComponentsDistribution(list_of_normal_dist)

    @staticmethod
    def make_unique(states_set):
        return list([frozenset(_s) for _s in set([tuple(v) for v in states_set])])

    @staticmethod
    def build_networkx_graph(states, outer_network_dict, all_edges_in_nodes=False, undirected=False,
                             return_mapping=False):
        if return_mapping:
            state_name_to_state_mapping = {}

        if not undirected:
            G = nx.DiGraph()
        else:
            G = nx.Graph()
        for state in states:
            state_name = str(sorted(tuple(state)))
            G.add_node(state_name)

            if return_mapping:
                state_name_to_state_mapping[state_name] = state

        for fstate, tstste in itertools.product(states, states):
            if fstate not in outer_network_dict.keys():
                print("?")
                continue

            tstste_name = str(sorted(tuple(tstste)))
            fstate_name = str(sorted(tuple(fstate)))

            if tstste in outer_network_dict[fstate]:
                if all_edges_in_nodes:
                    if tstste_name not in G.nodes():
                        G.add_node(tstste_name)
                    if fstate_name not in G.nodes():
                        G.add_node(fstate_name)

                    if return_mapping:
                        state_name_to_state_mapping[tstste_name] = tstste
                        state_name_to_state_mapping[fstate_name] = fstate

                G.add_edge(fstate_name, tstste_name)

        if return_mapping:
            return G, state_name_to_state_mapping
        return G

    @staticmethod
    def _build_long_state_vector(set_of_states, n_dim_of_chain):
        def build_long_state(small_state, i):
            return [dim + i * n_dim_of_chain for dim in small_state]

        state_vector = [build_long_state(small_state, i) for small_state, i in
                        zip(set_of_states, range(len(set_of_states)))]
        flatten = [item for sublist in state_vector for item in sublist]
        return frozenset(flatten)

    @staticmethod
    def set_to_multihot_vector(set_seq,n_dim_of_chain,n_of_chains):
        def return_multi_hot(state_set, n_dim_of_chain, n_of_chains):
            multi_hot_vector_state = np.zeros((n_of_chains * n_dim_of_chain, 1))
            multi_hot_vector_state[list(state_set)] = 1
            return multi_hot_vector_state.T[0]

        return np.array([return_multi_hot(vector, n_dim_of_chain, n_of_chains) for vector in set_seq])