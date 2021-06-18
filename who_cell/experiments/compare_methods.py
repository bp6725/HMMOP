import sys
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import random
import functools
from functools import reduce
import copy

import pomegranate
from pomegranate import HiddenMarkovModel ,State
from pomegranate.distributions import IndependentComponentsDistribution
from pomegranate.distributions import NormalDistribution,DiscreteDistribution

import networkx as nx

import pickle as pkl
import matplotlib.pyplot as plt
import IPython
import seaborn as sns
from bokeh.plotting import figure, from_networkx
from bokeh.io import show
import holoviews as hv

from scipy.stats import spearmanr

from who_cell.simulation.small_network import SmallNetwork
from who_cell.models.silent_model import SilentModel
from who_cell.simulation.pomegranate_network_model_builder import PomegranateNetworkModelBuilder



if __name__ == '__main__':

    ###build small network###

    kwargs = {
        "n_dim_of_chain": 5,
        "n_of_chains": 1,
        "possible_number_of_walks": [1, 2, 3],
        "number_of_possible_states_limit": 2000,
        "chance_to_be_in_path": 1,
        "prob_of_dim": 0.7,
        "max_number_of_trans_in_network": 8,
        "size_of_pomegranate_network": 1000,
        "n_nodes_to_start_with": 5,
        "n_walks_per_node_start": 1,
        "welk_length": 5,
        "n_pn_dynamic_net": 30,
        "n_obs_dynamic_net": 4,
        "p_obs_dynamic_net": 0.3}

    small_network_builder = SmallNetwork(**kwargs)

    pn_samples, hstates, full_h_G, state_name_to_state_mapping, hstate_comb_to_walks_comb_dict = small_network_builder.build_small_network_from_meta_network()
    pns_data_to_possible_and_connected_states_trans = small_network_builder.build_samples_to_possible_states_mapping(
        pn_samples, hstates, full_h_G, kwargs)

    pn_to_start_with = 12
    pn_data_to_possible_and_connected_states_trans = pns_data_to_possible_and_connected_states_trans[pn_to_start_with]
    global_G, all_walks_of_pn = small_network_builder.build_pn_specific_network(
        pn_data_to_possible_and_connected_states_trans, state_name_to_state_mapping,
        hstate_comb_to_walks_comb_dict, full_h_G, propagation_factor=2, verbose=False)

    # all_walks_of_pn = the walks we get from connecting the possible states.. not the original ones


    ###now we find all the relevant obs of the other pns###

    trainig_set_from_other_pns = []

    for pn, samples in pn_samples.items():
        if pn == pn_to_start_with:
            continue

        sample_to_states_options = small_network_builder.build_observations_to_states_mapping(samples, hstates, kwargs)
        relevant_obs = [k for k, v in sample_to_states_options.items() if len(v) > 0]
        if len(relevant_obs) > 1:
            trainig_set_from_other_pns.append(relevant_obs)


    ###build and fit different models###
    def return_multi_hot(state_set, n_dim_of_chain, n_of_chains):
        multi_hot_vector_state = np.zeros((n_of_chains * n_dim_of_chain, 1))
        multi_hot_vector_state[list(state_set)] = 1
        return multi_hot_vector_state.T[0]


    def return_multi_hot_vectors(vectors, n_dim_of_chain, n_of_chains):
        return np.array([return_multi_hot(vector, n_dim_of_chain, n_of_chains) for vector in vectors])


    # sampled_seqs = [return_multi_hot_vectors(random.choices(s,k=8),n_dim_of_chain,n_of_chains) for s in seqs]
    sampled_seqs = [return_multi_hot_vectors(s, n_dim_of_chain=5, n_of_chains=1) for s in all_walks_of_pn]


    # ###VT###
    # model_builder = PomegranateNetworkModelBuilder()
    # few_obs_model = model_builder.build_model(hstate_comb_to_walks_comb_dict, args=kwargs)
    #
    # fitted_few_model = few_obs_model.fit(trainig_set_from_other_pns, algorithm="viterbi_fews_obs", verbose=True)


    ###silent_model###
    silent_model_builder = SilentModel()
    DAG_G, _ = silent_model_builder.return_all_possible_G_without_cycles(global_G,
                                                                         pn_data_to_possible_and_connected_states_trans,
                                                                         all_walks_of_pn)
    silent_model = silent_model_builder.build_silent_pomemodel(global_G, DAG_G, state_name_to_state_mapping, kwargs)
    fitted_silent_model = silent_model.fit(trainig_set_from_other_pns)
