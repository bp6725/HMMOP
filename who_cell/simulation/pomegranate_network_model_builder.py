from who_cell.simulation.meta_network_simulator import MetaNetworkSimulator

import sys
sys.path.insert(0,'C:\Repos\pomegranate')
import pomegranate
from pomegranate import HiddenMarkovModel ,State
from pomegranate.distributions import IndependentComponentsDistribution
from pomegranate.distributions import NormalDistribution,DiscreteDistribution

from who_cell.Utils import PomegranateUtils


import itertools
from tqdm import tqdm

import numpy as np
import networkx as nx

class PomegranateNetworkModelBuilder() :


    def __init__(self):
        pass

    def build_model(self,meta_state_comb_to_walks_comb,args,start_states = None,end_states = None):
        # region args
        n_dim_of_chain = args["n_dim_of_chain"]
        n_of_chains = args["n_of_chains"]
        prob_of_dim = args["prob_of_dim"]
        # endregion

        first = True

        with tqdm(len(meta_state_comb_to_walks_comb)) as pbar:
            state_holder = {}

            markov_model = HiddenMarkovModel('markov_model')

            for sample in meta_state_comb_to_walks_comb.items():
                # curr_state = PomegranateUtils._build_long_state_vector(sample[0], n_dim_of_chain)
                curr_state = sample[0]

                if curr_state not in state_holder.keys():
                    curr_pomp_state = self._return_relevant_state(curr_state, prob_of_dim, n_dim_of_chain, n_of_chains)
                    markov_model.add_states(curr_pomp_state)
                    state_holder[curr_state] = curr_pomp_state
                else:
                    curr_pomp_state = state_holder[curr_state]

                for _next in sample[1]:
                    next_possible = _next
                    if next_possible not in state_holder.keys():
                        next_pomp_state = self._return_relevant_state(next_possible, prob_of_dim, n_dim_of_chain, n_of_chains)
                        markov_model.add_states(next_pomp_state)
                        state_holder[next_possible] = next_pomp_state
                    else:
                        next_pomp_state = state_holder[next_possible]

                    if first:
                        if start_states is None :
                            markov_model.add_transition(markov_model.start, curr_pomp_state, 0.1)
                        else :
                            for ss in start_states :
                                if ss not in state_holder.keys():
                                    s_pomp_state = self._return_relevant_state(ss, prob_of_dim,
                                                                                  n_dim_of_chain, n_of_chains)
                                    markov_model.add_states(s_pomp_state)
                                    state_holder[ss] = s_pomp_state

                                markov_model.add_transition(markov_model.start, state_holder[ss], 0.1)

                        first = False
                    markov_model.add_transition(curr_pomp_state, next_pomp_state, 0.1)
                pbar.update(1)

            if end_states is None :
                markov_model.add_transition(next_pomp_state, markov_model.end, 0.1)
            else :
                for es in end_states :
                    if es not in state_holder.keys():
                        e_pomp_state = self._return_relevant_state(es, prob_of_dim,
                                                                   n_dim_of_chain, n_of_chains)
                        markov_model.add_states(e_pomp_state)
                        state_holder[es] = e_pomp_state
                    markov_model.add_transition(state_holder[es], markov_model.end, 0.1)

        markov_model.bake()

        return markov_model

    def build_silent_model(self,global_G,DAG_G,state_name_to_state_mapping,args,with_silent_mode = True):

        #region mappings

        state_to_states_mapping = {
        (tuple(state_name_to_state_mapping[node]),): [(state_name_to_state_mapping[n],) for n in global_G[node]] for
        node in
        global_G.nodes()}
        state_name_to_state_tuple_mapping = {state_name: (tuple(state),) for state_name, state in
                                             state_name_to_state_mapping.items()}
        state_name_to_state_froz_tuple_mapping = {state_name: (state,) for state_name, state in
                                                  state_name_to_state_mapping.items()}

        nodes_tuple_in_DAG_G = [state_name_to_state_tuple_mapping[node] for node in DAG_G.nodes]
        nodes_froz_tuple_in_DAG_G = [state_name_to_state_froz_tuple_mapping[node] for node in DAG_G.nodes]
        edges_in_DAG_G = [(state_name_to_state_tuple_mapping[edge[0]], state_name_to_state_froz_tuple_mapping[edge[1]])
                          for edge in DAG_G.edges]

        #endregion

        #region args

        prob_of_dim = args["prob_of_dim"]
        n_dim_of_chain = args["n_dim_of_chain"]
        n_of_chains = args["n_of_chains"]

        #endregion

        first = True

        with tqdm(len(state_to_states_mapping)) as pbar:
            state_holder = {}
            silent_state_holder = {}

            markov_model = HiddenMarkovModel('profile_hmm')
            for sample in state_to_states_mapping.items():
                curr_state = PomegranateUtils._build_long_state_vector(sample[0], n_dim_of_chain)

                if curr_state not in state_holder.keys():
                    curr_pomp_state = self._return_relevant_state(curr_state, prob_of_dim, n_dim_of_chain, n_of_chains)
                    curr_silent_pomp_state = self._return_relevant_state(curr_state, prob_of_dim, n_dim_of_chain, n_of_chains,True)

                    markov_model.add_states(curr_pomp_state)
                    state_holder[curr_state] = curr_pomp_state

                    # only add states that exists in DAG_G
                    if (sample[0] in nodes_tuple_in_DAG_G) and with_silent_mode:
                        markov_model.add_states(curr_silent_pomp_state)
                        silent_state_holder[curr_state] = curr_silent_pomp_state
                else:
                    curr_pomp_state = state_holder[curr_state]
                    if (sample[0] in nodes_tuple_in_DAG_G) and with_silent_mode:
                        curr_silent_pomp_state = silent_state_holder[curr_state]

                for _next in sample[1]:
                    next_possible = PomegranateUtils._build_long_state_vector(_next, n_dim_of_chain)

                    if next_possible not in state_holder.keys():
                        next_pomp_state = self._return_relevant_state(next_possible, prob_of_dim, n_dim_of_chain, n_of_chains)
                        markov_model.add_states(next_pomp_state)
                        state_holder[next_possible] = next_pomp_state

                        if (_next in nodes_froz_tuple_in_DAG_G) and with_silent_mode:
                            next_silent_pomp_state = self._return_relevant_state(next_possible, prob_of_dim, n_dim_of_chain,
                                                                           n_of_chains, True)
                            markov_model.add_states(next_silent_pomp_state)
                            silent_state_holder[next_possible] = next_silent_pomp_state
                    else:
                        next_pomp_state = state_holder[next_possible]
                        if (_next in nodes_froz_tuple_in_DAG_G) and with_silent_mode:
                            next_silent_pomp_state = silent_state_holder[next_possible]

                    if first:
                        _first_transition_group_name = f"start->{next_pomp_state.name}"
                        markov_model.add_transition(markov_model.start, curr_pomp_state, probability=0.5,
                                                    group=_first_transition_group_name)
                        if (sample[0] in nodes_tuple_in_DAG_G) and with_silent_mode:
                            markov_model.add_transition(markov_model.start, curr_silent_pomp_state, probability=0.5,
                                                        group=_first_transition_group_name)
                        first = False

                    _transition_group_name = f"{curr_pomp_state.name}->{next_pomp_state.name}"
                    markov_model.add_transition(curr_pomp_state, next_pomp_state, probability=0.1,
                                                group=_transition_group_name)

                    if ((sample[0], _next) in edges_in_DAG_G) and with_silent_mode:
                        pass
                        markov_model.add_transition(curr_pomp_state, next_silent_pomp_state, probability=0.1,
                                                    group=_transition_group_name)
                        markov_model.add_transition(curr_silent_pomp_state, next_pomp_state, probability=0.1,
                                                    group=_transition_group_name)
                        markov_model.add_transition(curr_silent_pomp_state, next_silent_pomp_state, probability=0.1,
                                                    group=_transition_group_name)

                pbar.update(1)
            markov_model.add_transition(next_pomp_state, markov_model.end, probability=0.1)

        markov_model.bake(verbose=True, merge="Partial")

        return markov_model

    #region utils

    def clean_pomegranate_networkx_graph(self,G,return_pandas_adjacency = False):
        _G_dict_of_lists = nx.to_dict_of_lists(G)
        _G_dict_of_lists_clean = {k.name: [_v.name for _v in v] for k, v in _G_dict_of_lists.items()}

        _G = nx.from_dict_of_lists(_G_dict_of_lists_clean)

        _pandas_adjacency = None
        if return_pandas_adjacency :
            _pandas_adjacency = nx.to_pandas_adjacency(_G)
        return  _G,_pandas_adjacency


    #endregion

    #region private methods

    def _create_state_comb_to_walks_comb_dict(self,_state_comb_to_walks_comb,n_dim_of_chain):
        state_comb_to_walks_comb_dict = {}

        for sample in _state_comb_to_walks_comb:
            curr_state = PomegranateUtils._build_long_state_vector(sample[0], n_dim_of_chain)
            next_possible = [PomegranateUtils._build_long_state_vector(_next, n_dim_of_chain) for _next in sample[1]]
            state_comb_to_walks_comb_dict[curr_state] = next_possible

        return state_comb_to_walks_comb_dict

    def _return_relevant_state(self,state_vector, prob_of_dim, n_dim_of_chain, n_of_chains,is_silent = False):
        if not is_silent:
            d= PomegranateUtils._return_relevant_multi_distribution(state_vector, prob_of_dim, n_dim_of_chain, n_of_chains)
            state_name = str(sorted(state_vector))
        else:
            d = None
            state_name = "s_" + str(sorted(state_vector))

        return State(d, state_name)

    #endregion

if __name__ == '__main__':
    args = {"n_dim_of_chain": 3, "n_of_chains": 2, "possible_number_of_walks": [1, 2],
            "number_of_possible_states_limit": 10000000, "chance_to_be_in_path": 1.1, "prob_of_dim": 0.7}

    mns = MetaNetworkSimulator()
    pmb = PomegranateNetworkModelBuilder()

    meta_network = mns.build_meta_network(args)
    meta_network_1,meta_network_2 = itertools.tee(meta_network)

    pomp_markov_model = pmb.build_model(meta_network_1,args)

    nx_G,_ = pmb.clean_pomegranate_networkx_graph(pomp_markov_model.graph, False)

    res = list(nx.all_simple_paths(nx_G, 'markov_model-start', 'markov_model-end'))

    print("finish")