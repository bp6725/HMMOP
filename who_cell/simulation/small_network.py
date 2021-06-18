import itertools
from functools import reduce
import matplotlib.pyplot as plt
from IPython.display import display, HTML,clear_output
import holoviews as hv

from tqdm import tqdm
import random
import networkx as nx
import pandas as pd
import numpy as np

from who_cell.simulation.pomegranate_network_model_builder import PomegranateNetworkModelBuilder
from who_cell.simulation.meta_network_simulator import MetaNetworkSimulator
from who_cell.simulation.data_samples_simulator import DataSamplesSimulator
from who_cell.Utils import PomegranateUtils


class SmallNetwork():

    def __init__(self,n_dim_of_chain ,n_of_chains ,possible_number_of_walks ,number_of_possible_states_limit ,chance_to_be_in_path ,prob_of_dim ,max_number_of_trans_in_network
                     ,size_of_pomegranate_network,n_nodes_to_start_with,n_walks_per_node_start,welk_length,n_pn_dynamic_net,n_obs_dynamic_net,p_obs_dynamic_net):
        self.n_dim_of_chain = n_dim_of_chain
        self.n_of_chains = n_of_chains
        self.possible_number_of_walks = possible_number_of_walks
        self.number_of_possible_states_limit = number_of_possible_states_limit
        self.chance_to_be_in_path = chance_to_be_in_path
        self.prob_of_dim = prob_of_dim
        self.max_number_of_trans_in_network = max_number_of_trans_in_network

        self.size_of_pomegranate_network = size_of_pomegranate_network
        self.n_nodes_to_start_with = n_nodes_to_start_with
        self.n_walks_per_node_start = n_walks_per_node_start
        self.welk_length = welk_length

        self.n_pn_dynamic_net = n_pn_dynamic_net
        self.n_obs_dynamic_net = n_obs_dynamic_net
        self.p_obs_dynamic_net = p_obs_dynamic_net  # chance to be part of the walk if you are in extrernal state

        self.pome_model_builder = PomegranateNetworkModelBuilder()
        self.meta_sim = MetaNetworkSimulator()
        self.samples_sim = DataSamplesSimulator()

    def build_small_network_from_meta_network(self):
        args = {"n_dim_of_chain" : self.n_dim_of_chain,
        "n_of_chains" : self.n_of_chains,
        "possible_number_of_walks" : self.possible_number_of_walks,
        "number_of_possible_states_limit" : self.number_of_possible_states_limit,
        "chance_to_be_in_path" : self.chance_to_be_in_path,
        "prob_of_dim" : self.prob_of_dim}

        meta_state_comb_to_walks_comb = self.meta_sim.build_meta_network(args)
        state_comb_to_walks_comb_dict = self._pick_sub_network(meta_state_comb_to_walks_comb)
        _G, hstate_comb_to_walks_comb_dict, hstates, full_h_G, state_name_to_state_mapping = self._pick_connected_graph_from_meta_network(state_comb_to_walks_comb_dict)
        pn_samples = self._sample_dynamic_network(hstate_comb_to_walks_comb_dict,state_comb_to_walks_comb_dict)


        return pn_samples,hstates,full_h_G,state_name_to_state_mapping,hstate_comb_to_walks_comb_dict

    def build_samples_to_possible_states_mapping(self,pn_samples,hstates,full_h_G,network_proprties = {"prob_of_dim" : 0.7, "n_dim_of_chain" : 10, "n_of_chains" :1 }):
        states_max_to_pick = 3
        sample_to_states_options = self.build_observations_to_states_mapping(pn_samples, hstates,network_proprties,
                                                                        states_max_to_pick=states_max_to_pick)

        pns_data_to_possible_and_connected_states_trans = {}
        for pn, pn_data in pn_samples.items():
            possible_states = [sample_to_states_options[tuple(d)] for d in pn_data]
            # keep only states with path between them
            possible_and_connected_states = list(
                filter(lambda seq: self.is_connected_seq(seq, full_h_G), itertools.product(*possible_states)))
            pns_data_to_possible_and_connected_states_trans[pn] = possible_and_connected_states

        return pns_data_to_possible_and_connected_states_trans

    def build_pn_specific_network(self,pn_data_to_possible_and_connected_states_trans,state_name_to_state_mapping,
                                  hstate_comb_to_walks_comb_dict,full_h_G,propagation_factor = 2,verbose = False):
        # first we find the shortest path between each two observation and take all states in the path
        _all_states_of_pn = []
        all_walks_of_pn = []
        for trans in pn_data_to_possible_and_connected_states_trans:
            walk = list(map(lambda x: state_name_to_state_mapping[x], itertools.chain(
                *[nx.shortest_path(full_h_G, str(sorted(tuple(ss))), str(sorted(tuple(fs)))) for ss, fs in
                  zip(trans, trans[1:])])))
            all_walks_of_pn.append(walk)
            _all_states_of_pn += walk

        all_states_of_pn = PomegranateUtils.make_unique(_all_states_of_pn)

        # now we propagate around each state
        _all_propagated_states_of_pn = [
            self.propagate_around_state([state], propagation_factor, hstate_comb_to_walks_comb_dict) for state in
            all_states_of_pn]
        all_propagated_states_of_pn = list(itertools.chain(*_all_propagated_states_of_pn))

        __G = PomegranateUtils.build_networkx_graph(all_propagated_states_of_pn, hstate_comb_to_walks_comb_dict,
                                        all_edges_in_nodes=True)
        if verbose :
            pass

            # hv.Graph.from_networkx(__G, nx.layout.spring_layout) \
            #     .options(cmap='Category20', edge_cmap='viridis', directed=True, arrowhead_length=0.03)

        return __G,all_walks_of_pn

    #region private method

    def _pick_sub_network(self,meta_state_comb_to_walks_comb):
        state_comb_to_walks_comb, _state_comb_to_walks_comb = itertools.tee(
            itertools.islice(meta_state_comb_to_walks_comb, self.size_of_pomegranate_network))

        state_comb_to_walks_comb_dict = {}

        with tqdm(self.size_of_pomegranate_network) as pbar:
            for sample in _state_comb_to_walks_comb:
                curr_state = PomegranateUtils._build_long_state_vector(sample[0],self.n_dim_of_chain)

                next_possible = [PomegranateUtils._build_long_state_vector(_next, self.n_dim_of_chain) for _next in sample[1]]

                state_comb_to_walks_comb_dict[curr_state] = next_possible
                pbar.update(1)
        return state_comb_to_walks_comb_dict

    def _pick_connected_graph_from_meta_network(self,state_comb_to_walks_comb_dict,verbose = False):
        # we starts with building networkx graph of the meta graph - for now undirected
        G, state_name_to_state_mapping = PomegranateUtils.build_networkx_graph(state_comb_to_walks_comb_dict.keys(),
                                                              state_comb_to_walks_comb_dict, undirected=True,
                                                              return_mapping=True)

        # we want to remove isolated nodes :
        G.remove_nodes_from(list(nx.isolates(G)))

        # we pick random number of nodes to start with
        idx_to_hstates = random.sample(range(1, len(state_comb_to_walks_comb_dict)), self.n_nodes_to_start_with)
        nodes_to_start_with = [params[1] for params in enumerate(G.nodes()) if params[0] in idx_to_hstates]

        # now we walk on G - states names
        walks_on_G = self.nx_random_walks(G, nodes_to_start_with, self.n_walks_per_node_start, self.welk_length,
                                     state_name_to_state_mapping)

        # build hstate_comb_to_walks_comb_dict from walks
        _unconnected_hstate_comb_to_walks_comb_dict = self.build_hstate_comb_to_walks_comb_dict_from_walks(walks_on_G)

        # remove_empty_states(no transitions)
        unconnected_hstate_comb_to_walks_comb_dict = dict(
            filter(lambda kv: len(kv[1]) > 0, _unconnected_hstate_comb_to_walks_comb_dict.items()))
        unconnected_hstates = [hstate for hstate in unconnected_hstate_comb_to_walks_comb_dict.keys()]


        biggest_cc = None
        biggest_cc_length = 0
        _G = PomegranateUtils.build_networkx_graph(unconnected_hstates, unconnected_hstate_comb_to_walks_comb_dict)
        for sg in [_G.subgraph(c) for c in nx.weakly_connected_component_subgraphs(_G)]:
            _size = len(sg.nodes())
            if _size > biggest_cc_length:
                biggest_cc = sg
                biggest_cc_length = _size

        # keep only nodes from connected component
        nodes_in_biggest_cc = [node for node in biggest_cc.nodes()]
        _hstate_comb_to_walks_comb_dict = {k: v for k, v in unconnected_hstate_comb_to_walks_comb_dict.items() if
                                           str(sorted(tuple(k))) in nodes_in_biggest_cc}

        # keep only tranisions inside hstate
        hstate_comb_to_walks_comb_dict = {_k: [_vv for _vv in _v if _vv in _hstate_comb_to_walks_comb_dict.keys()] for
                                          _k, _v in _hstate_comb_to_walks_comb_dict.items()}

        hstates = [hstate for hstate in hstate_comb_to_walks_comb_dict.keys()]

        full_h_G = PomegranateUtils.build_networkx_graph(hstates, hstate_comb_to_walks_comb_dict, all_edges_in_nodes=True)

        if verbose:
            _d = list(map(lambda x: len(x), unconnected_hstate_comb_to_walks_comb_dict.values()))
            plt.hist(_d)

            hv.Graph.from_networkx(_G, nx.layout.spring_layout) \
                .options(cmap='Category20', edge_cmap='viridis', directed=True, arrowhead_length=0.005)

            hv.Graph.from_networkx(full_h_G, nx.layout.spring_layout) \
                .options(cmap='Category20', edge_cmap='viridis', directed=True, arrowhead_length=0.05)

        return _G,hstate_comb_to_walks_comb_dict,hstates,full_h_G,state_name_to_state_mapping

    def _sample_dynamic_network(self,hstate_comb_to_walks_comb_dict,state_comb_to_walks_comb_dict):
        '''
        how do we pick dynamic network :
            1) we use random walk on the real network
            2) chance to "observe" step in the walk  : p_obs ,we stop when we have n samples
            3) we take every chosen step and "blur' him , so some of them wont be in the network..
        '''

        _pn_to_samples = {}
        _pn_to_samples_for_valid = {}

        for pn in range(self.n_pn_dynamic_net):
            _samples_of_pn = []
            _samples_of_pn_for_valid = {}
            _n_obs = 0

            _, _trans = random.choice(list(hstate_comb_to_walks_comb_dict.items()))
            while (_n_obs < self.n_obs_dynamic_net):
                if len(_trans) == 0:
                    _, _trans = random.choice(list(hstate_comb_to_walks_comb_dict.items()))
                    continue
                _state = random.choice(_trans)

                # if we pick "lonely" state  wich we filter before, for now we pick new one
                if _state not in state_comb_to_walks_comb_dict.keys():
                    _, _trans = random.choice(list(hstate_comb_to_walks_comb_dict.items()))
                    continue

                _trans = state_comb_to_walks_comb_dict[_state]

                if (_state in hstate_comb_to_walks_comb_dict.keys()) or (random.random() < self.p_obs_dynamic_net):
                    _samples_of_pn.append(_state)
                    _n_obs += 1

                _samples_of_pn_for_valid[_state] = _trans

            _pn_to_samples[pn] = _samples_of_pn
            _pn_to_samples_for_valid = _samples_of_pn_for_valid
            pn_samples = {pn: [PomegranateUtils._return_relevant_multi_distribution(st, self.prob_of_dim, self.n_dim_of_chain,
                               self.n_of_chains,dist_option="discrete").sample() for st in _pn_to_samples[pn]] for pn in _pn_to_samples}

        return pn_samples

    def nx_random_walks(self,G, nodes_to_start_with, n_walks_per_node_start=2, num_steps=30,
                        state_name_to_state_mapping=None):
        node_to_walks = {}
        for i in nodes_to_start_with:
            walks = []
            for walk in range(n_walks_per_node_start):
                if state_name_to_state_mapping is None:
                    curr_walk = [i]
                else:
                    curr_walk = [state_name_to_state_mapping[i]]
                curr = i
                for step in range(num_steps):
                    neighbors = list(G.neighbors(curr))
                    if len(neighbors) > 0:
                        curr = random.choice(neighbors)
                    if state_name_to_state_mapping is None:
                        curr_walk.append(curr)
                    else:
                        curr_walk.append(state_name_to_state_mapping[curr])
                walks.append(curr_walk)

            if state_name_to_state_mapping is None:
                node_to_walks[i] = walks
            else:
                node_to_walks[state_name_to_state_mapping[i]] = walks
        return node_to_walks

    def build_hstate_comb_to_walks_comb_dict_from_walks(self,walks_on_G):
        all_transitions = []
        for start_node, walks in walks_on_G.items():
            for walk in walks:
                all_transitions.append(tuple([start_node, walk[0]]))
                _walk_trans = list(map(lambda xx: tuple([xx[0], xx[1]]), zip(walk, walk[1:])))
                all_transitions += _walk_trans

        _unconnected_hstate_comb_to_walks_comb_dict = {}
        for s_st, t_st in all_transitions:
            if s_st not in _unconnected_hstate_comb_to_walks_comb_dict.keys():
                _unconnected_hstate_comb_to_walks_comb_dict[s_st] = []

            if t_st not in _unconnected_hstate_comb_to_walks_comb_dict[s_st]:
                _unconnected_hstate_comb_to_walks_comb_dict[s_st].append(t_st)
        return _unconnected_hstate_comb_to_walks_comb_dict

    def propagate_around_state(self,states, prop_factor, outer_network, depth=0):
        if prop_factor == depth:
            return PomegranateUtils.make_unique(states)

        _new_states = states.copy()
        #     print(f"_new_states : {_new_states}")
        for state in states:
            #         print(f"state : {state}")
            if state not in outer_network.keys():
                print(state)
                print("----")
                print(outer_network.keys())
                raise Exception()
                continue
            _trans = outer_network[state]
            _new_states += _trans

        return self.propagate_around_state(_new_states, prop_factor, outer_network, depth + 1)

    def build_observations_to_states_mapping(self,pn_samples, hstates,network_proprties, threshold_prob=0.001, states_max_to_pick=5):
        '''
        mapping observations to possible states,given threshold_prob and states_max_to_pick
        :param pn_samples: if dict - we flatten the dict to a long list.
        :param hstates: the hmm states
        :param network_proprties:
        :param threshold_prob:
        :param states_max_to_pick:
        :return:
        '''
        if type(pn_samples) is dict :
            if isinstance([v for v in pn_samples.values()][0],np.ndarray) :
                all_samples = list(itertools.chain(*pn_samples.values()))
            else :
                all_samples = reduce(lambda x, y: x + y, pn_samples.values())
        else :
            all_samples = pn_samples

        all_samples = list(set([tuple(v) for v in all_samples]))

        sample_to_state_prob = {}
        for sample in all_samples:
            if sample in sample_to_state_prob.keys():
                continue

            state_to_prob = {}
            for state in hstates:
                state_to_prob[state] = self.prob_sample_to_state(sample, state,network_proprties)
            sample_to_state_prob[sample] = state_to_prob

        sample_to_state_prob_df = pd.DataFrame(sample_to_state_prob)
        sample_to_state_prob_df = sample_to_state_prob_df[sample_to_state_prob_df > threshold_prob]

        sample_to_states_options = {}
        for column in sample_to_state_prob_df.columns:
            sample_to_states_options[column] = sample_to_state_prob_df.nlargest(states_max_to_pick,
                                                                                column).index.tolist()

        return sample_to_states_options

    def prob_sample_to_state(self,sample, state,network_proprties):
        dist_state = PomegranateUtils._return_relevant_multi_distribution(state, network_proprties["prob_of_dim"], network_proprties["n_dim_of_chain"], network_proprties["n_of_chains"])
        return np.exp2(dist_state.log_probability([sample]))

    def is_connected_seq(self,seq, full_h_G):
        return all(list(map(lambda xy: nx.has_path(full_h_G, str(sorted(tuple(xy[0]))), str(sorted(tuple(xy[1])))),
                            zip(seq, seq[1:]))))

    #endregion

if __name__ == '__main__':
    kwargs = {
        "n_dim_of_chain": 5,
        "n_of_chains": 1,
        "possible_number_of_walks": [1, 2, 3],
        "number_of_possible_states_limit": 2000,
        "chance_to_be_in_path": 0.7,
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
        hstate_comb_to_walks_comb_dict, full_h_G, propagation_factor=2, verbose=True)

    for pn, pn_samples in pn_samples.items():
        sample_to_states_options = small_network_builder.build_observations_to_states_mapping(pn_samples, hstates,
                                                                                              kwargs)
