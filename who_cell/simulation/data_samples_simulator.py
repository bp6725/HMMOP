from who_cell.simulation.meta_network_simulator import MetaNetworkSimulator
from who_cell.simulation.pomegranate_network_model_builder import PomegranateNetworkModelBuilder
from who_cell.Utils import PomegranateUtils

import copy
import random
import numpy as np
import itertools



class DataSamplesSimulator() :

    def __init__(self):
        pass

    def generate_samples(self,state_comb_to_walks_comb,size_of_possible_rw,number_of_seqs,samples_from_walk,args):
        # region args
        n_dim_of_chain = args["n_dim_of_chain"]
        n_of_chains = args["n_of_chains"]
        possible_number_of_walks = args["possible_number_of_walks"]
        number_of_possible_states_limit = args["number_of_possible_states_limit"]
        chance_to_be_in_path = args["chance_to_be_in_path"]
        prob_of_dim = args["prob_of_dim"]
        # endregion

        state_comb_to_walks_comb_dict = self._create_state_comb_to_walks_comb_dict(state_comb_to_walks_comb,
                                                                                   n_dim_of_chain)

        all_possible_states = list(state_comb_to_walks_comb_dict.keys())
        n_of_states_in_meta_network = len(all_possible_states)

        seqs = []
        for i in range(number_of_seqs):
            seq = []
            random_state_idx = random.randint(1, n_of_states_in_meta_network)
            curr_random_state = all_possible_states[random_state_idx - 1]
            seq.append(curr_random_state)

            for j in range(size_of_possible_rw):
                possible_next_steps = copy.copy(state_comb_to_walks_comb_dict[curr_random_state])
                curr_random_state = self._pick_random_next_stage(possible_next_steps, state_comb_to_walks_comb_dict,all_possible_states)

                if curr_random_state is None:
                    break
                    print("dude")
                    random_state_idx = random.randint(1, n_of_states_in_meta_network)
                    curr_random_state = all_possible_states[random_state_idx - 1]

                seq.append(curr_random_state)

            seqs.append(seq)

        sampled_seqs = [self._return_multi_hot_vectors(random.choices(s,k=samples_from_walk), n_dim_of_chain, n_of_chains) for s in seqs]
        # sampled_seqs = [return_multi_hot_vectors(s, n_dim_of_chain, n_of_chains) for s in seqs]

        return sampled_seqs

    #region private methods

    def _create_state_comb_to_walks_comb_dict(self, _state_comb_to_walks_comb, n_dim_of_chain):
        state_comb_to_walks_comb_dict = {}

        for sample in _state_comb_to_walks_comb:
            curr_state = PomegranateUtils._build_long_state_vector(sample[0], n_dim_of_chain)
            next_possible = [PomegranateUtils._build_long_state_vector(_next, n_dim_of_chain) for _next in sample[1]]
            state_comb_to_walks_comb_dict[curr_state] = next_possible

        return state_comb_to_walks_comb_dict

    def _pick_random_next_stage(self,_possible_next_steps, state_comb_to_walks_comb_dict,all_possible_states, counter=0):
        if counter == 50:
            return None
        if len(_possible_next_steps) == 0:
            return None

        first_pick = random.choice(_possible_next_steps)
        if first_pick in all_possible_states:
            return first_pick
        _possible_next_steps.remove(first_pick)
        counter = counter + 1
        return self._pick_random_next_stage(_possible_next_steps, state_comb_to_walks_comb_dict,all_possible_states, counter)

    def _return_multi_hot(self,state_set, n_dim_of_chain, n_of_chains):
        multi_hot_vector_state = np.zeros((n_of_chains * n_dim_of_chain, 1))
        multi_hot_vector_state[list(state_set)] = 1
        return multi_hot_vector_state.T[0]

    def _return_multi_hot_vectors(self,vectors, n_dim_of_chain, n_of_chains):
        return np.array([self._return_multi_hot(vector, n_dim_of_chain, n_of_chains) for vector in vectors])

    #endregion


if __name__ == '__main__':
    args = {"n_dim_of_chain": 3, "n_of_chains": 2, "possible_number_of_walks": [1, 2],
            "number_of_possible_states_limit": 10000000, "chance_to_be_in_path": 1.1, "prob_of_dim": 0.7}

    mns = MetaNetworkSimulator()
    pmb = PomegranateNetworkModelBuilder()
    dss = DataSamplesSimulator()

    meta_network = mns.build_meta_network(args)
    meta_network_1,meta_network_2 = itertools.tee(meta_network)

    pomp_markov_model = pmb.build_model(meta_network_1,args["number_of_possible_states_limit"] ,args)
    data_samples = dss.generate_samples(meta_network_2,50,20000,40,args)

    print("finish")