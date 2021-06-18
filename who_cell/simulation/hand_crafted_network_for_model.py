import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pprint
import pandas as pd
import networkx as nx
import inspect
import pandas as pd

import sys
sys.path.append("C:\Repos\WhoCell")

from who_cell.simulation.pomegranate_network_model_builder import PomegranateNetworkModelBuilder
from who_cell.Utils import PomegranateUtils
from who_cell.simulation.small_network import SmallNetwork
from who_cell.models.silent_model import SilentModel


class HandCraftedNetworkForModel() :
    def __init__(self):
        self.all_binarys = []

    def build_network(self,possible_transitions):
        all_states = set(itertools.chain(*possible_transitions))
        n_dim_of_chain = int(np.log2(len(all_states))) +1
        self._generate_all_binarys(n_dim_of_chain,[None]*n_dim_of_chain,0)

        bynaris = [b for b in self.all_binarys if len(b) > 0 ]
        states_to_binarys_map = {state:frozenset(binary) for state,binary in zip(all_states,bynaris)  }


        state_to_walk = {}
        for _s,_t in possible_transitions :
            s_state = states_to_binarys_map[_s]
            t_state = states_to_binarys_map[_t]

            if s_state in state_to_walk.keys() :
                state_to_walk[s_state].append(t_state)
            else :
                state_to_walk[s_state] = [t_state]

        return state_to_walk,states_to_binarys_map



    def simulate_partial_walks(self,real_trajectories,number_of_time_points,number_of_observations):
        real_trajs_observations = []
        for real_trajectory in real_trajectories :
            real_traj_observation = []

            for i in range(number_of_observations) :
                full_sampled_walk = [state.distribution.sample() for state in real_trajectory]
                partial_walk = self.return_partial_seq(full_sampled_walk,number_of_time_points)
                real_traj_observation.append(partial_walk)

            real_trajs_observations.append(np.array(real_traj_observation))

        return real_trajs_observations

    def _generate_all_binarys(self, n, arr, i):
        if i == n:
            self.all_binarys.append(np.nonzero(arr.copy())[0].tolist())
            return

        arr[i] = 0
        self._generate_all_binarys(n, arr, i + 1)

        arr[i] = 1
        self._generate_all_binarys(n, arr, i + 1)

    def return_partial_seq(self,seq,n_to_take):
        partial_walk_idx = random.sample(range(len(seq)), k=n_to_take)
        partial_walk = [seq[i] for i in sorted(partial_walk_idx)]
        return partial_walk

    def fit_few_obs_model(self, picked_real_trajctory, state_to_walk, states_to_binarys_map, start_states, end_states):
        pmb = PomegranateNetworkModelBuilder()
        model = pmb.build_model(state_to_walk, args, start_states=[states_to_binarys_map[s] for s in start_states],
                                end_states=[states_to_binarys_map[s] for s in end_states])

        # build trans matrix df
        # pre_trained_dense_matrix = model.dense_transition_matrix()
        # nos = [n.name for n in model.graph.nodes]
        # pre_trained_model_df = pd.DataFrame(index = nos,columns = nos,data = pre_trained_dense_matrix)

        # #build real trans matrix df
        # real_walk_matrix_df = pd.DataFrame(index=nos,columns=nos,data=0)
        # for d,t in zip(real_walks[picked_real_trajctory_idx],real_walks[picked_real_trajctory_idx][1:]) :
        #     real_walk_matrix_df.loc[(str(sorted(states_to_binarys_map[d])),str(sorted(states_to_binarys_map[t])))] = 1

        model.fit(picked_real_trajctory, algorithm="viterbi_fews_obs")
        return model

    def evel_test(self,model,test_set,pomstates_name_to_state_map,is_few_obs = True):
        print("-------------------------start test-------------------------")

        best_paths = []
        for test_sample in test_set:
            if is_few_obs :
                best_path, best_log_pos, best_idx_comb_in_path = model._few_observations_viterbi(test_sample)
                if np.isinf(best_log_pos):
                    pass
                    # print(test_sample)
                    # print("stop")

                best_paths.append(
                    ([pomstates_name_to_state_map[model.states[s].name] for s in best_path], best_log_pos))
            else :
                best_log_pos, best_path = model.viterbi(test_sample)
                if np.isinf(best_log_pos):
                    pass
                    # print(test_sample)
                    # print("stop")
                if best_path is None :
                    best_path = []

                best_paths.append(
                    ([pomstates_name_to_state_map[path[1].name] for path in best_path], best_log_pos))

        return best_paths

    def measure_results(self,best_paths,real_walk,results_per_size,all_best_paths):
        not_in_place_predictions = []
        missing_predictions = []
        non_convarge = 0

        for path, pos in best_paths:
            if np.isinf(pos):
                non_convarge += 1
                continue

            _not_in_place_predictions = set(path).difference(set(real_walk))
            _not_in_place_predictions -= set(['start', 'end'])

            if len(_not_in_place_predictions) > 0:
                not_in_place_predictions.append(_not_in_place_predictions)
            else:
                not_in_place_predictions.append(set([]))

            _missing_predictions = set(real_walk).difference(set(path))
            if len(_missing_predictions) > 0:
                missing_predictions.append(_missing_predictions)
            else:
                missing_predictions.append(set([]))

        all_best_paths.append(best_paths)
        pprint.pprint(best_paths)
        print(real_walk)

        results_per_size.append(("not_in_place", training_size, np.mean([len(it) for it in not_in_place_predictions])))
        results_per_size.append(("missing", training_size, np.mean([len(it) for it in missing_predictions])))
        results_per_size.append(("non_convarge", training_size, non_convarge))

        return results_per_size,all_best_paths

    def create_and_fit_silent_model(self,state_to_walk,picked_real_trajctory, states_to_binarys_map, start_states, end_states,args):
        kwargs = {
            "n_dim_of_chain": args["n_dim_of_chain"],
            "n_of_chains": args["n_of_chains"],
            "possible_number_of_walks": [1, 2, 3],
            "number_of_possible_states_limit": 2000,
            "chance_to_be_in_path": 1,
            "prob_of_dim":args["prob_of_dim"],
            "max_number_of_trans_in_network": 8,
            "size_of_pomegranate_network": 1000,
            "n_nodes_to_start_with": 5,
            "n_walks_per_node_start": 1,
            "welk_length": 5,
            "n_pn_dynamic_net": 30,
            "n_obs_dynamic_net": 4,
            "p_obs_dynamic_net": 0.3}

        small_network_builder = SmallNetwork(**kwargs)
        silent_model_builder = SilentModel()

        hstates = set(itertools.chain(*state_to_walk.values())).union(set(state_to_walk.keys()))
        hstate_comb_to_walks_comb_dict = state_to_walk
        full_h_G,state_name_to_state_mapping = PomegranateUtils.build_networkx_graph(hstates, hstate_comb_to_walks_comb_dict,
                                                         all_edges_in_nodes=True,return_mapping=True)

        pns_data_to_possible_and_connected_states_trans = small_network_builder.build_samples_to_possible_states_mapping(
            {k:v for k,v in zip(range(len(picked_real_trajctory)),picked_real_trajctory)}, hstates, full_h_G, kwargs)

        DAG_G, _ = silent_model_builder.return_all_possible_G_without_cycles(full_h_G,
                                                                             pns_data_to_possible_and_connected_states_trans[0],
                                                                             picked_real_trajctory)
        silent_model = silent_model_builder.build_silent_pomemodel(full_h_G, DAG_G, state_name_to_state_mapping, kwargs)

        silent_model.fit(picked_real_trajctory)

        return silent_model

if __name__ == '__main__':
    transitions = [(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(1,3),(1,5),(1,8),(3,4),(3,8),(3,6),(4,3),(6,8),(6,9),(7,5),(8,10)]
    start_states = [1]
    end_states = [9,10]
    transitions = list(set(transitions))
    real_walks = [[1,2,3,6,9],
                  [1,2,3,8,10],
                  [1,3,8,10]]

    hcns = HandCraftedNetworkForModel()
    state_to_walk, states_to_binarys_map = hcns.build_network(transitions)
    pomstates_name_to_state_map = {str(sorted(v)): k for k, v in states_to_binarys_map.items()}
    pomstates_name_to_state_map_silent = {f"s_{k}":v for k,v in pomstates_name_to_state_map.items()}
    pomstates_name_to_state_map = {**pomstates_name_to_state_map,**pomstates_name_to_state_map_silent}
    pomstates_name_to_state_map['markov_model-start'] = 'start'
    pomstates_name_to_state_map['markov_model-end'] = 'end'
    pomstates_name_to_state_map['profile_hmm-start'] = 'start'
    pomstates_name_to_state_map['profile_hmm-end'] = 'end'

    pprint.pprint(pomstates_name_to_state_map)

    args = {}
    args["n_dim_of_chain"] = max(list(itertools.chain(*[state for _, state in states_to_binarys_map.items()]))) + 1
    args["n_of_chains"] = 1
    args["prob_of_dim"] = 0.9

    pmb = PomegranateNetworkModelBuilder()

    real_walks_states = [
        [pmb._return_relevant_state(states_to_binarys_map[obs], args["prob_of_dim"], args["n_dim_of_chain"], 1) for obs
         in real_walk] for real_walk in real_walks]

    max_training_size = 50
    max_observations_per_real_traj = hcns.simulate_partial_walks(real_walks_states, 3, max_training_size)
    test_sets = hcns.simulate_partial_walks(real_walks_states, 3, 10)
    picked_real_trajctory_idx = 0
    all_best_paths = [];all_best_paths_silent = []
    results_per_size = [];results_per_size_silent = []
    for training_size in range(5,max_training_size,3) :
        for picked_real_trajctory_idx in [0,1,2] :
            picked_real_trajctory = hcns.return_partial_seq(max_observations_per_real_traj[picked_real_trajctory_idx], training_size)
            # _fe_obs_model = hcns.fit_few_obs_model(picked_real_trajctory, state_to_walk, states_to_binarys_map, start_states, end_states)
            _silent_model = hcns.create_and_fit_silent_model(state_to_walk,picked_real_trajctory, states_to_binarys_map, start_states, end_states,args)

            test_set = test_sets[picked_real_trajctory_idx]
            # best_paths =hcns.evel_test(_fe_obs_model,test_set,pomstates_name_to_state_map)
            best_paths_silent = hcns.evel_test(_silent_model, test_set,pomstates_name_to_state_map,is_few_obs = False)

            # # build trans matrix df
            # post_trained_dense_matrix = model.dense_transition_matrix()
            # nos = [n.name for n in model.graph.nodes]
            # post_trained_model_df = pd.DataFrame(index=nos, columns=nos, data=post_trained_dense_matrix)


            #measures :

            # results_per_size,all_best_paths = hcns.measure_results(best_paths,real_walks[picked_real_trajctory_idx],results_per_size,all_best_paths)
            results_per_size_silent, all_best_paths_silent = hcns.measure_results(best_paths_silent, real_walks[picked_real_trajctory_idx],
                                                                    results_per_size_silent, all_best_paths_silent)

    # results_df = pd.DataFrame(columns = ["error type","training size","value"],data =results_per_size )
    # agg_result_df = results_df.groupby(["error type","training size"]).mean().reset_index().set_index("training size")
    # plt.plot(agg_result_df[agg_result_df["error type"] == "not_in_place"]["value"],label="not in place")
    # plt.plot(agg_result_df[agg_result_df["error type"] == "missing"]["value"],label="missing")
    # plt.plot(agg_result_df[agg_result_df["error type"] == "non_convarge"]["value"],label="non convarge")
    # plt.legend(loc="upper left")
    # plt.show()

    results_df_silent = pd.DataFrame(columns=["error type", "training size", "value"], data=results_per_size_silent)
    agg_result_df_silent = results_df_silent.groupby(["error type", "training size"]).mean().reset_index().set_index("training size")
    plt.plot(agg_result_df_silent[agg_result_df_silent["error type"] == "not_in_place"]["value"], label="not in place")
    plt.plot(agg_result_df_silent[agg_result_df_silent["error type"] == "missing"]["value"], label="missing")
    plt.plot(agg_result_df_silent[agg_result_df_silent["error type"] == "non_convarge"]["value"], label="non convarge")
    plt.legend(loc="upper left")
    plt.show()

    print("finish")
