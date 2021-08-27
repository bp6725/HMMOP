from who_cell.models.gibbs_sampler import GibbsSampler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import itertools
from functools import reduce
from pos_data_builder import PosDataBuilder

class PosCodeBase() :
    def __init__(self):
        pass

    #region Public

    @staticmethod
    def run_pos_experiment(test_set_words, start_probs, emms_probs, number_of_iters,
                           N, description, known_transitions, title, is_only_observed,
                           comper_transitions, comper_transitions_title, state_order_for_plot):
        gs = GibbsSampler(2, 5, transition_sampling_profile="observed" if is_only_observed else 'all')
        sampled_transitions, ws, transitions, states_picked_by_w = gs.sample_known_emissions(test_set_words,
                                                                                             start_probs,
                                                                                             emms_probs,
                                                                                             number_of_iters, N=N)

        print(description)
        print(f"the format is \"known / {comper_transitions_title}/{title} \") ")

        if comper_transitions is None:
            PosCodeBase.compare_transition(known_transitions, transitions[-1])
        else:
            PosCodeBase.compare_transition(known_transitions, comper_transitions, transitions[-1])

        PosCodeBase.plot_w_dist(ws, test_set_words)

        PosCodeBase.plot_states_transitions_as_lines(known_transitions, transitions[-1], state_order_for_plot, f"{title}")

        res = {"sampled_transitions": sampled_transitions,
               "ws": ws,
               "transitions": transitions,
               "states_picked_by_w": states_picked_by_w}
        return res

    @staticmethod
    def extrect_observed_transitions(all_ws_knownN,_states_picked_by_w):
        return  [PosCodeBase._extrect_observed_transitions(ws_list,_states_picked) for
                 ws_list,_states_picked in zip(all_ws_knownN,_states_picked_by_w)]

    #endregion

    #region Plots
    @staticmethod
    def plot_results_convergence(transitions_results_list,experiments_name_list,
                                 transitions_probs, start_probs, emms_probs):
        original_model = PosDataBuilder.build_pome_for_pos_exp(transitions_probs, start_probs, emms_probs)
        transitions_probs = None
        N = None

        #region lambda functions
        __l1_distance = lambda dist, state, known: abs(dist[state] - known) if state in dist.keys() else known
        __cross_entropy_distance = lambda dist, state, known: -1 * known * np.log(
            dist[state]) if state in dist.keys() else (-1 * known * np.log(0.0001))

        _l1_distance = lambda known_dist, comp_dist: sum(
            ([__l1_distance(comp_dist, state, prob) for state, prob in known_dist.items()]))
        _cross_entropy_distance = lambda known_dist, comp_dist: sum(
            [__cross_entropy_distance(comp_dist, state, prob) for state, prob in known_dist.items()])

        l1_distance = lambda known_trns, comp_trans: np.mean(
            [_l1_distance(dist, comp_trans[state]) for state, dist in known_trns.items()])
        cross_entropy_distance = lambda known_trns, comp_trans: np.mean(
            [_cross_entropy_distance(dist, comp_trans[state]) for state, dist in known_trns.items()])

        #endregion

        #region KL distance setup

        _trajectory_prob = lambda traj, model_trans: reduce(lambda x, y: x * y,
                                                            [model_trans[_f][_t] for _f, _t in zip(traj, traj[1:])])


        sampled_trajs = original_model.sample(n=300, length=N, path=True)
        sampled_trajs_states = [[obs.name for obs in traj[1] if 'start' not in obs.name] for traj in sampled_trajs]
        known_probs = [_trajectory_prob(traj, transitions_probs) for traj in sampled_trajs_states]

        #endregion

        #region build measures curves
        all_kl_results = {}
        all_l1_results = {}
        all_ce_results = {}
        for _trans_list, exp_name in zip(transitions_results_list, experiments_name_list):
            kl_dist = [PosCodeBase._kl_distance_transitions(_t, sampled_trajs_states, known_probs) for _t in _trans_list]
            l1_dist = [l1_distance(transitions_probs, _t) for _t in _trans_list]
            ce_dist = [cross_entropy_distance(transitions_probs, _t) for _t in _trans_list]

            all_kl_results[exp_name] = kl_dist
            all_l1_results[exp_name] = l1_dist
            all_ce_results[exp_name] = ce_dist

        #endregion

        #region plot measure curves

        for all_results, name in zip([all_kl_results, all_l1_results, all_ce_results], ['kl', 'crossEntropy', 'l1']):
            fig, sub = plt.subplots(1, 1, figsize=(8, 8))
            model_results_df = pd.DataFrame(all_results)

            sns.lineplot(data=model_results_df, ax=sub, legend='full', dashes=False)
            sub.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            sub.set_title(f"{name}")
            sub.set_xlabel("iter")
            sub.set_ylabel(f"{name}")

            plt.subplots_adjust(hspace=0.8)
            plt.subplots_adjust(wspace=0.8)

            plt.show()

        #endregion

    @staticmethod
    def plow_w_hist(_states_picked_by_w,few_obs_test_set_tags, test_set_tags):

        few_seq_of_tags = [[seq[i] for i in ws] for ws, seq in zip(few_obs_test_set_tags, test_set_tags)]
        tags_from_sampler = _states_picked_by_w
        tags_random_allocation = [[seq[i] for i in np.random.randint(0, len(seq), size=len(ws))] for ws, seq in
                                  zip(few_obs_test_set_tags, test_set_tags)]

        distance_func = lambda a, b: sum(map(lambda x: int(x[0] != x[1]), zip(a, b))) / len(a)

        random_dists = [distance_func(_rand, known) for _rand, known in zip(tags_random_allocation, few_seq_of_tags)]
        sapmled_dists = [distance_func(_sampled, known) for _sampled, known in zip(tags_from_sampler, few_seq_of_tags)]

        plt.hist(random_dists, bins=50)
        plt.hist(sapmled_dists, bins=50)
        plt.title("Normalized Hamming distance per sentence")
        plt.legend(["distance from random allocation", "distance from W"])
        plt.show()


    @staticmethod
    def plot_w_dist(all_ws, test_set_words,N = 30):
        samples_to_take = np.random.randint(0, len(all_ws), 100)
        max_sentence = max(map(len, [test_set_words[i] for i in samples_to_take]))
        max_sentence = max_sentence if max_sentence > N else N

        w_matrix = np.ones((max_sentence, len(samples_to_take))) * -1

        for i, i_word in enumerate(samples_to_take):
            word = test_set_words[i_word]
            w_matrix[0:len(word), i] = 0
            w_matrix[all_ws[-1][i_word], i] = 1

        sns.heatmap(w_matrix)
        plt.show()

    @staticmethod
    def plot_states_transitions_as_lines(known, _compr, state_order_for_plot, title):
        fig, subs = plt.subplots(2, 1)
        _compr_df = pd.DataFrame(_compr)
        known_df = pd.DataFrame(known)

        _compr_df = _compr_df[state_order_for_plot].loc[state_order_for_plot]
        known_df = known_df[state_order_for_plot].loc[state_order_for_plot]

        known_df.plot(figsize=(8, 8), ax=subs[0])
        _compr_df.plot(figsize=(8, 8), ax=subs[1])

        subs[0].set_title("known")
        subs[1].set_title(title)
        plt.show()

    @staticmethod
    def compare_transition(first_df, second_df, third_df=None):
        if type(first_df) is dict:
            first_df = pd.DataFrame(first_df)
        if type(second_df) is dict:
            second_df = pd.DataFrame(second_df)

        if third_df is not None:
            if type(third_df) is dict:
                third_df = pd.DataFrame(third_df)

        second_df = second_df[first_df.columns]
        second_df = second_df.loc[first_df.index]

        if third_df is not None:
            third_df = third_df[first_df.columns]
            third_df = third_df.loc[first_df.index]

        if third_df is not None:
            return first_df.round(decimals=3).astype(str) + ' / ' + second_df.round(decimals=3).astype(
                str) + ' / ' + third_df.round(decimals=3).astype(str)

        comps_plot = first_df.round(decimals=3).astype(str) + ' / ' + second_df.round(decimals=3).astype(str)
        return comps_plot

    @staticmethod
    def plot_compersion(first_df_stack, second_df_stack,
                        first_title='first', second_title='second',
                        third_df_stack=None, third_title='third'):
        if type(first_df_stack) is dict:
            first_df_stack = pd.DataFrame(first_df_stack).stack()
        if type(second_df_stack) is dict:
            second_df_stack = pd.DataFrame(second_df_stack).stack()

        if third_df_stack is not None:
            if type(third_df_stack) is dict:
                third_df_stack = pd.DataFrame(third_df_stack).stack()

        if third_df_stack is None:
            cobined_stack_df = pd.concat([first_df_stack, second_df_stack], axis=1).fillna(0)
            cobined_stack_df = cobined_stack_df.rename(columns={0: first_title, 1: second_title})

            plt.scatter(cobined_stack_df[first_title], cobined_stack_df[second_title])
            plt.xlabel(first_title)
            plt.ylabel(f"{second_title}")
        else:
            cobined_stack_df = pd.concat([first_df_stack, second_df_stack, third_df_stack], axis=1).fillna(0)
            cobined_stack_df = cobined_stack_df.rename(columns={0: first_title, 1: second_title, 2: third_title})

            plt.scatter(cobined_stack_df[first_title], cobined_stack_df[second_title])
            plt.scatter(cobined_stack_df[first_title], cobined_stack_df[third_title])
            plt.legend([second_title, third_title])
            plt.xlabel(first_title)
            plt.ylabel(f"{second_title}/{third_title}")
        plt.show()

    #endregion

    #region Private

    @staticmethod
    def _extrect_observed_transitions(ws, _states_picked):
        real_walks = []
        for w, states_walk in zip(ws, _states_picked):
            real_transitions = [(states_walk[_f], states_walk[_t]) for _f, _t in zip(range(len(w)), range(1, len(w))) if
                                (w[_t] - w[_f]) == 1]
            real_walks.append(real_transitions)

        real_transition_dict = {}
        for (_from, _to), count in Counter(itertools.chain(*real_walks)).items():
            if _from not in real_transition_dict.keys():
                real_transition_dict[_from] = {_to: count}
            else:
                real_transition_dict[_from][_to] = count

        real_transition_dict_normalized = {_from: {__from: _count / sum(to.values()) for __from, _count in to.items()}
                                           for _from, to in real_transition_dict.items()}
        return real_transition_dict_normalized

    @staticmethod
    def _kl_distance_transitions(_trajectory_prob,comp_transitions, sampled_trajs_states, known_probs):
        comp_probs = [_trajectory_prob(traj, comp_transitions) for traj in sampled_trajs_states]
        return sum([np.log(known / comp) * known for known, comp in zip(known_probs, comp_probs)])

    #endregion