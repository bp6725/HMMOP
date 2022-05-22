import numpy as np
from itertools import chain
import itertools
from who_cell.models.gibbs_sampler import GibbsSampler
from tqdm import tqdm
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution
from collections import Counter
import networkx as nx
from collections import Mapping
import copy
from collections import Mapping
from pomegranate.distributions import NormalDistribution,DiscreteDistribution
from who_cell.models.utils import Utils
from functools import reduce
import pandas as pd

class EmSequenceLabeling():
    def __init__(self):
        pass

    # @staticmethod
    # def sequence_labeling_known_emissions(all_relvent_observations, transitions_probs, start_probs,
    #                                       emissions_table, Ng_iters=10, N=2):
    #     transitions_probs = {k: v for k, v in transitions_probs.items() if not k in ['start', 'end']}
    #     N = EmSequenceLabeling.__rebuild_N(N, all_relvent_observations)
    #
    #     N = list(map(lambda x: int(x * 1.2), N))
    #
    #     gibbs_sampler = GibbsSampler(N, number_of_states_in_time=None,
    #                                  transition_sampling_profile='all', multi_process=False)
    #
    #     emissions_table = gibbs_sampler.impute_emissions_table_with_zeros(emissions_table, all_relvent_observations,
    #                                                                       True)
    #     curr_trans = transitions_probs
    #
    #     if type(N) is list:
    #         curr_w = [sorted(np.random.choice(range(max(_N, len(obs))), len(obs), replace=False)) for obs, _N in
    #                   zip(all_relvent_observations, N)]
    #     else:
    #         curr_w = [sorted(np.random.choice(range(max(N, len(obs))), len(obs), replace=False)) for obs in
    #                   all_relvent_observations]
    #
    #     optimal_states_seqs, _ = gibbs_sampler.sample_walk_from_params(all_relvent_observations, N,
    #                                                                    emissions_table, start_probs,
    #                                                                    curr_w, curr_trans)
    #     with tqdm(total=Ng_iters) as pbar:
    #         for i in range(Ng_iters):
    #             optimal_Ws = EmSequenceLabeling.calculate_optimal_Ws(optimal_states_seqs, all_relvent_observations,
    #                                                                  emissions_table)
    #             optimal_states_seqs = EmSequenceLabeling.calculate_optimal_states_seqs(optimal_Ws,
    #                                                                                    all_relvent_observations, N,
    #                                                                                    emissions_table,
    #                                                                                    transitions_probs, start_probs)
    #             pbar.update(1)
    #
    #     return [[optimal_states_seq[w] for w in optimal_W] for optimal_W, optimal_states_seq in
    #             zip(optimal_Ws, optimal_states_seqs)]

    @staticmethod
    def most_likely_path(observations, transitions_prob, start_probs,
                         emissions_table, Ng_iters, N,W = None):
        is_known_emissions = type(
            emissions_table[list(emissions_table.keys())[0]]) is dict

        if is_known_emissions :
            emissions_table = EmSequenceLabeling.impute_emissions_table_with_zeros(emissions_table,observations)

        pome_model = EmSequenceLabeling._build_pome_model_of_chain(transitions_prob, start_probs, emissions_table)[0]

        guess_state_per_obs = [
            [EmSequenceLabeling._closest_state(emissions_table, obs, is_known_emissions) for obs in sentence] for
            sentence in observations]

        n_steps_transitions = GibbsSampler.build_n_steps_transitions_dicts(transitions_prob, 10)

        all_ml_paths = []
        with tqdm(len(observations)) as p:
            for i,(sentence, guess,n) in enumerate(zip(observations, guess_state_per_obs,N)):
                w = None if (W is None) else W[i]
                w = list(range(n)) if len(sentence) == n else w
                mlp = EmSequenceLabeling._ml_path(sentence, n, pome_model,n_steps_transitions,
                                                  emissions_table, guess, Ng_iters,is_known_emissions,w)
                all_ml_paths.append(mlp)
                p.update(1)

        return all_ml_paths

    @staticmethod
    def impute_emissions_table_with_zeros( emissions_table,all_relvent_observations,impute_zeros = False) :
        all_possible_emissions = set(chain(*all_relvent_observations))
        new_emissions_table = emissions_table
        for _emm in all_possible_emissions :
            for k,v in emissions_table.items():
                if _emm not in v.keys() :
                    new_emissions_table[k][_emm] = 0 if not impute_zeros else 1e-6

        if impute_zeros :
            return {k:{kk:(vv if vv != 0 else 1e-9) for kk,vv in v.items()} for k,v in new_emissions_table.items()}

        return new_emissions_table

    @staticmethod
    def _build_pome_model_of_chain(transition_matrix_sparse, start_probs, state_to_distrbution_param_mapping) :
        is_known_emissions = type(
            state_to_distrbution_param_mapping[list(state_to_distrbution_param_mapping.keys())[0]]) is dict

        if is_known_emissions:
            all_params_to_distrbutions = {st: DiscreteDistribution(dist) for st, dist in
                                          state_to_distrbution_param_mapping.items()}
            state_to_distrbution_param_mapping['start'] = {}
        else:
            # we need this because we need to share the dist instances for pomegranate
            all_params_to_distrbutions = {(_params[0], _params[1]): NormalDistribution(_params[0], _params[1]) for
                                          k, _params in
                                          state_to_distrbution_param_mapping.items() if
                                          ((k != 'start') and (k != 'end'))}
            if 'start' not in state_to_distrbution_param_mapping.keys():
                state_to_distrbution_param_mapping['start'] = (-1, -1)

        all_model_pome_states = {}
        model = HiddenMarkovModel()
        for state_name, _params in state_to_distrbution_param_mapping.items():
            if state_name == 'start':
                model.add_state(model.start)
                all_model_pome_states['start'] = model.start
                continue
            if state_name == 'end':
                model.add_state(model.end)
                all_model_pome_states['end'] = model.end
                continue
            if is_known_emissions:
                state = State(all_params_to_distrbutions[state_name], name=f"{state_name}")
            else:
                state = State(all_params_to_distrbutions[_params], name=f"{state_name}")
            model.add_state(state)
            all_model_pome_states[state_name] = state

        for _from_state_name, _to_states in transition_matrix_sparse.items():
            if _from_state_name == "end" : continue
            _from_state = all_model_pome_states[_from_state_name]
            for _to_state_name, _trans_prob in _to_states.items():
                _to_state = all_model_pome_states[_to_state_name]
                model.add_transition(_from_state, _to_state, _trans_prob)
        for _to_state,_trans_prob in start_probs.items() :
            model.add_transition(model.start, all_model_pome_states[_to_state], _trans_prob)

        model.bake()

        return model, all_model_pome_states

    @staticmethod
    def _closest_state(emissions_table, obs,is_known_emissions):
        max_state = None
        max_prob = 0
        for s,params in emissions_table.items():
            if s in ["start","end"] : continue
            if is_known_emissions :
                prob = emissions_table[s][obs]
            else :
                prob = Utils.normpdf(obs,params[0],params[1])

            if prob > max_prob:
                max_prob = prob
                max_state = s
        return max_state

    @staticmethod
    def _ml_path(sentence, N, pome_model,n_steps_transitions, emissions_table, guess, Ng_iters,is_known_emissions,W = None):
        is_unknown_w = (W is None)
        if is_unknown_w :
            W = EmSequenceLabeling.init_guess_w_per_sentence( sentence,n_steps_transitions,guess,N)
            if len(W) != len(sentence) :
                # print((len(W) , len(sentence)) )
                # raise Exception()
                W = sorted(np.random.choice(range(max(N, len(sentence))), len(sentence), replace=False))

        # X = EmSequenceLabeling._return_initial_X(W, guess, N)

        for iter in range(Ng_iters):
            # if is_unknown_w:
            #     W = EmSequenceLabeling._return_optimal_W(X, sentence, emissions_table,is_known_emissions)
            if is_unknown_w :
                X = EmSequenceLabeling._return_optimal_X(sentence, W, pome_model, N)
                W = EmSequenceLabeling._return_optimal_W(X, sentence, emissions_table, is_known_emissions)
            else :
                X = EmSequenceLabeling._return_optimal_X(sentence, W, pome_model, N)
                break

        return [X[w] for w in W]

    @staticmethod
    def _return_optimal_X(sentence, W, pome_model, N):
        sentence_with_nulls = []
        j = 0
        for i in range(N):
            if i in W:
                sentence_with_nulls.append(sentence[j])
                j += 1
            else:
                sentence_with_nulls.append(None)

        state_pred_idx = pome_model.predict(sentence_with_nulls)
        states_prd = [pome_model.states[si].name for si in state_pred_idx]
        return states_prd

    @staticmethod
    def _return_initial_X(W, guess, N):
        init_X = []
        j = 0
        for i in range(N):
            if i in W:
                init_X.append(guess[j])
                j += 1
            else:
                init_X.append(np.random.choice(guess))
        return init_X

    @staticmethod
    def __rebuild_N(N, all_relvent_observations):
        if type(N) is list:
            return N
        else:
            return [max(N, len(seq)) for seq in all_relvent_observations]

    @staticmethod
    def build_pome_model_from_trnaisiotns(predicted_transitions, words_emms_probs, start_probs, all_states):
        states_name_to_state_mapping = {state: (
            State(DiscreteDistribution(words_emms_probs[state]), state) if state not in ['start', 'end'] else state) for
                                        state in
                                        all_states}

        _model = HiddenMarkovModel()
        for _from, _tos in predicted_transitions.items():
            if _from in ["start", "end"]: continue

            for _to, val in _tos.items():
                if _to in ['end', 'start']: continue
                _to_state = states_name_to_state_mapping[_to]

                _from_state = states_name_to_state_mapping[_from]
                _model.add_transition(_from_state, _to_state, val)

        for state, start_prob in start_probs.items():
            _to_state = states_name_to_state_mapping[state]
            _model.add_transition(_model.start, _to_state, start_prob)

        _model.bake()
        return _model

    @staticmethod
    def calculate_optimal_states_seqs(optimal_Ws, all_relvent_observations, N,
                                      emissions_table, transitions_probs, start_probs):
        all_states = list(transitions_probs.keys())
        pome_model_for_prediction = EmSequenceLabeling.build_pome_model_from_trnaisiotns(transitions_probs,
                                                                                         emissions_table,
                                                                                         start_probs,
                                                                                         all_states)
        states_name_list = [state.name for state in pome_model_for_prediction.states]

        optimal_states_seqs = []
        for i, (optimal_w, relvent_observations) in enumerate(zip(optimal_Ws, all_relvent_observations)):
            sentence_with_null_as_gaps = []
            k = 0
            for i in range(N[i]):
                if i in optimal_w:
                    obs = relvent_observations[k]
                    k += 1
                else:
                    obs = None

                sentence_with_null_as_gaps.append(obs)

            _predicted = pome_model_for_prediction.predict(sentence_with_null_as_gaps)
            predicted_tags = [states_name_list[i] for i in _predicted]

            optimal_states_seqs.append(predicted_tags)

        return optimal_states_seqs

    @staticmethod
    def calculate_optimal_states_seq(optimal_w, relvent_observations, N, emissions_table, transitions_probs,
                                     start_probs):
        w_adj_emissions_table = GibbsSampler._build_emmisions_for_sample(relvent_observations, optimal_w,
                                                                         emissions_table, N, True)
        states = list(filter(lambda x: x not in ['start', 'end'], start_probs.keys()))
        optimal_seq = EmSequenceLabeling.viterbi_flat_emiss_matrix(relvent_observations, states, start_probs,
                                                                   transitions_probs,
                                                                   w_adj_emissions_table)

        return optimal_seq

    @staticmethod
    def _build_local_emissions_probs(obs, states, emit_p):
        local_emm_probs = {st: emit_p[st][obs] for st in states}
        if sum(local_emm_probs.values()) == 0:
            local_emm_probs = {st: 1 / len(states) for st in states}
        local_emm_probs = {k: v / sum(local_emm_probs.values()) for k, v in local_emm_probs.items()}
        return {k: (v if v != 0 else 1e-5) for k, v in local_emm_probs.items()}

    @staticmethod
    def viterbi(obs, states, start_p, trans_p, emit_p):
        V = [{}]
        for st in states:
            local_emm_probs = EmSequenceLabeling._build_local_emissions_probs(obs[0], states, emit_p)
            V[0][st] = {"prob": start_p[st] * local_emm_probs[st], "prev": None}
        # Run Viterbi when t > 0
        for t in range(1, len(obs)):
            V.append({})
            local_emm_probs = EmSequenceLabeling._build_local_emissions_probs(obs[t], states, emit_p)
            for st in states:
                max_tr_prob = V[t - 1][states[0]]["prob"] * trans_p[states[0]][st]
                prev_st_selected = states[0]
                for prev_st in states[1:]:
                    tr_prob = V[t - 1][prev_st]["prob"] * trans_p[prev_st][st]
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_st

                max_prob = max_tr_prob * local_emm_probs[st]
                V[t][st] = {"prob": max_prob, "prev": prev_st_selected}

        opt = []
        max_prob = 0.0
        best_st = None
        # Get most probable state and its backtrack
        for st, data in V[-1].items():
            if data["prob"] >= max_prob:
                max_prob = data["prob"]
                best_st = st
        opt.append(best_st)

        previous = best_st

        # Follow the backtrack till the first observation
        for t in range(len(V) - 2, -1, -1):
            if previous is None:
                print("---" + str(previous))
            opt.insert(0, V[t + 1][previous]["prev"])
            previous = V[t + 1][previous]["prev"]

        return opt

    @staticmethod
    def viterbi_flat_emiss_matrix(obs, states, start_p, trans_p, emit_p):
        V = [{}]
        for st in states:
            V[0][st] = {"prob": start_p[st] * emit_p[(st, 0)], "prev": None}
        # Run Viterbi when t > 0
        for t in range(1, len(obs)):
            V.append({})
            for st in states:
                max_tr_prob = V[t - 1][states[0]]["prob"] * trans_p[states[0]][st]
                prev_st_selected = states[0]
                for prev_st in states[1:]:
                    tr_prob = V[t - 1][prev_st]["prob"] * trans_p[prev_st][st]
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_st

                max_prob = max_tr_prob * emit_p[(st, t)]
                V[t][st] = {"prob": max_prob, "prev": prev_st_selected}

        opt = []
        max_prob = 0.0
        best_st = None
        # Get most probable state and its backtrack
        for st, data in V[-1].items():
            if data["prob"] > max_prob:
                max_prob = data["prob"]
                best_st = st
        opt.append(best_st)

        previous = best_st

        # Follow the backtrack till the first observation
        for t in range(len(V) - 2, -1, -1):
            opt.insert(0, V[t + 1][previous]["prev"])
            previous = V[t + 1][previous]["prev"]

        return opt

    @staticmethod
    def _return_optimal_W(optimal_states_seq, relvent_observations,
                          emissions_table,is_known_emissions):
        if len(optimal_states_seq) == len(relvent_observations):
            _optimal_W = list(range(len(optimal_states_seq)))
        else:
            _optimal_W = EmSequenceLabeling.calculate_optimal_W(optimal_states_seq, relvent_observations,
                                                                emissions_table,is_known_emissions)
        return _optimal_W

    @staticmethod
    def calculate_optimal_W(states_seq, relvent_observations, emissions_table,is_known_emissions):
        def _is_connected(s_from, s_to, K=1):
            if s_to == "start": return False
            if s_from == "end": return False

            if s_from == "start":
                return s_to[0] == 0
            if s_to == "end":
                return s_from[0] == K

            return int((s_to[0] - s_from[0]) == 1 and s_to[1] > s_from[1])

        def _return_eq_state_weigth(eq_state, emissions_table):
            if eq_state == "end": return 0
            if is_known_emissions :
                orig_emm = emissions_table[states_seq[eq_state[1]]][relvent_observations[eq_state[0]]]
            else :
                obs = relvent_observations[eq_state[0]]
                mean = emissions_table[states_seq[eq_state[1]]][0]
                std = emissions_table[states_seq[eq_state[1]]][1]
                orig_emm = Utils.normpdf(obs,mean,std)
            log_emm = np.log(orig_emm)
            neg_log_emm = -log_emm
            return neg_log_emm

        K = len(relvent_observations)
        N = len(states_seq)

        states = [(k, n) for k, n in itertools.product(list(range(K)), list(range(N))) if k <= n]
        states += ["start", "end"]
        transitions = {state: {_state: _return_eq_state_weigth(_state, emissions_table) for _state in states if
                               _is_connected(state, _state, K - 1)} for state in states}
        graph_data = {k: v for k, v in transitions.items() if len(v) > 0}

        G = nx.DiGraph()
        weigth_dict = {}
        q = list(graph_data.items())
        while q:
            v, d = q.pop()
            for nv, nd in d.items():
                G.add_edge(v, nv, color='b', weight=1)
                weigth_dict[(v, nv)] = nd

        nx.set_edge_attributes(G, weigth_dict, "weight")

        try:
            shortest = nx.algorithms.shortest_path(G, "start", "end", weight="weight")[1:-1]
        except:
            shortest = nx.algorithms.johnson(G, weight="weight")["start"][ "end"][1:-1]
        optimal_w = list(map(lambda x: x[1], shortest))

        return optimal_w

    @staticmethod
    def _build_transitions_probs(states_ind):
        trans_probs = {}
        for state_f in states_ind:
            states_t_dist = {}
            _sum = 0
            for state_t in states_ind:
                _val = 1 if state_t > state_f else 0
                states_t_dist[state_t] = _val
                _sum += _val

            trans_probs[state_f] = states_t_dist
        return trans_probs

    @staticmethod
    def calculate_emmissions_probs_per_index(states_seq, relvent_observations,
                                             emissions_table):
        states_ind = range(len(states_seq))
        observations_ind = range(len(relvent_observations))

        emm_probs = {}
        for state_ind in states_ind:
            states_dist = {}
            _sum = 0
            for obs_ind in observations_ind:
                obs = relvent_observations[obs_ind]
                state = states_seq[state_ind]
                states_dist[obs_ind] = emissions_table[state][obs] if obs in emissions_table[state].keys() else 0
                _sum += states_dist[obs_ind]

            if _sum == 0:
                for obs_ind in observations_ind:
                    states_dist[obs_ind] = 1 / len(observations_ind)
                _sum = 1

            emm_probs[state_ind] = {k: v / _sum for k, v in states_dist.items()}

        return emm_probs, states_ind, observations_ind

    @staticmethod
    def no_transitions_viterbi(obs, states, emit_p):
        '''
        #the only condition on the transitions is - we must move forword : index of t bigger then t-1
        :param obs:
        :param states:
        :param emit_p:
        :return:
        '''

        V = [{}]
        for st in states:
            V[0][st] = {"prob": emit_p[st][obs[0]], "prev": None}
        # Run Viterbi when t > 0
        for t in range(1, len(obs)):
            V.append({})
            for st in states:
                max_tr_prob = V[t - 1][states[0]]["prob"]
                prev_st_selected = states[0]
                for prev_st in states[1:]:
                    tr_prob = V[t - 1][prev_st]["prob"]
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_st

                max_prob = max_tr_prob * emit_p[st][obs[t]]
                V[t][st] = {"prob": max_prob, "prev": prev_st_selected}

        opt = []
        max_prob = 0.0
        best_st = None
        # Get most probable state and its backtrack
        for st, data in V[-1].items():
            if data["prob"] > max_prob:
                max_prob = data["prob"]
                best_st = st
        opt.append(best_st)
        previous = best_st

        # Follow the backtrack till the first observation
        for t in range(len(V) - 2, -1, -1):
            opt.insert(0, V[t + 1][previous]["prev"])
            previous = V[t + 1][previous]["prev"]
        return opt

    @staticmethod
    def multi_choice_ksp(data_df, max_weight):
        n_bins = (len(data_df.bin.unique()))

        K = np.zeros((n_bins, max_weight + 1)) + np.finfo(float).eps
        # track = {i:{ii:[] for ii in range(max_weight+1)} for i in range(-1,n_bins)}
        track = np.zeros((data_df.shape[0], max_weight + 1))

        min_w = 0  # data_df.groupby("bin").min().sum()['Cost']
        for i in range(0, n_bins):
            for w in range(min_w, max_weight + 1):
                op_in_bin = data_df[data_df["bin"] == i]
                options = []
                options.append(K[i - 1][w])
                for j in range(op_in_bin.shape[0]):
                    adj_w = w - op_in_bin.iloc[j]["Cost"]
                    rel_point = op_in_bin.iloc[j]["Points"]
                    opt = K[i - 1][adj_w] + rel_point if (
                            (op_in_bin.iloc[j]["Cost"] < w) and (K[i - 1][adj_w] != 0)) else 0
                    options.append(opt)
                best_option_j = np.argmax(options)
                K[i][w] = options[best_option_j]

                if best_option_j != 0:
                    track[op_in_bin.iloc[best_option_j - 1]["id"]][w] = op_in_bin.iloc[best_option_j - 1]["Cost"]

        alloc_ind = [i for i, w in enumerate(track.sum(axis=0) < max_weight) if w][-1]
        alloc = [i for i in range(track.shape[0]) if track[i][alloc_ind] > 0]
        return K[-1][-1], alloc

    @staticmethod
    def _calculate_first_state_time(first_obs, n_steps_transitions, N_factor):
        all_states = n_steps_transitions[1].keys()

        probs_per_pos_state = []
        for _pos_orig_state in all_states:
            _probs_pos_state = EmSequenceLabeling.calculate_probs_single_orig(_pos_orig_state, first_obs, n_steps_transitions, N_factor)
            probs_per_pos_state.append(np.array(_probs_pos_state))

        probs_per_N = reduce(lambda x, y: x + y, probs_per_pos_state)
        return probs_per_N

    @staticmethod
    def _calculate_last_time_from_state(N_factor, dept=5):
        prob_function = lambda n: (N_factor) * ((1 - N_factor) ** (n - 1))
        probs = np.array(list(map(prob_function, range(1, dept))))
        norm_probs = probs / probs.sum()

        return norm_probs

    @staticmethod
    def calculate_probs_single_orig(_pos_orig_state, first_obs, n_steps_transitions, N_factor):
        all_n_steps = [trans_dict[_pos_orig_state][first_obs] for trans_dict in n_steps_transitions.values()]
        P_ab_all_N = np.mean(all_n_steps)

        probs = [
            n_steps_transitions[i][_pos_orig_state][first_obs] * P_ab_all_N * (N_factor) * ((1 - N_factor) ** (i - 1))
            for
            i in n_steps_transitions.keys()]
        return probs

    @staticmethod
    def _sample_N_window(from_state, to_state, n_steps_transitions, N_factor):
        prob_function = lambda n: n_steps_transitions[n][from_state][to_state] * (N_factor) * (
                    (1 - N_factor) ** (n - 1))

        probs = np.array(list(map(prob_function, range(1, len(n_steps_transitions) + 1))))
        norm_probs = probs / probs.sum()

        return probs

    @staticmethod
    def multi_choice_ksp(data_df, max_weight):
        n_bins = (len(data_df.bin.unique()))

        K = np.zeros((n_bins, max_weight + 1)) + np.finfo(float).eps
        # track = {i:{ii:[] for ii in range(max_weight+1)} for i in range(-1,n_bins)}
        track = np.zeros((data_df.shape[0], max_weight + 1))

        min_w = 0  # data_df.groupby("bin").min().sum()['Cost']
        for i in range(0, n_bins):
            for w in range(min_w, max_weight + 1):
                op_in_bin = data_df[data_df["bin"] == i]
                options = []
                options.append(K[i - 1][w])
                for j in range(op_in_bin.shape[0]):
                    adj_w = int(w - op_in_bin.iloc[j]["Cost"])
                    rel_point = op_in_bin.iloc[j]["Points"]
                    opt = K[i - 1][adj_w] + rel_point if (
                            (op_in_bin.iloc[j]["Cost"] < w) and (K[i - 1][adj_w] != 0)) else 0
                    options.append(opt)
                best_option_j = np.argmax(options)
                K[i][w] = options[best_option_j]

                if best_option_j != 0:
                    track[int(op_in_bin.iloc[best_option_j - 1]["id"])][w] = op_in_bin.iloc[best_option_j - 1]["Cost"]

        alloc_ind = [i for i, w in enumerate(track.sum(axis=0) < max_weight) if w][-1]
        alloc = [i for i in range(track.shape[0]) if track[i][alloc_ind] > 0]
        return K[-1][-1], alloc

    @staticmethod
    def init_guess_w_per_sentence(miss_sentence, n_steps_transitions,guess_state_per_obs, N=50):
        N_factor = len(miss_sentence) / N

        # extrect probabilites per skip len
        first_state_time = EmSequenceLabeling._calculate_first_state_time(guess_state_per_obs[0], n_steps_transitions, N_factor)
        last_time_from_state = EmSequenceLabeling._calculate_last_time_from_state(N_factor, len(n_steps_transitions) + 1)
        transitions_windows_time = [EmSequenceLabeling._sample_N_window(_f, _t, n_steps_transitions, N_factor)
                                    for _f, _t in zip(guess_state_per_obs, guess_state_per_obs[1:])]

        probs_list = [list(first_state_time)] + list(map(list, transitions_windows_time)) + [list(last_time_from_state)]

        # set up for mu-ch kans
        n_options = len(probs_list[0])
        n_bins = len(probs_list)

        bins_ind = list(itertools.chain(*[[i for ii in range(n_options)] for i in range(n_bins)]))
        idxs = list(range(len(list(itertools.chain(*probs_list)))))
        weights = list(itertools.chain(*[[ii for ii in range(1, n_options + 1)] for i in range(n_bins)]))
        points = list(itertools.chain(*probs_list))
        data_df = pd.DataFrame(columns=["id", "bin", "Cost", "Points"], data=list(zip(idxs, bins_ind, weights, points)))

        # multi_choice_ksp
        _, alloc = EmSequenceLabeling.multi_choice_ksp(data_df, N)
        W = np.cumsum(data_df[data_df["id"].isin(alloc)]["Cost"].values)[:-1]

        return W -1
