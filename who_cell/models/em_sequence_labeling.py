import numpy as np
from itertools import chain
import itertools
from who_cell.models.gibbs_sampler import GibbsSampler
from  tqdm import tqdm
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution
from collections import Counter
import networkx as nx
from collections import Mapping
import copy

class EmSequenceLabeling():
    def __init__(self):
        pass

    @staticmethod
    def sequence_labeling_known_emissions(all_relvent_observations,transitions_probs, start_probs,
                               emissions_table, Ng_iters=10,N = 2):
        transitions_probs = {k: v for k, v in transitions_probs.items() if not k in ['start', 'end']}
        N = EmSequenceLabeling.__rebuild_N(N,all_relvent_observations)
        gibbs_sampler = GibbsSampler(N,number_of_states_in_time = None,
                 transition_sampling_profile = 'all', multi_process = True)

        emissions_table = gibbs_sampler.impute_emissions_table_with_zeros(emissions_table, all_relvent_observations,True)
        curr_trans = transitions_probs

        if type(N) is list:
            curr_w = [sorted(np.random.choice(range(max(_N, len(obs))), len(obs), replace=False)) for obs, _N in
                      zip(all_relvent_observations, N)]
        else:
            curr_w = [sorted(np.random.choice(range(max(N, len(obs))), len(obs), replace=False)) for obs in
                      all_relvent_observations]

        optimal_states_seqs, _ = gibbs_sampler.sample_walk_from_params(all_relvent_observations, N,
                                                        emissions_table, start_probs,
                                                        curr_w, curr_trans)
        with tqdm(total=Ng_iters) as pbar:
            for i in range(Ng_iters):
                optimal_Ws = EmSequenceLabeling.calculate_optimal_Ws(optimal_states_seqs,all_relvent_observations,
                                                                   emissions_table)
                optimal_states_seqs = EmSequenceLabeling.calculate_optimal_states_seqs(optimal_Ws, all_relvent_observations,N,
                                                                     emissions_table,transitions_probs, start_probs)
                pbar.update(1)

        return  [[optimal_states_seq[w] for w in optimal_W] for optimal_W,optimal_states_seq in zip(optimal_Ws,optimal_states_seqs)]

    @staticmethod
    def most_likely_path(observations,transitions_prob,start_probs,
                               emissions_table, Ng_iters,N = 2):
        guess_state_per_obs = [[EmSequenceLabeling._closest_state(emissions_table,obs) for obs in sentence] for sentence in observations]
        pome_model = EmSequenceLabeling._build_pome_model_of_chain(transitions_prob,start_probs,emissions_table)

        all_ml_paths = []
        for sentence,guess in zip(observations,guess_state_per_obs) :
            mlp = EmSequenceLabeling._ml_path(sentence,N,pome_model,
                               emissions_table,guess, Ng_iters)
            all_ml_paths.append(mlp)

        return all_ml_paths

    @staticmethod
    def _closest_state(emissions_table,obs) :
        max_state = None
        max_prob = 0
        for s in emissions_table.keys():
            prob = emissions_table[s][obs]

            if prob > max_prob :
                max_prob = prob
                max_state = s
        return max_state


    @staticmethod
    def _ml_path(sentence,N,pome_model,emissions_table,guess, Ng_iters):
        W = sorted(np.random.choice(range(max(N, len(sentence))), len(sentence), replace=False))
        X = EmSequenceLabeling._return_initial_X(W,guess,N)

        for iter in range(Ng_iters) :
            W = EmSequenceLabeling._return_optimal_W(sentence,X,emissions_table)
            X = EmSequenceLabeling._return_optimal_X(sentence,W,pome_model,N)

        return None

    @staticmethod
    def _return_optimal_W(sentence, X, emissions_table) :
        def is_connected(s_from, s_to):
            return int((s_to[0] - s_from[0]) == 1 and s_to[1] > s_from[1])

        graph_states = [(k, n) for k, n in itertools.product(list(range(len(sentence))), list(range(len(X)))) if k <= n]
        transitions = {state: {_state: 1 for _state in graph_states if is_connected(state, _state)} for state in graph_states}
        transitions = {k: v for k, v in transitions.items() if len(v) > 0}

        final_transitions = copy.copy(transitions)
        final_transitions["start"] = {}
        for f_state,tos in graph_states:
            if f_state[0] == 0:
                final_transitions["start"][f_state] = 1
            for t_state,_ in tos.items():
                if t_state[0] == len(sentence)-1 :
                    final_transitions[t_state] = {"end":1}

        G = nx.DiGraph()
        q = list(transitions.items())
        while q:
            v, d = q.pop()
            for nv, nd in d.items():
               G.add_edge(v, nv, color='y', weight=2)
               if isinstance(nd, Mapping):
                   q.append((nv, nd))




    @staticmethod
    def _return_optimal_X(sentence,W,pome_model,N):
        sentence_with_nulls = []
        j = 0
        for i in range(N):
            if i in W:
                sentence_with_nulls.append(sentence[j])
                j += 1
            else:
                sentence_with_nulls.append(None)

        return pome_model.predict(sentence_with_nulls)

    @staticmethod
    def _return_initial_X(W,guess,N) :
        init_X = []
        j=0
        for i in range(N) :
            if i in W :
                init_X.append(guess[j])
                j += 1
            else:
                init_X.append(np.random.choice(guess))
        return init_X

    @staticmethod
    def __rebuild_N(N, all_relvent_observations) :
        if type(N) is list :
            return N
        else :
            return [max(N,len(seq)) for seq in all_relvent_observations]

    @staticmethod
    def build_pome_model_from_trnaisiotns(predicted_transitions, words_emms_probs, start_probs, all_states):
        states_name_to_state_mapping = {state: (State(DiscreteDistribution(words_emms_probs[state]), state) if state not in ['start','end'] else state) for state in
                                        all_states}

        _model = HiddenMarkovModel()
        for _from, _tos in predicted_transitions.items():
            if _from in ["start","end"]: continue

            for _to, val in _tos.items():
                if _to in ['end','start']: continue
                _to_state = states_name_to_state_mapping[_to]

                _from_state = states_name_to_state_mapping[_from]
                _model.add_transition(_from_state, _to_state, val)

        for state,start_prob in start_probs.items() :
            _to_state = states_name_to_state_mapping[state]
            _model.add_transition(_model.start,_to_state,start_prob)

        _model.bake()
        return _model

    @staticmethod
    def calculate_optimal_states_seqs(optimal_Ws, all_relvent_observations,N,
                                  emissions_table, transitions_probs, start_probs) :
        all_states = list(transitions_probs.keys())
        pome_model_for_prediction = EmSequenceLabeling.build_pome_model_from_trnaisiotns(transitions_probs,
                                                                                         emissions_table,
                                                                                         start_probs,
                                                                                         all_states)
        states_name_list = [state.name for state in pome_model_for_prediction.states ]

        optimal_states_seqs = []
        for i,(optimal_w,relvent_observations) in enumerate(zip(optimal_Ws,all_relvent_observations) ):
            sentence_with_null_as_gaps = []
            k = 0
            for i in range(N[i]):
                if i in optimal_w :
                    obs = relvent_observations[k]
                    k += 1
                else :
                    obs = None

                sentence_with_null_as_gaps.append(obs)

            _predicted = pome_model_for_prediction.predict(sentence_with_null_as_gaps)
            predicted_tags = [states_name_list[i] for i in _predicted]

            optimal_states_seqs.append(predicted_tags)

        return optimal_states_seqs

    @staticmethod
    def calculate_optimal_states_seq(optimal_w,relvent_observations,N,emissions_table, transitions_probs,start_probs):
        w_adj_emissions_table = GibbsSampler._build_emmisions_for_sample(relvent_observations, optimal_w,
                                                                         emissions_table, N, True)
        states = list(filter(lambda x:x not in ['start','end'],start_probs.keys()))
        optimal_seq = EmSequenceLabeling.viterbi_flat_emiss_matrix(relvent_observations, states, start_probs, transitions_probs,
                                                 w_adj_emissions_table)

        return optimal_seq

    @staticmethod
    def _build_local_emissions_probs(obs,states,emit_p):
        local_emm_probs = {st:emit_p[st][obs] for st in states}
        if sum(local_emm_probs.values()) == 0 :
            local_emm_probs = {st:1/len(states) for st in states}
        local_emm_probs = {k: v/sum(local_emm_probs.values()) for k, v in local_emm_probs.items()}
        return {k:(v if v != 0 else 1e-5) for k,v in local_emm_probs.items()}

    @staticmethod
    def viterbi(obs, states, start_p, trans_p, emit_p):
        V = [{}]
        for st in states:
            local_emm_probs = EmSequenceLabeling._build_local_emissions_probs(obs[0],states,emit_p)
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
            if previous is None :
                print("---" + str(previous))
            opt.insert(0, V[t + 1][previous]["prev"])
            previous = V[t + 1][previous]["prev"]

        return opt

    @staticmethod
    def viterbi_flat_emiss_matrix(obs, states, start_p, trans_p, emit_p):
        V = [{}]
        for st in states:
            V[0][st] = {"prob": start_p[st] * emit_p[(st,0)], "prev": None}
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

                max_prob = max_tr_prob * emit_p[(st,t)]
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
    def calculate_optimal_Ws(optimal_states_seqs, all_relvent_observations,
                        emissions_table) :
        optimal_Ws = []
        for optimal_states_seq,relvent_observations in zip(optimal_states_seqs,all_relvent_observations) :
            if len(optimal_states_seq) == len(relvent_observations) :
                _optimal_W = list(range(len(optimal_states_seq)))
            else :
                _optimal_W = EmSequenceLabeling.calculate_optimal_W(optimal_states_seq, relvent_observations,
                                                                emissions_table)
            optimal_Ws.append(_optimal_W)
        return optimal_Ws

    @staticmethod
    def calculate_optimal_W(states_seq, relvent_observations, emissions_table) :
        emm_probs_for_ind,states_ind,observations_ind= EmSequenceLabeling.calculate_emmissions_probs_per_index(states_seq, relvent_observations,
                                                                                    emissions_table)
        transitions_probs = EmSequenceLabeling._build_transitions_probs(states_ind)
        start_p = {i:1/len(states_ind) for i in states_ind}

        optimal_path = EmSequenceLabeling.viterbi(observations_ind, states_ind, start_p,
                                                  transitions_probs, emm_probs_for_ind)

        return optimal_path

    @staticmethod
    def _build_transitions_probs(states_ind) :
        trans_probs = {}
        for state_f in states_ind :
            states_t_dist = {}
            _sum=0
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

        return emm_probs,states_ind,observations_ind

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