import pomegranate
from pomegranate import *
import numpy as np
import who_cell.config as base_config
import copy
import pandas as pd
import itertools
from functools import reduce
from scipy.stats import binom
from tqdm import tqdm
from collections import Counter
import random
import sys
from who_cell.transitions_dict import transitions_dict
from functools import lru_cache
from functools import partial
random.seed(10)
from multiprocessing import Pool
from who_cell.models.utils import Utils
from  itertools import chain

from who_cell.simulation.simulation_for_gibbs import Simulator_for_Gibbs

class GibbsSampler() :
    def __init__(self,length_of_chain,number_of_states_in_time = None,transition_sampling_profile = 'all'):
        self.N = length_of_chain

        self.transition_sampling_profile = transition_sampling_profile

    # region Public

    def sample(self,is_acyclic, all_relvent_observations, start_probs,
               known_mues,sigmas, Ng_iters, w_smapler_n_iter = 20,N=None,is_mh = False):
        N = self.N if N is None else N
        states = list(set(list(start_probs.keys()) + ['start','end']))
        state_to_distrbution_param_mapping = self.__build_initial_state_to_distrbution_param_mapping(known_mues,sigmas,
                                                                                                     states)

        # TODO : why we dont know is_known_mues in this function ? if we know mues whay we need thw params ?
        #TODO : send isacyclic - if cyclic think of another way to calculate, if not validate the previous case
        priors = self._calc_distributions_prior(is_acyclic,all_relvent_observations, (len(states) -2) if is_acyclic else N)
        curr_mus = self.build_initial_mus(sigmas,priors,known_mues,is_acyclic)
        curr_trans = self.build_initial_transitions(states,is_acyclic)
        if is_acyclic :
            curr_w = [list(range(len(obs))) for obs in all_relvent_observations]
        else :
            curr_w = [sorted(np.random.choice(range(N), len(obs), replace=False)) for obs in all_relvent_observations]

        state_to_distrbution_param_mapping = self._update_distributions_params(state_to_distrbution_param_mapping, curr_mus)
        curr_walk,alpha= self.sample_walk_from_params(is_acyclic,all_relvent_observations,N, state_to_distrbution_param_mapping,start_probs,
                                                 curr_w, curr_trans)

        sampled_states,observations_sum = self._exrect_samples_from_walk(curr_walk,all_relvent_observations,curr_w,
                                                                         state_to_distrbution_param_mapping,
                                                                         curr_mus,sigmas)
        sampled_transitions = self._exrect_transitions_from_walk(curr_walk,states,curr_w)

        all_sampled_transitions = [sampled_transitions]
        all_transitions = [curr_trans]
        all_states = [sampled_states]
        all_observations_sum = [observations_sum]
        all_mues = [curr_mus]
        all_ws = [curr_w]
        with tqdm(total=Ng_iters) as pbar:
            for i in range(Ng_iters):
                curr_mus = self.sample_mus_from_params(sampled_states, observations_sum,priors, sigmas,known_mues)
                state_to_distrbution_param_mapping = self._update_distributions_params(
                    state_to_distrbution_param_mapping, curr_mus)

                curr_trans,_ = self.sample_trans_from_params(sampled_transitions,states,
                                                             curr_params =[curr_trans,curr_w,curr_walk,None,state_to_distrbution_param_mapping],
                                                             stage_name="transitions" if is_mh else "no_mh" ,
                                                             observations = all_relvent_observations)
                curr_w,_ = self.sample_ws_from_params(all_relvent_observations, curr_walk,
                                                      state_to_distrbution_param_mapping,N,
                                                      n_iters=w_smapler_n_iter,
                                                      curr_params=[curr_trans, curr_w, curr_walk, None, state_to_distrbution_param_mapping],
                                                      stage_name="w"  if is_mh else "no_mh",
                                                      observations=all_relvent_observations)

                curr_walk,_ = self.sample_walk_from_params(is_acyclic,all_relvent_observations,N,
                                                         state_to_distrbution_param_mapping,start_probs,
                                                         curr_w, curr_trans,
                                                           curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                      state_to_distrbution_param_mapping],
                                                         stage_name="walk"  if is_mh else "no_mh",
                                                         observations=all_relvent_observations)


                sampled_transitions = self._exrect_transitions_from_walk(curr_walk,states,curr_w)
                sampled_states,observations_sum = self._exrect_samples_from_walk(curr_walk,all_relvent_observations,
                                                                                 curr_w,state_to_distrbution_param_mapping,
                                                                                 curr_mus,sigmas)

                all_sampled_transitions.append(sampled_transitions)
                all_transitions.append(curr_trans)
                all_states.append(sampled_states)
                all_observations_sum.append(observations_sum)
                all_mues.append(curr_mus)
                all_ws.append(curr_w)
                pbar.update(1)
        return all_states,all_observations_sum, all_sampled_transitions,all_mues,all_ws,all_transitions

    def sample_known_emissions(self, all_relvent_observations, start_probs,
                               emissions_table, Ng_iters, w_smapler_n_iter = 5,N = None,is_mh = False):
        emissions_table = self.impute_emissions_table_with_zeros(emissions_table,all_relvent_observations)
        N = self.N if N is None else N
        states = list(set(list(start_probs.keys()) + ['start', 'end']))

        curr_trans = self.build_initial_transitions(states, True)

        curr_w = [list(range(len(obs))) for obs in all_relvent_observations]

        curr_walk,alpha = self.sample_walk_from_params(True, all_relvent_observations, N,
                                                 emissions_table, start_probs,
                                                 curr_w, curr_trans)
        _states_picked_by_w = [[seq[i] for i in ws] for ws, seq in zip(curr_w, curr_walk)]

        sampled_transitions = self._exrect_transitions_from_walk(curr_walk, states, curr_w)

        all_alphas = [alpha]
        all_sampled_transitions = [sampled_transitions]
        all_transitions = [curr_trans]
        all_ws = [curr_w]
        all_states_picked_by_w = [_states_picked_by_w]
        with tqdm(total=Ng_iters) as pbar:
            for i in range(Ng_iters):
                curr_trans,alpha0 = self.sample_trans_from_params(sampled_transitions, states,
                                                           curr_params =[curr_trans,curr_w,curr_walk,None,emissions_table],
                                                           stage_name="transitions"  if is_mh else "no_mh",
                                                           observations = all_relvent_observations)
                curr_w,alpha1 = self.sample_ws_from_params(all_relvent_observations, curr_walk,emissions_table, N,
                                                     n_iters=w_smapler_n_iter,
                                                    curr_params=[curr_trans, curr_w, curr_walk, None, emissions_table],
                                                    stage_name="w"  if is_mh else "no_mh",
                                                    observations=all_relvent_observations)
                _states_picked_by_w = [[seq[i] for i in ws] for ws, seq in zip(curr_w, curr_walk)]

                curr_walk,alpha2 = self.sample_walk_from_params(True, all_relvent_observations, N,
                                                         emissions_table, start_probs,
                                                         curr_w, curr_trans,
                                                         curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                      emissions_table],
                                                         stage_name="walk"  if is_mh else "no_mh",
                                                         observations=all_relvent_observations)

                sampled_transitions = self._exrect_transitions_from_walk(curr_walk, states, curr_w)

                all_alphas.append(np.mean([alpha0,alpha1,alpha2]))
                all_sampled_transitions.append(sampled_transitions)
                all_transitions.append(curr_trans)
                all_states_picked_by_w.append(_states_picked_by_w)
                all_ws.append(curr_w)
                pbar.update(1)
        return  all_sampled_transitions, all_ws, all_transitions,all_states_picked_by_w,all_alphas

    def compare_transitions_prob_to_count(self,transitions_prob,transitions_count,n_traj,N,d):

        #the previous mathods need states as tuple and not str
        transitions_prob = {eval(k):{eval(kk):vv for kk,vv in v.items() } for k,v in transitions_prob.items()}
        transitions_count = {eval(k): {eval(kk): vv for kk, vv in v.items()} for k, v in transitions_count.items()}

        states = transitions_prob.keys()
        start_prob = self._build_start_prob(transitions_prob,N,d)
        prior = self._fwd_bkw(states, start_prob, transitions_prob, None, N, d, only_forward = True)

        _transitions_prob_to_count_list = []
        for _from_state,_to_states in transitions_prob.items() :
            for _to_state,prob in _to_states.items() :
                _from_prior = prior[_from_state[1]][_from_state]
                _to_prior = _from_prior * prob
                expected_trans_count = _to_prior * n_traj
                trans_count = transitions_count[_from_state][_to_state]

                _transitions_prob_to_count_list.append([str(_from_state),str(_to_state),expected_trans_count,trans_count])

        compr_df = pd.DataFrame(columns=['from','to','expected_count','count'],data = _transitions_prob_to_count_list)

        return compr_df

    # endregion

    # region Private
    def __build_initial_state_to_distrbution_param_mapping(self,known_mues, sigmas,states) :
        if known_mues is not None :
            return {state:(known_mues[state],sigmas[state])  for state in states if state not in ['start','end']}
        else :
            return {state:(None,sigmas[state])  for state in states  if state not in ['start','end']}

    def _calc_priors(self,all_obs_flat_arr,predictions,n_states) :
        state_to_params = {}
        for k in range(n_states):
            y_k = all_obs_flat_arr[predictions == k]
            _max_k = y_k.max()
            _min_k = y_k.min()

            chi = (_max_k + _min_k) / 2
            kapa = 1 / (_max_k - _min_k) ** 2

            state_to_params[k] = (chi, kapa)
        return state_to_params

    def _calc_distributions_prior(self,is_acyclic,all_relvent_observations,N):
        '''
        :param is_acyclic:
        :param all_relvent_observations:
        :param N: if  is_acyclic == True we assume N == number of states
        :return:
        '''
        if not is_acyclic :
            weighted_obs = self._calc_weighted_obs_for_init(all_relvent_observations, N)
            chi = (weighted_obs.max(axis=0) + weighted_obs.min(axis=0)) / 2
            kapa = 1 / ((weighted_obs.max(axis=0) - weighted_obs.min(axis=0)) ** 2)
            time_to_params = {i:(chi[i],kapa[i]) for i in range(len(chi))}
            return time_to_params
        else :
            # if is_acyclic == True we assume N == number of states
            all_obs_flat = list(itertools.chain(*all_relvent_observations))
            all_obs_flat_arr = np.array(all_obs_flat).reshape(len(all_obs_flat), 1)

            gmm_model = gmm.GeneralMixtureModel.from_samples(distributions=NormalDistribution,
                                                             n_components=N,
                                                             X=all_obs_flat_arr, n_init=5)
            predictions = gmm_model.predict(all_obs_flat_arr)
            time_to_params = self._calc_priors(all_obs_flat_arr,predictions,N)
            return time_to_params

    def _update_distributions_params(self, state_to_distrbution_param_mapping, curr_mus):
        return {state:(curr_mus[state], params[1]) for state,params in
                state_to_distrbution_param_mapping.items() if state not in ['start','end']}

    @lru_cache(225)
    def _prob_for_assigment(self,time, obs_ind, N, n_obs):
        p_prob_of_observation = n_obs / N

        _pre_prob = binom.pmf(obs_ind, time, p_prob_of_observation)
        _post_prob = binom.pmf(n_obs - obs_ind, N - time, p_prob_of_observation)

        return _pre_prob * _post_prob

    def _calc_weighted_obs_for_init(self,sampled_trajs, N):
        #TODO: this is only the case for not acyclic ?
        weighted_obs_matrix_for_initial_conditions = np.zeros((len(sampled_trajs), N))
        for l in range(N):
            for traj_ind, traj in enumerate(sampled_trajs):
                weighted_obs = 0
                probs_sum = 0
                for obs_ind, obs in enumerate(traj):
                    _prob = self._prob_for_assigment(l, obs_ind, N, len(traj))
                    probs_sum += _prob
                    weighted_obs += obs * _prob

                weighted_obs_matrix_for_initial_conditions[traj_ind, l] = weighted_obs / probs_sum
        return weighted_obs_matrix_for_initial_conditions

    # @jit
    def _msf_creator_old(self,y_from_x_probs, not_y_from_x_probs, w):
        prob = 1
        i = 0
        i_w = 0
        while (i < len(y_from_x_probs)):
            if i_w > len(y_from_x_probs):
                raise Exception()

            if i_w >= len(w):
                prob = prob * not_y_from_x_probs[i]
            else:
                if i < w[i_w]:
                    prob = prob * not_y_from_x_probs[i]
                if i == w[i_w]:
                    prob = prob * y_from_x_probs[i]
                    i_w += 1
            i += 1
        return prob

    def _msf_creator(self,y_from_x_probs,N, w):
        _w = np.concatenate([np.array([-1]),w,np.array([N])])
        prob = 1

        i_w = 0
        while (i_w < len(w)):
            _prob_sum = 0
            for i in range(_w[i_w] + 1, _w[i_w + 2]):
                _prob_sum += y_from_x_probs[(i_w,i)]
            prob = prob * (y_from_x_probs[(i_w,w[i_w])] / _prob_sum)
            i_w += 1

        return prob

    def msf_creator(self,y_from_x_probs,N, is_rec=False):
        if not is_rec:
            return partial(self._msf_creator, y_from_x_probs,N)
        else:
            raise NotImplementedError()
            # return partial(self._rec_msf_creator, y_from_x_probs, not_y_from_x_probs)

    def choice(self,options, probs):
        x = np.random.rand()
        cum = 0
        for i, p in enumerate(probs):
            cum += p
            if x < cum:
                break
        return options[i]

    def sample_cond_prob_single_dim(self,k,N,dims_vector, ind_dim_for_sample,y_from_x_probs):
        _pre_value = dims_vector[(ind_dim_for_sample - 1)] if ind_dim_for_sample != 0 else -1
        _post_value = dims_vector[(ind_dim_for_sample + 1)] if (ind_dim_for_sample != (k - 1)) else N

        possible_options_for_dim = range(_pre_value + 1, _post_value)
        probs_of_opts = [y_from_x_probs[(ind_dim_for_sample,poss_opt)] for poss_opt in possible_options_for_dim]
        probs_of_opts = probs_of_opts if sum(probs_of_opts) != 0 else np.array([1 for i in probs_of_opts])

        return self.choice(possible_options_for_dim, np.array(probs_of_opts)/sum(probs_of_opts))

    def _calc_alpha(self,_old_dim_vector, _curr_dim_vector, y_from_x_probs) :
        old_prob = reduce(lambda x, y: x * y, [y_from_x_probs[(k, w)] for k, w in enumerate(_old_dim_vector)])
        curr_prob = reduce(lambda x, y: x * y, [y_from_x_probs[(k, w)] for k, w in enumerate(_curr_dim_vector)])

        if curr_prob == 0 :
            if old_prob == 0 :
                return 1
            else :
                return 0
        elif old_prob == 0:
            return 1

        return curr_prob / old_prob


        return None

    def sample_msf_using_sim(self,k,N, n_iter,y_from_x_probs):
        initial_vector = sorted(random.sample(range(N), k))

        _curr_dim_vector = copy.copy(initial_vector)
        flag = 0
        for _ in range(n_iter):
            # if flag > 3*k : break
            for dim in range(k):
                _sample = self.sample_cond_prob_single_dim(k,N,_curr_dim_vector, dim,y_from_x_probs)
                _old_dim_vector = _curr_dim_vector
                _curr_dim_vector[dim] = _sample
                alpha_for_mh = self._calc_alpha(_old_dim_vector,_curr_dim_vector,y_from_x_probs)
                if np.random.rand() < alpha_for_mh :
                    if _old_dim_vector == _curr_dim_vector :
                        flag += 1
                    continue
                else :
                    flag += 1
                    _curr_dim_vector = _old_dim_vector

        return copy.copy(_curr_dim_vector)

    def _build_states_map(self, known_pome_model):
        states = {}
        for state in known_pome_model.states:
            if ("start" in state.name ) or ('end' in state.name) :
                continue
            state_name = eval(state.name)
            states[state_name] = state

        return states

    @staticmethod
    def __probability_from_dist_params(observation,params) :
        if observation is None :
            return 1
        if type(params) is tuple or type(params) is list :
            dist = pomegranate.NormalDistribution(params[0],params[1])
            return dist.probability(observation)
        if type(params) is pomegranate.NormalDistribution :
            return params.probability(observation)
        if type(params) is dict :
            return params[observation] if observation in params.keys() else 0

    @staticmethod
    def _build_emmisions_for_sample(is_acyclic,sample, w, state_to_distrbution_param_mapping,N,normalized_emm =  True,
                                    is_known_emm = False):
        states = list(state_to_distrbution_param_mapping.keys())
        emmisions = {}
        ind_obs = 0
        for time_ind in range(N):
            if time_ind in w:
                observation = sample[ind_obs]
                ind_obs += 1
            else:
                observation = None
            _sum = 0

            for state in states :
                if (state in ['start','end']) :
                    _emm = 0
                else :
                    _emm = GibbsSampler.__probability_from_dist_params(observation, state_to_distrbution_param_mapping[state])
                #if not cyclic case : if the start is out of time is zero
                if (not is_acyclic) and (state[1] != time_ind) :
                    _emm = 0

                emmisions[(state, time_ind)] = _emm
                _sum += _emm

            _sum = _sum if (_sum > sys.float_info.epsilon) else sys.float_info.epsilon
            for state in states :
                if normalized_emm :
                    _res = emmisions[(state, time_ind)] / _sum
                else :
                    _res = emmisions[(state, time_ind)]

                _res = sys.float_info.epsilon if pd.isnull(_res) else _res
                emmisions[(state, time_ind)] = _res

        return emmisions

    def _build_start_prob(self,states,start_probabilites):
        if start_probabilites is not None : return start_probabilites
        _start_prob = {}
        _sum = 0
        for state in states.keys():
            if type(state) is str :
                state = eval(state)
            _start_prob[state] = 1 if state[1] == 0 else 0
            _sum += 1 if state[1] == 0 else 0

        _start_prob = {k:(v/_sum) for k,v in _start_prob.items()}
        return _start_prob

    @staticmethod
    def __sample_emmisions_matrix(emm_prob,state,observation_ind,is_acyclic):
        if (not is_acyclic) :
            return emm_prob[(state,observation_ind)]  if state[1] == observation_ind else 0
        else :
            return emm_prob[(state,observation_ind)]

    @staticmethod
    def _sample_trans_matrix( trans_prob, from_state, to_state,is_acyclic):
        if ((not is_acyclic) and ((to_state[1] - from_state[1]) != 1)) : return 0
        if from_state not in trans_prob.keys() : return 0
        if to_state not in trans_prob[from_state].keys() : return 0

        return trans_prob[from_state][to_state]

    @staticmethod
    def __sample_single_time(is_acyclic,prev_state, walk, fwd, trans_prob, n) :
        if n == -1 :
            return walk

        states_of_time = []
        prob_of_states_of_time = []

        for state,prob in fwd[n].items() :
            if ((not is_acyclic) and (state[1] != n)) :
                continue

            prob_for_sample = prob * GibbsSampler._sample_trans_matrix(trans_prob, state, prev_state,is_acyclic)

            states_of_time.append(state)
            prob_of_states_of_time.append(prob_for_sample)

        #TODO: maybe we should remove zero chance transitions ?
        prob_of_states_of_time = prob_of_states_of_time if sum(prob_of_states_of_time) != 0 else [1 for i in prob_of_states_of_time]

        _sum = sum(prob_of_states_of_time)
        sampled_state_inde = np.random.choice(range(len(states_of_time)),
                                              p = [p/_sum for p in prob_of_states_of_time] )
        sampled_state = states_of_time[sampled_state_inde]

        walk.append(sampled_state)

        return GibbsSampler.__sample_single_time(is_acyclic,sampled_state,walk,fwd, trans_prob, n - 1)

    @staticmethod
    def _build_single_walk_from_postrior(is_acyclic,fwd,trans_prob,N) :
        walk = []
        states_of_time = []
        prob_of_states_of_time = []

        for state,prob in fwd[N-1].items() :
            if ((not is_acyclic) and (state[1] != (N-1))) :
                continue

            states_of_time.append(state)
            prob_of_states_of_time.append(prob)

        prob_of_states_of_time = prob_of_states_of_time if sum(prob_of_states_of_time) != 0 else np.array([1 for i in prob_of_states_of_time])
        # print(f"{prob_of_states_of_time}:{sum(prob_of_states_of_time)}")
        # print(f"{list((np.array(prob_of_states_of_time)/sum(prob_of_states_of_time)))}")
        # print('---')
        sampled_state_inde = np.random.choice(range(len(states_of_time)), p = list((np.array(prob_of_states_of_time)/sum(prob_of_states_of_time))))
        sampled_state = states_of_time[sampled_state_inde]

        walk.append(sampled_state)

        _walk  = GibbsSampler.__sample_single_time(is_acyclic,sampled_state,walk,fwd,trans_prob,N-2)
        walk = list(reversed(_walk))

        return walk

    @staticmethod
    def _fwd_bkw(is_acyclic, states, start_prob, trans_prob, emm_prob,N,only_forward = False):
        """Forward–backward algorithm."""
        # Forward part of the algorithm
        fwd = []
        for observation_i in range(N):
            f_curr = {}
            for st in states:
                if (not is_acyclic) and (st[1] != observation_i) :
                    f_curr[st] = 0
                    continue

                if observation_i == 0:
                    # base case for the forward part
                    if st in start_prob.keys() :
                        prev_f_sum = start_prob[st]
                    else :
                        prev_f_sum = start_prob[str(st)]
                else:
                    prev_f_sum = 0
                    prev_f_sum = sum([f_prev[k] * GibbsSampler._sample_trans_matrix(trans_prob, k, st,is_acyclic) for k in states if f_prev[k] !=0])

                if emm_prob is not None :
                    # F-B case
                    f_curr[st] = GibbsSampler.__sample_emmisions_matrix(emm_prob,st,observation_i,is_acyclic) * prev_f_sum
                else :
                    # when we want to calculate only transitions (deviation of prob from expected sample count)
                    _emm = int(st[1] == observation_i) if (not is_acyclic) else 1
                    f_curr[st] = _emm * prev_f_sum

            fwd.append(f_curr)
            f_prev = f_curr

        if only_forward :
            return fwd

        # Backward part of the algorithm
        return GibbsSampler._build_single_walk_from_postrior(is_acyclic,fwd,trans_prob, N)

    def sample_mus_from_params(self,all_sampled_states, sum_relvent_observations, priors,  sigmas,known_mues):
        if known_mues is not None :
            return known_mues

        sis = sum_relvent_observations
        nis = all_sampled_states

        new_mues = {}
        for state in sis.keys() :
            time = int(state[1])
            ξ = 0 # priors[time][0]
            κ = 0 # priors[time][1]
            _mue = (sis[state] + ξ * κ * sigmas[state]) / (κ * sigmas[state] + nis[state])
            _sig = (sigmas[state]) #/ (κ * sigmas[state] + nis[state])
            new_mues[state] = np.random.normal(_mue, _sig)

        return new_mues

    @Utils.update_based_on_alpha
    def sample_trans_from_params(self,all_transitions,states):
        sampled_transitions = {state: {} for state in states}

        if self.transition_sampling_profile == 'all' :
            _transition_dict = all_transitions.items()

        if self.transition_sampling_profile == 'observed':
            _transition_dict = all_transitions.observed_transitions_dict.items()

        if self.transition_sampling_profile == 'extended':
            _transition_dict = all_transitions.extended_observed_transitions_dict.items()

        for state,poss_trans in _transition_dict:
            poss_trans_states = [state for state in poss_trans.keys()]
            poss_trans_counts = [state for state in poss_trans.values()]

            sample = np.random.dirichlet(poss_trans_counts)

            for _state_ind,_state in enumerate(poss_trans_states):
                sampled_transitions[state][_state] = sample[_state_ind]

        return sampled_transitions

    def __prob_obs_for_state(self,state,obs,state_to_distrbution_param_mapping,is_tuple):
        if is_tuple :
            return pomegranate.distributions.NormalDistribution(state_to_distrbution_param_mapping[state][0],
                                                            state_to_distrbution_param_mapping[state][1]).probability(obs)
        else :
            return state_to_distrbution_param_mapping[state][obs] if obs in state_to_distrbution_param_mapping[state] else 0

    def _extract_relevent_probs_from_walk(self, traj, walk, state_to_distrbution_param_mapping) :
        is_tuple = type(next(iter(state_to_distrbution_param_mapping.values()))) is tuple
        y_from_x_probs = {(obs,state): self.__prob_obs_for_state(walk[state],traj[obs],state_to_distrbution_param_mapping,is_tuple) for
                          obs,state in itertools.product(range(len(traj)), range(len(walk))) }

        return y_from_x_probs

    def _sample_ws_from_params(self,state_to_distrbution_param_mapping,n_iters,N,sample_data) :
        if type(N) is list :
            traj, _curr_walk,_N = sample_data
            seq_length = max(_N, len(traj))
        else :
            traj, _curr_walk = sample_data
            seq_length = max(N, len(traj))

        if len(traj) == len(_curr_walk) :
            return list(range(len(traj)))

        y_from_x_probs = self._extract_relevent_probs_from_walk(traj, _curr_walk, state_to_distrbution_param_mapping)

        _simulted_w = self.sample_msf_using_sim(len(traj), seq_length, n_iters, y_from_x_probs)
        return _simulted_w

    @Utils.update_based_on_alpha
    def sample_ws_from_params(self,sampled_trajs, curr_walk,state_to_distrbution_param_mapping,N, n_iters=10):
        if type(N) is list :
            samples_data = [(sampled_trajs[i],curr_walk[i],N[i]) for i in  range(len(sampled_trajs))]
        else :
            samples_data = zip(sampled_trajs, curr_walk)

        _partial_sample_ws_from_params = partial(self._sample_ws_from_params,state_to_distrbution_param_mapping,n_iters,N)

        with Pool(base_config.n_cores) as p :
            result_per_traj = p.map(_partial_sample_ws_from_params,samples_data)
        return result_per_traj

    def _sample_walk_from_params(self,is_acyclic, N, state_to_distrbution_param_mapping,start_prob,  curr_trans,samples_data):
        if type(N) is list :
            sample, _curr_ws,_N = samples_data
            seq_length = max(len(sample), _N)
        else :
            sample, _curr_ws = samples_data
            seq_length = max(len(sample), N)

        emmisions = GibbsSampler._build_emmisions_for_sample(is_acyclic, sample,
                                                     _curr_ws, state_to_distrbution_param_mapping, seq_length)
        posterior = self._fwd_bkw(is_acyclic, state_to_distrbution_param_mapping.keys(),
                                  start_prob, curr_trans, emmisions, seq_length)
        return posterior

    @Utils.update_based_on_alpha
    def sample_walk_from_params(self,is_acyclic, sampled_trajs,N, state_to_distrbution_param_mapping,
                                start_prob, curr_ws, curr_trans):
        if type(N) is list :
            samples_data = [(sampled_trajs[i], curr_ws[i],N[i]) for i in range(len(sampled_trajs))]
        else :
            samples_data = list(zip(sampled_trajs , curr_ws))

        _partial_sample_walk_from_params = partial(self._sample_walk_from_params,is_acyclic, N,
                                                   state_to_distrbution_param_mapping,start_prob,  curr_trans)
        with Pool(base_config.n_cores) as p:
            walks = p.map(_partial_sample_walk_from_params, samples_data)

        return walks

    def build_initial_transitions(self,states,is_acyclic):
        if not is_acyclic :
            N_time = max([state[1] for state in states if str(state[1]).isnumeric()])
        initial_transitions = {state: {} for state in states}

        normalization_factors = {}
        for _s1 in initial_transitions.keys():
            if _s1 == 'end' : continue

            _sum = 0
            for _s2 in initial_transitions.keys():
                if _s2 == 'start' : continue

                _val = 0
                if not is_acyclic :
                    if _s1 == 'start' :
                        _val = int(_s2[1] == 0)
                    elif _s2 == 'end':
                        _val =  int(_s1[1] == N_time)
                    else :
                        _val =  int((_s2[1] - _s1[1]) == 1)
                else :
                    _val = np.random.rand()

                initial_transitions[_s1][_s2] = _val
                _sum += _val
            normalization_factors[_s1] = _sum

        initial_transitions = {_s1:{__s1:__s2/normalization_factors[_s1] for __s1,__s2 in s2.items()} for _s1,s2 in initial_transitions.items()}
        return initial_transitions

    def build_initial_mus(self,sigmas,priors,known_mues,is_acyclic):
        if known_mues is not None:
            return known_mues
        if not is_acyclic :
            return {state:(np.random.randn() + np.array(priors[state[1]][0]))
                    for state in sigmas.keys() if state not in ['start','end']}
        else :
            sorted_states = sorted([eval(sig) for sig in sigmas.keys() if sig != 'start'],key= lambda x:x[1])
            sorted_states = [str(v) for v in sorted_states]

            prior_dist_params = sorted(list(priors.items()),key= lambda x:x[1][0])
            prior_states = [k for k,v in prior_dist_params]

            states_to_prior_maping = {k:v for k,v in zip(sorted_states,prior_states)}

            return {state: (priors[states_to_prior_maping[state]][0])
                    for state in sigmas.keys() if state not in ['start', 'end']}

    def is_ordered_states_in_sequence(self,ordered_states, sequence):
        i = 0
        for val_to_look in ordered_states:
            while (i < len(sequence)):
                if sequence[i] == val_to_look: break
                i = i + 1
        return not i == len(sequence)

    def is_optional_transition_is_seen(self,opt_seen_trans, pre_states_from_trajectory, post_states_from_trajectory,
                                       all_transitions_of_walk, _curr_walk):
        idx_item_in_list = lambda item, _list: [i for i, val in enumerate(_list) if val == item]
        idx_of_trans = idx_item_in_list(opt_seen_trans, all_transitions_of_walk)
        if len(idx_of_trans) == 0: return False

        for possible_idx in idx_of_trans:
            pre_states_from_walk = _curr_walk[:possible_idx]
            post_states_from_walk = _curr_walk[possible_idx + 2:]

            is_possible_pre = self.is_ordered_states_in_sequence(pre_states_from_trajectory, pre_states_from_walk) or (
                        (len(pre_states_from_trajectory) == 0) and (len(pre_states_from_walk) == 0))

            is_possible_post = self.is_ordered_states_in_sequence(post_states_from_trajectory, post_states_from_walk) or (
                        (len(post_states_from_trajectory) == 0) and (len(post_states_from_walk) == 0))

            if is_possible_pre and is_possible_post:
                return True
        return False

    def get_all_seen_transitions(self,_curr_w, _curr_walk):
        all_transitions_of_walk = [(_f, _t) for _f, _t in zip(_curr_walk, _curr_walk[1:])]

        seen_states = [_curr_walk[i] for i in _curr_w]
        optional_seen_transitions = [(_f, _t) for _f, _t in zip(seen_states, seen_states[1:])]

        seen_transitions = []
        for optional_seen_transitions_idx in range(len(optional_seen_transitions)):
            opt_seen_trans = optional_seen_transitions[optional_seen_transitions_idx]

            pre_states_from_trajectory = seen_states[:optional_seen_transitions_idx]
            post_states_from_trajectory = seen_states[optional_seen_transitions_idx + 2:]

            if self.is_optional_transition_is_seen(opt_seen_trans, pre_states_from_trajectory, post_states_from_trajectory
                    , all_transitions_of_walk, _curr_walk):
                seen_transitions.append(opt_seen_trans)
        return seen_transitions

    def _exrect_transitions_from_walk(self,curr_walk,states,Ws):
        '''
        :param curr_walk:
        :param N:
        :param d:
        :param Ws: we take W because we want to sample transitions between two "nulls" differently
        :return:
        '''
        _transitions = {state: {} for state in states}
        transitions = transitions_dict.create_from_another_dict(_transitions)

        for W,walk in zip(Ws,curr_walk) :
            for time in range(len(walk)-1) :
                is_null_transition = not ((time in W) and ((time + 1) in W))
                transitions.update_with_none(walk[time], walk[time+1], value = 1, is_null = is_null_transition)

            seen_transitions = self.get_all_seen_transitions(W,walk)
            transitions.update_extended_seen_transitions(seen_transitions)

        return transitions

    def __swipe_dict_key_value(self,dict):
        result = {}
        [result.setdefault(i[1], []).append(i[0]) for i in list(dict.items())]
        return  result

    def __sample_dist_from_params(self,mean,sigma):
        dist = pomegranate.distributions.NormalDistribution(mean,sigma)
        return dist.sample()

    def __sample_dist_from_params_known_emissions(self,known_emissions) :
        states = list(known_emissions.keys())
        probs = list(known_emissions.values())
        return np.random.choice(states, 1, p=probs)

    def __calc_initial_states_count(self,curr_walk,state_to_distrbution_mapping):
        if type(state_to_distrbution_mapping) is dict :
            all_states = state_to_distrbution_mapping.keys()
        else :
            all_states = [s for s in state_to_distrbution_mapping if s not in ['start','end']]

        initial_states_count = dict(Counter(itertools.chain(*curr_walk)))
        non_seen_states = {state:0 for state in all_states if state not in initial_states_count.keys()}
        return {**initial_states_count,**non_seen_states}

    def _inner_exrect_samples_from_walk(self,curr_mus,sigmas,states,samples_data) :
        traj, w, walk  = samples_data
        initial_observations_sum = {state : 0 for state in states}
        ind_obs = 0
        for time_ind in range(len(walk)):
            walked_state = walk[time_ind]
            if time_ind in w:
                observation = traj[ind_obs]
                ind_obs += 1
            else:
                observation = self.__sample_dist_from_params(curr_mus[walked_state], sigmas[walked_state])

            initial_observations_sum[walked_state] = initial_observations_sum[walked_state] + observation
        return initial_observations_sum

    def _inner_exrect_samples_from_walk_known_emmisions(self,known_emissions,states,samples_data) :
        traj, w, walk  = samples_data
        initial_observations_sum = {state : 0 for state in states}
        ind_obs = 0
        for time_ind in range(len(walk)):
            walked_state = walk[time_ind]
            if time_ind in w:
                observation = traj[ind_obs]
                ind_obs += 1
            else:
                observation = self.__sample_dist_from_params_known_emissions(known_emissions[walked_state])

            initial_observations_sum[walked_state] = initial_observations_sum[walked_state] + observation
        return initial_observations_sum

    def __agg_list_of_dict(self,dicts_list) :
        agg_dict = {}

        for _dict in dicts_list :
            for k,v in _dict.items():
                if k in agg_dict.keys() :
                    agg_dict[k] += v
                else :
                    agg_dict[k] = v
        return agg_dict

    def _exrect_samples_from_walk(self,curr_walk,all_relvent_observations,curr_w,state_to_distrbution_mapping, curr_mus,sigmas) :
        # first we just count the number of emissions from each states, for now regardless the tied states. states count :
        initial_states_count = self.__calc_initial_states_count(curr_walk,state_to_distrbution_mapping)
        distrbution_to_states_mapping = self.__swipe_dict_key_value(state_to_distrbution_mapping)

        # now we calculate the observations sum because we want to keep using the paper instructions .
        # sum before tie states :
        states = list(initial_states_count.keys())

        samples_data = [(traj,curr_w[traj_ind],curr_walk[traj_ind]) for traj_ind,traj in enumerate(all_relvent_observations)]
        _partial__exrect_samples_from_walk = partial(self._inner_exrect_samples_from_walk, curr_mus, sigmas, states)

        with Pool(base_config.n_cores) as p :
            all_initial_observations_sum = p.map(_partial__exrect_samples_from_walk,samples_data)

        initial_observations_sum = self.__agg_list_of_dict(all_initial_observations_sum)

        # we remember to count over the tied states. observations sum and count :
        dists_sum = {}
        dists_count = {}
        for dist,tied_states in distrbution_to_states_mapping.items():
            dists_sum[dist] = sum([initial_observations_sum[tied_state] for tied_state in tied_states if tied_state not in ['start','end']])
            dists_count[dist] = sum([initial_states_count[tied_state] for tied_state in tied_states if tied_state not in ['start','end']])

        states_count_matrix = {}
        observations_sum_matrix = {}

        for _state,_dist in state_to_distrbution_mapping.items() :
            states_count_matrix[_state] = dists_count[_dist]
            observations_sum_matrix[_state] = dists_sum[_dist]


        return states_count_matrix,observations_sum_matrix

    def _exrect_samples_from_walk_known_emissions(self,states,curr_walk,all_relvent_observations,curr_w,known_emissions) :
        # first we just count the number of emissions from each states, for now regardless the tied states. states count :
        states_count = self.__calc_initial_states_count(curr_walk,states)

        samples_data = [(traj,curr_w[traj_ind],curr_walk[traj_ind]) for traj_ind,traj in enumerate(all_relvent_observations)]
        _partial__exrect_samples_from_walk = partial(self._inner_exrect_samples_from_walk_known_emmisions, known_emissions, states)

        with Pool(base_config.n_cores) as p :
            all_initial_observations_sum = p.map(_partial__exrect_samples_from_walk,samples_data)

        observations_sum = self.__agg_list_of_dict(all_initial_observations_sum)

        return states_count,observations_sum

    def impute_emissions_table_with_zeros(self, emissions_table,all_relvent_observations) :
        all_possible_emissions = set(chain(*all_relvent_observations))
        new_emissions_table = emissions_table
        for _emm in all_possible_emissions :
            for k,v in emissions_table.items():
                if _emm not in v.keys() :
                    new_emissions_table[k][_emm] = 0
        return new_emissions_table
    # endregion

if __name__ == '__main__':
    pass
    # N = 10  # chain length
    # d = 5  # possible states
    #
    # number_of_smapled_traj = 100
    # p_prob_of_observation = 0.7
    #
    # simulator = Simulator_for_Gibbs(N,d,number_of_smapled_traj)
    # #all_relvent_observations,sigmas,pome_model = simulator.simulate_observations()
    #
    # sampler = GibbsSampler(N,d)
    # sampler.sample(all_relvent_observations, sigmas, 50)
