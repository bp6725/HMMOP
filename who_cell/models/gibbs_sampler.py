import pomegranate
from pomegranate import *
import numpy as np
import math
import matplotlib.pyplot as plt
from functools import partial
import itertools
import random
import copy
import pandas as pd
import itertools
import networkx as nx
import holoviews as hv
from IPython.display import display, HTML,clear_output
from scipy.stats import binom
from numba import jit
import numba
from tqdm import tqdm
from collections import Counter
import random
import sys
from who_cell.transitions_dict import transitions_dict
from functools import lru_cache

random.seed(10)


from who_cell.simulation.simulation_for_gibbs import Simulator_for_Gibbs

class GibbsSampler() :
    def __init__(self,length_of_chain,number_of_states_in_time,mues_for_sampler = None,transition_sampling_profile = 'all'):
        self.N = length_of_chain
        self.d = number_of_states_in_time

        self.transition_sampling_profile = transition_sampling_profile

        self.is_known_mues = mues_for_sampler is not None
        if mues_for_sampler is not None :
            self.mues_for_sampler = mues_for_sampler

    # region Public

    def sample(self,is_acyclic, all_relvent_observations, pome_results, sigmas, Ng_iters, w_smapler_n_iter = 500):
        state_to_distrbution_param_mapping = pome_results["state_to_distrbution_param_mapping"]
        state_to_distrbution_mapping = pome_results["state_to_distrbution_mapping"]
        start_probs = pome_results["start_probabilites"]
        states = list(state_to_distrbution_mapping.keys())

        # TODO : why we dont know is_known_mues in this function ? if we know mues whay we need thw params ?
        #TODO : send isacyclic - if cyclic think of another way to calculate, if not validate the previous case
        chi,kapa = self._calc_distributions_prior(all_relvent_observations, self.N)
        curr_mus = self.build_initial_mus(sigmas,chi, kapa)
        curr_trans = self.build_initial_transitions(states,is_acyclic)
        if is_acyclic :
            curr_w = [list(range(len(obs))) for obs in all_relvent_observations]
        else :
            curr_w = [sorted(np.random.choice(range(self.N), len(obs), replace=False)) for obs in all_relvent_observations]

        state_to_distrbution_param_mapping = self._update_distributions_params(state_to_distrbution_param_mapping, curr_mus)
        curr_walk = self.sample_walk_from_params(is_acyclic,all_relvent_observations,self.N, state_to_distrbution_param_mapping,start_probs,
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
                curr_mus = self.sample_mus_from_params(sampled_states, observations_sum, chi, kapa, sigmas)
                curr_trans = self.sample_trans_from_params(sampled_transitions,states)
                curr_ws = self.sample_ws_from_params(all_relvent_observations, curr_walk,curr_mus,sigmas,self.N, n_iters=w_smapler_n_iter)

                state_to_distrbution_param_mapping = self._update_distributions_params(state_to_distrbution_param_mapping,curr_mus)
                curr_walk = self.sample_walk_from_params(is_acyclic,all_relvent_observations,self.N,
                                                         state_to_distrbution_param_mapping,start_probs,
                                                         curr_w, curr_trans)

                sampled_transitions = self._exrect_transitions_from_walk(curr_walk,states,curr_w)
                sampled_states,observations_sum = self._exrect_samples_from_walk(curr_walk,all_relvent_observations,
                                                                                 curr_w,state_to_distrbution_param_mapping,
                                                                                 curr_mus,sigmas)

                all_sampled_transitions.append(sampled_transitions)
                all_transitions.append(curr_trans)
                all_states.append(sampled_states)
                all_observations_sum.append(observations_sum)
                all_mues.append(curr_mus)
                all_ws.append(curr_ws)
                pbar.update(1)
        return all_states,all_observations_sum, all_sampled_transitions,all_mues,all_ws,all_transitions

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

    def _calc_distributions_prior(self,all_relvent_observations,N):
        weighted_obs = self._calc_weighted_obs_for_init(all_relvent_observations, N)
        chi = (weighted_obs.max(axis=0) + weighted_obs.min(axis=0)) / 2
        kapa = 1 / ((weighted_obs.max(axis=0) - weighted_obs.min(axis=0)) ** 2)

        return chi, kapa

    def _update_distributions_params(self, state_to_distrbution_param_mapping, curr_mus):
        new_state_to_distrbution_param_mapping = {}
        for state,params in state_to_distrbution_param_mapping.items():
            if ('start' in state) or ('end' in state): continue
            _curr_mu = curr_mus[state]
            new_params = (_curr_mu,params[1])

            new_state_to_distrbution_param_mapping[state] = new_params
        return new_state_to_distrbution_param_mapping

    @lru_cache(225)
    def _prob_for_assigment(self,time, obs_ind, N, n_obs):
        p_prob_of_observation = n_obs / N

        _pre_prob = binom.pmf(obs_ind, time, p_prob_of_observation)
        _post_prob = binom.pmf(n_obs - obs_ind, N - time, p_prob_of_observation)

        return _pre_prob * _post_prob

    def _calc_weighted_obs_for_init(self,sampled_trajs, N):
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

    def sample_cond_prob_single_dim(self,k,N,dims_vector, ind_dim_for_sample, msf, recursion_msf,y_from_x_probs):
        _pre_value = dims_vector[(ind_dim_for_sample - 1)] if ind_dim_for_sample != 0 else -1
        _post_value = dims_vector[(ind_dim_for_sample + 1)] if (ind_dim_for_sample != (k - 1)) else N

        possible_options_for_dim = range(_pre_value + 1, _post_value)
        probs_of_opts = [y_from_x_probs[(ind_dim_for_sample,poss_opt)] for poss_opt in possible_options_for_dim]
        probs_of_opts = probs_of_opts if sum(probs_of_opts) != 0 else np.array([1 for i in probs_of_opts])

        return np.random.choice(possible_options_for_dim, p=probs_of_opts/sum(probs_of_opts))

    def sample_msf_using_sim(self,msf,k,N, n_iter,y_from_x_probs, recursion_msf=False):
        initial_vector = sorted(random.sample(range(N), k))
        all_sampled_full_dims = []

        res_samples_per_dim = np.zeros((N, k))
        _curr_dim_vector = copy.copy(initial_vector)
        # with tqdm(total=n_iter) as pbar:
        for _ in range(n_iter):
            for dim in range(k):
                _sample = self.sample_cond_prob_single_dim(k,N,_curr_dim_vector, dim, msf, recursion_msf,y_from_x_probs)
                _curr_dim_vector[dim] = _sample
                res_samples_per_dim[_sample, dim] += 1
                all_sampled_full_dims.append(copy.copy(_curr_dim_vector))
                # pbar.update(1)

        return all_sampled_full_dims[-1]

    def _build_states_map(self, known_pome_model):
        states = {}
        for state in known_pome_model.states:
            if ("start" in state.name ) or ('end' in state.name) :
                continue
            state_name = eval(state.name)
            states[state_name] = state

        return states

    def __probability_from_dist_params(self,observation,params) :
        if observation is None :
            return 1
        dist = pomegranate.NormalDistribution(params[0],params[1])
        return dist.probability(observation)

    def _build_emmisions_for_sample(self,is_acyclic,sample, w, state_to_distrbution_param_mapping,N,normalized_emm =  True):
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
                    _emm = self.__probability_from_dist_params(observation, state_to_distrbution_param_mapping[state])

                #if not cyclic case : if the start is out of time is zero
                if (not is_acyclic) and (state[1] != time_ind) :
                    _emm = 0

                emmisions[(state, time_ind)] = _emm
                _sum += _emm

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

    def __sample_emmisions_matrix(self,emm_prob,state,observation_ind,is_acyclic):
        if (not is_acyclic) :
            return emm_prob[(state,observation_ind)]  if state[1] == observation_ind else 0
        else :
            return emm_prob[(state,observation_ind)]

    def _sample_trans_matrix(self, trans_prob, from_state, to_state,is_acyclic):
        if ((not is_acyclic) and ((to_state[1] - from_state[1]) != 1)) : return 0
        if from_state not in trans_prob.keys() : return 0
        if to_state not in trans_prob[from_state].keys() : return 0

        return trans_prob[from_state][to_state]

    def __sample_single_time(self,is_acyclic,prev_state, walk, fwd, trans_prob, n) :
        if n == -1 :
            return walk

        states_of_time = []
        prob_of_states_of_time = []

        for state,prob in fwd[n].items() :
            if ((not is_acyclic) and (state[1] != n)) :
                continue

            prob_for_sample = prob * self._sample_trans_matrix(trans_prob, state, prev_state,is_acyclic)

            states_of_time.append(state)
            prob_of_states_of_time.append(prob_for_sample)

        #TODO: maybe we should remove zero chance transitions ?
        prob_of_states_of_time = prob_of_states_of_time if sum(prob_of_states_of_time) != 0 else [1 for i in prob_of_states_of_time]

        _sum = sum(prob_of_states_of_time)
        sampled_state_inde = np.random.choice(range(len(states_of_time)),
                                              p = [p/_sum for p in prob_of_states_of_time] )
        sampled_state = states_of_time[sampled_state_inde]

        walk.append(sampled_state)

        return self.__sample_single_time(is_acyclic,sampled_state,walk,fwd, trans_prob, n - 1)

    def _build_single_walk_from_postrior(self,is_acyclic,fwd,trans_prob,N) :
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

        _walk  = self.__sample_single_time(is_acyclic,sampled_state,walk,fwd,trans_prob,N-2)
        walk = list(reversed(_walk))

        return walk

    def _fwd_bkw(self,is_acyclic, states, start_prob, trans_prob, emm_prob,N,only_forward = False):
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
                    prev_f_sum = sum([f_prev[k] * self._sample_trans_matrix(trans_prob, k, st,is_acyclic) for k in states if f_prev[k] !=0])

                if emm_prob is not None :
                    # F-B case
                    f_curr[st] = self.__sample_emmisions_matrix(emm_prob,st,observation_i,is_acyclic) * prev_f_sum
                else :
                    # when we want to calculate only transitions (deviation of prob from expected sample count)
                    _emm = int(st[1] == observation_i) if (not is_acyclic) else 1
                    f_curr[st] = _emm * prev_f_sum

            fwd.append(f_curr)
            f_prev = f_curr

        if only_forward :
            return fwd

        # Backward part of the algorithm
        return self._build_single_walk_from_postrior(is_acyclic,fwd,trans_prob, N)

    def sample_mus_from_params(self,all_sampled_states, sum_relvent_observations, ξ, κ,  sigmas):
        if self.is_known_mues :
            return self.mues_for_sampler

        sis = sum_relvent_observations
        nis = all_sampled_states

        _mues = (sis + ξ * κ * sigmas) / (κ * sigmas + nis)
        _sigs = (sigmas) / (κ * sigmas + nis)

        return np.random.normal(_mues, _sigs)

    def sample_trans_from_params(self,all_transitions,states):
        sampled_transitions = {state: {} for state in states}

        if self.transition_sampling_profile == 'all' :
            _transition_dict = all_transitions.items()

        if self.transition_sampling_profile == 'observed':
            _transition_dict = all_transitions.observed_transitions_dict.items()

        for state,poss_trans in _transition_dict:
            poss_trans_states = [state for state in poss_trans.keys()]
            poss_trans_counts = [state for state in poss_trans.values()]

            sample = np.random.dirichlet(poss_trans_counts)

            for _state_ind,_state in enumerate(poss_trans_states):
                sampled_transitions[state][_state] = sample[_state_ind]

        return sampled_transitions

    def __prob_obs_for_state(self,state,obs,curr_mus,sigmas):
        return pomegranate.distributions.NormalDistribution(curr_mus[state], sigmas[state]).probability(obs)

    def _extract_relevent_probs_from_walk(self, traj, walk, curr_mus, sigmas) :
        y_from_x_probs = {(obs,state): self.__prob_obs_for_state(walk[state],traj[obs],curr_mus,sigmas) for
                          obs,state in itertools.product(range(len(traj)), range(len(walk))) }

        return y_from_x_probs

    def sample_ws_from_params(self,sampled_trajs, curr_walk,curr_mus,sigmas,N, n_iters=500):
        result_per_traj = []
        for traj_ind, traj in enumerate(sampled_trajs):
            y_from_x_probs = self._extract_relevent_probs_from_walk(traj, curr_walk[traj_ind], curr_mus, sigmas)

            seq_length = max(N,len(curr_walk[traj_ind]))
            msf = self.msf_creator(y_from_x_probs,seq_length)

            _simulted_w = self.sample_msf_using_sim(msf,len(traj),seq_length, n_iters,y_from_x_probs, False)
            result_per_traj.append(_simulted_w)
        return result_per_traj

    def sample_walk_from_params(self,is_acyclic, sampled_trajs,N, state_to_distrbution_param_mapping,
                                start_prob, curr_ws, curr_trans):
        walks = []
        for i, sample in enumerate(sampled_trajs):
            seq_length = max(len(sample),N)
            emmisions = self._build_emmisions_for_sample(is_acyclic,sample,
                                                         curr_ws[i], state_to_distrbution_param_mapping,seq_length)
            posterior = self._fwd_bkw(is_acyclic,state_to_distrbution_param_mapping.keys(),
                                      start_prob, curr_trans, emmisions,seq_length)
            walks.append(posterior)

        return walks

    def build_initial_transitions(self,states,is_acyclic):
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
                        _val = int(_s2[1] == 1)
                    elif _s2 == 'end':
                        _val = 1
                    else :
                        _val =  1  if ((_s2[1] - _s1[1]) == 1) else 0
                else :
                    _val = 1

                initial_transitions[_s1][_s2] = _val
                _sum += _val
            normalization_factors[_s1] = _sum

        initial_transitions = {_s1:{__s1:__s2/normalization_factors[_s1] for __s1,__s2 in s2.items()} for _s1,s2 in initial_transitions.items()}
        return initial_transitions

    def build_initial_mus(self,mues,ξ, κ):
        if self.is_known_mues :
            return self.mues_for_sampler

        return np.random.randn(self.d, self.N) * np.sqrt(np.array(κ)) + np.array(ξ)
        # endregion

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
                is_null_transition =  not ((time in W) and ((time + 1) in W))
                transitions.update_with_none(walk[time],walk[time+1],value = 1,is_null = is_null_transition)
        return transitions

    def __swipe_dict_key_value(self,dict):
        result = {}
        [result.setdefault(i[1], []).append(i[0]) for i in list(dict.items())]
        return  result

    def __sample_dist_from_params(self,mean,sigma):
        dist = pomegranate.distributions.NormalDistribution(mean,sigma)
        return dist.sample()

    def __calc_initial_states_count(self,curr_walk,state_to_distrbution_mapping):
        all_states = state_to_distrbution_mapping.keys()
        initial_states_count = dict(Counter(itertools.chain(*curr_walk)))
        non_seen_states = {state:0 for state in all_states if state not in initial_states_count.keys()}
        return {**initial_states_count,**non_seen_states}

    def _exrect_samples_from_walk(self,curr_walk,all_relvent_observations,curr_w,state_to_distrbution_mapping, curr_mus,sigmas) :
        # first we just count the number of emissions from each states, for now regardless the tied states. states count :
        initial_states_count = self.__calc_initial_states_count(curr_walk,state_to_distrbution_mapping)
        distrbution_to_states_mapping = self.__swipe_dict_key_value(state_to_distrbution_mapping)

        # now we calculate the observations sum because we want to keep using the paper instructions .
        # sum before tie states :
        initial_observations_sum = {state : 0 for state in initial_states_count.keys()}
        for traj_ind,traj in enumerate(all_relvent_observations) :
            w = curr_w[traj_ind]
            walk = curr_walk[traj_ind]

            ind_obs = 0
            for time_ind in range(len(walk)):
                walked_state = walk[time_ind]
                if time_ind in w:
                    observation = traj[ind_obs]
                    ind_obs += 1
                else:
                    observation = self.__sample_dist_from_params(curr_mus[walked_state],sigmas[walked_state])

                initial_observations_sum[walked_state] = initial_observations_sum[walked_state] + observation

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
