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
import snakeviz


class Simulator_for_Gibbs():
    def __init__(self,N,d,n_states,easy_mode=False,max_number_of_sampled_traj = None ):
        self.N = N
        self.d = d
        self.n_states = n_states
        self.states_known_mues = None
        self.states_known_sigmas = None
        self.max_number_of_sampled_traj = max_number_of_sampled_traj

        self.known_Ws = []
        self.pre_sampled_traj = None

        if not easy_mode :
            self.mues = np.round(np.random.rand(1,n_states+1)*10,2)
        else :
            self.mues = np.array(range(1,n_states+1))

        #TODO: replace with std from outside
        self.sigmas = np.ones_like(self.mues)/10

    @staticmethod
    def _sample_n_points_from_traj(full_sample,_n):
        ws = sorted(np.random.choice(range(len(full_sample)), _n, replace=False))
        return ([full_sample[i] for i in ws],ws)

    def _sample_single_traj_from_pome_model(self,pome_model,return_emissions = True):
        long_sampled = pome_model.sample(path=True) # sample with inner loops
        sampled_emissions = long_sampled[0]
        sampled_states = [ss.name for ss in long_sampled[1]]

        if return_emissions :
            _sampled = [sampled_emissions[i-1] for i in range(1,len(sampled_states)-1) if
                        sampled_states[i] != sampled_states[i-1]] # remove consucutive duplications
        else :
            _sampled = [sampled_states[i] for i in range(1, len(sampled_states) - 1) if
                        sampled_states[i] != sampled_states[i - 1]]  # remove consucutive duplications

        return _sampled

    def __build_known_mues_and_sigmes_to_state_mapping(self, mues,sigmas, state_to_distrbution_mapping):
        _known_mues_to_state = np.zeros((self.d,self.N))
        _known_sigmes_to_state = np.zeros((self.d, self.N))
        for in_time,time in itertools.product(range(self.d),range(self.N)) :
            _known_mues_to_state[in_time,time] = mues[state_to_distrbution_mapping[in_time,time]]
            _known_sigmes_to_state[in_time,time] = sigmas[state_to_distrbution_mapping[in_time,time]]

        return _known_mues_to_state,_known_sigmes_to_state

    def _sample_N_traj_from_pome_model(self,pome_model,N):
        _traj = []
        _traj_states = []
        for i in range(N) :
            __t = self._sample_single_traj_from_pome_model(pome_model,True)
            __s = self._sample_single_traj_from_pome_model(pome_model, False)

            _traj.append(__t)
            _traj_states.append(__s)

        return _traj,_traj_states

    @staticmethod
    def sample_traj_for_few_obs(N, p_prob_of_observation,number_of_smapled_traj,all_full_sampled_trajs):

        binom_dist = binom(N, p_prob_of_observation)
        n_of_obs_per_traj = binom_dist.rvs(number_of_smapled_traj)
        #TODO: is this assumption valid ? we have more 2 then random
        n_of_obs_per_traj[n_of_obs_per_traj < 2] = 2

        all_relvent_observations_and_ws = list(
            map(lambda x: Simulator_for_Gibbs._sample_n_points_from_traj(x[0], x[1]), zip(all_full_sampled_trajs, n_of_obs_per_traj)))

        all_relvent_observations = [ro[0] for ro in all_relvent_observations_and_ws]
        all_ws = [ro[1] for ro in all_relvent_observations_and_ws]

        return all_relvent_observations, all_ws

    def simulate_observations(self,pome_model,p_prob_of_observation,number_of_smapled_traj,from_pre_sampled_traj = False):
        if from_pre_sampled_traj :
            if self.pre_sampled_traj is not None :
                all_full_sampled_trajs_max_len = self.pre_sampled_traj
                all_full_sampled_trajs_states_max_len = self.pre_sampled_traj_states
                all_full_sampled_trajs = all_full_sampled_trajs_max_len[:number_of_smapled_traj]
                all_full_sampled_trajs_states = all_full_sampled_trajs_states_max_len[:number_of_smapled_traj]
            else :
                # all_full_sampled_trajs_max_len = [self._sample_single_traj_from_pome_model(pome_model)
                #                                   for _ in range(self.max_number_of_sampled_traj)]

                all_full_sampled_trajs_max_len,all_full_sampled_trajs_states_max_len = \
                    self._sample_N_traj_from_pome_model(pome_model,self.max_number_of_sampled_traj)
                self.pre_sampled_traj = all_full_sampled_trajs_max_len
                self.pre_sampled_traj_states = all_full_sampled_trajs_states_max_len

                all_full_sampled_trajs = all_full_sampled_trajs_max_len[:number_of_smapled_traj]
                all_full_sampled_trajs_states = all_full_sampled_trajs_states_max_len[:number_of_smapled_traj]
        else :
            # all_full_sampled_trajs = [self._sample_single_traj_from_pome_model(pome_model) for _ in
            #                           range(number_of_smapled_traj)]
            all_full_sampled_trajs,all_full_sampled_trajs_states = self._sample_N_traj_from_pome_model(pome_model,number_of_smapled_traj)

        all_relvent_observations,all_ws = Simulator_for_Gibbs.sample_traj_for_few_obs(self.N, p_prob_of_observation,
                                                                                      number_of_smapled_traj,all_full_sampled_trajs)
        all_relvent_sampled_trajs_states = list(map(lambda x: [x[0][i] for i in x[1]], zip(all_full_sampled_trajs_states, all_ws)))
        self.known_Ws = all_ws

        return all_relvent_observations,all_full_sampled_trajs,all_full_sampled_trajs_states,all_relvent_sampled_trajs_states

    def build_transition_prob_from_known(self,pome_model):
        known_transitions_summary = {}
        sampled_walks =  [self._sample_single_traj_from_pome_model(pome_model,return_emissions=False) for i in range(5000)]

        for sampled_walk in sampled_walks :
            for _from,_to in zip(sampled_walk,sampled_walk[1:]) :
                if ('start' in _from) or ('start' in _to) or ('end' in _from) or ('end' in _to) :
                    continue

                if _from in known_transitions_summary.keys() :
                    if _to in known_transitions_summary[_from].keys() :
                        known_transitions_summary[_from][_to] = known_transitions_summary[_from][_to] + 1
                    else :
                        known_transitions_summary[_from][_to] = 1
                else :
                    known_transitions_summary[_from] = {_to:1}

        return known_transitions_summary

    def build_pome_model(self,N, d, mues, sigmas,is_mutual_distrbutions = True):
        '''

        :param N:
        :param d:
        :param mues:
        :param sigmas:
        :param is_mutual_distrbutions: if True, the number of distrbutions is smaller then the number of states (tied states)
        :return:
        '''

        # initial hmms
        model = HiddenMarkovModel()
        topological_pome_model = HiddenMarkovModel()

        # build the distrbutions and shuffale them
        all_distrbutions = [NormalDistribution(mu, sigmas[i]) for i, mu in enumerate(mues)]
        all_empty_distrbutions = [NormalDistribution(mu, sigmas[i]) for i, mu in enumerate(mues)]

        # add states to bouth hmms
        print("change fo chack")
        if not is_mutual_distrbutions:
            _picked_dists_inds_for_non_mutual_case = np.random.choice(range(len(all_distrbutions)), size=N * d,
                                                                      replace=False)

        state_to_distrbution_mapping = {}
        all_pome_states = {}
        all_empty_states = {}
        dist_ind = 0

        for time_ind in range(N):
            if is_mutual_distrbutions:
                _picked_dists_inds_for_mutual_case = np.random.choice(range(len(all_distrbutions)), size=d,
                                                                      replace=False)

            for in_time_ind in range(d):
                if is_mutual_distrbutions:
                    _dist = all_distrbutions[_picked_dists_inds_for_mutual_case[in_time_ind]]
                    _empty_dist = all_empty_distrbutions[_picked_dists_inds_for_mutual_case[in_time_ind]]
                else:
                    _dist = all_distrbutions[_picked_dists_inds_for_non_mutual_case[dist_ind]]
                    _empty_dist = all_empty_distrbutions[_picked_dists_inds_for_non_mutual_case[dist_ind]]
                    dist_ind += 1

                # we need that so we can know which states are tied
                state_to_distrbution_mapping[(in_time_ind, time_ind)] = _picked_dists_inds_for_mutual_case[in_time_ind]

                state = State(_dist, name=f"({in_time_ind},{time_ind})")
                empty_state = State(_empty_dist, name=f"({in_time_ind},{time_ind})")

                all_pome_states[(in_time_ind, time_ind)] = state
                all_empty_states[(in_time_ind, time_ind)] = empty_state

                model.add_state(state)
                topological_pome_model.add_state(empty_state)

        # add transitions to known model
        for time_ind in range(N - 1):
            for in_time_ind in range(d):
                n_of_out_trans = np.random.randint(2, d - 1)
                out_trans = np.random.choice(d, size=n_of_out_trans, replace=False)

                for out_t in out_trans:
                    # print(f"({in_time_ind},{time_ind})=>({out_t},{time_ind+1})")
                    _trans_val = (np.random.rand() + 0.13) / 1.3
                    model.add_transition(all_pome_states[in_time_ind, time_ind], all_pome_states[out_t, time_ind + 1],
                                         _trans_val)

        for in_time_ind in range(d):
            model.add_transition(model.start, all_pome_states[in_time_ind, 0], 1)
            model.add_transition(all_pome_states[in_time_ind, N - 1], model.end, 1)

            topological_pome_model.add_transition(model.start, all_empty_states[in_time_ind, 0], 1)
            topological_pome_model.add_transition(all_empty_states[in_time_ind, N - 1], model.end, 1)

        # add all possible transitions to empty(topological) model
        for time_ind in range(N - 1):
            for in_time_ind_from in range(d):
                for in_time_ind_to in range(d):
                    topological_pome_model.add_transition(all_empty_states[in_time_ind_from, time_ind],
                                                          all_empty_states[in_time_ind_to, time_ind + 1], 1)

        model.bake()
        topological_pome_model.bake()
        
        return model, all_pome_states,topological_pome_model,state_to_distrbution_mapping

    def update_known_mues_and_sigmes_to_state_mapping(self,state_to_distrbution_mapping) :
        self.states_known_mues, self.states_known_sigmas = self.__build_known_mues_and_sigmes_to_state_mapping(
            self.mues, self.sigmas,
            state_to_distrbution_mapping)
