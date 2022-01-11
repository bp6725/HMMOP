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
from collections import Counter

from who_cell.simulation.simulation_for_gibbs import Simulator_for_Gibbs

class GibbsSampler() :
    def __init__(self,length_of_chain,number_of_states_in_time = None,
                 transition_sampling_profile = 'all', multi_process = True):
        self.N = length_of_chain

        self.transition_sampling_profile = transition_sampling_profile

        self.multi_process = multi_process

    # region Public

    def sample(self, all_relvent_observations, start_probs,
               known_mues, sigmas, Ng_iters, w_smapler_n_iter=100, N=None, is_mh=False):
        # (all_relvent_observations,known_states) = all_relvent_observations

        N = self.N if N is None else N
        states = list(set(list(start_probs.keys()) + ['start', 'end']))
        state_to_distrbution_param_mapping = self.__build_initial_state_to_distrbution_param_mapping(known_mues, sigmas,
                                                                                                     states)

        priors = self._calc_distributions_prior(all_relvent_observations, (len(states) - 2)) if not known_mues else None
        curr_mus = self.build_initial_mus(sigmas, priors, known_mues)
        curr_trans = self.build_initial_transitions(states)

        if type(N) is list:
            curr_w = [sorted(np.random.choice(range(max(_N, len(obs))), len(obs), replace=False)) for obs, _N in
                      zip(all_relvent_observations, N)]
        else:
            curr_w = [sorted(np.random.choice(range(max(N, len(obs))), len(obs), replace=False)) for obs in
                      all_relvent_observations]

        state_to_distrbution_param_mapping = self._update_distributions_params(state_to_distrbution_param_mapping,
                                                                               curr_mus)
        curr_walk, alpha = self.sample_walk_from_params(all_relvent_observations, N, state_to_distrbution_param_mapping,
                                                        start_probs,
                                                        curr_w, curr_trans)

        sampled_states, observations_sum = self._exrect_samples_from_walk(curr_walk, all_relvent_observations, curr_w,
                                                                          state_to_distrbution_param_mapping,
                                                                          curr_mus, sigmas)
        sampled_transitions = self._exrect_transitions_from_walk(curr_walk, states, curr_w)

        all_sampled_transitions = [sampled_transitions]
        all_transitions = [curr_trans]
        all_states = [sampled_states]
        all_observations_sum = [observations_sum]
        all_mues = [curr_mus]
        all_ws = [curr_w]
        with tqdm(total=Ng_iters) as pbar:
            for i in range(Ng_iters):
                curr_mus = self.sample_mus_from_params(sampled_states, observations_sum, priors, sigmas, known_mues)
                state_to_distrbution_param_mapping = self._update_distributions_params(
                    state_to_distrbution_param_mapping, curr_mus)

                curr_trans, _ = self.sample_trans_from_params(sampled_transitions, states,
                                                              curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                           state_to_distrbution_param_mapping],
                                                              stage_name="transitions" if is_mh else "no_mh",
                                                              observations=all_relvent_observations)
                curr_w, _ = self.sample_ws_from_params(all_relvent_observations, curr_walk,
                                                       state_to_distrbution_param_mapping, N,
                                                       n_iters=w_smapler_n_iter,
                                                       curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                    state_to_distrbution_param_mapping],
                                                       stage_name="w" if is_mh else "no_mh",
                                                       observations=all_relvent_observations)
                _states_picked_by_w = [[seq[i] for i in ws] for ws, seq in zip(curr_w, curr_walk)]

                curr_walk, _ = self.sample_walk_from_params(all_relvent_observations, N,
                                                            state_to_distrbution_param_mapping, start_probs,
                                                            curr_w, curr_trans,
                                                            curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                         state_to_distrbution_param_mapping],
                                                            stage_name="walk" if is_mh else "no_mh",
                                                            observations=all_relvent_observations)

                sampled_transitions = self._exrect_transitions_from_walk(curr_walk, states, curr_w)
                sampled_states, observations_sum = self._exrect_samples_from_walk(curr_walk, all_relvent_observations,
                                                                                  curr_w,
                                                                                  state_to_distrbution_param_mapping,
                                                                                  curr_mus, sigmas)

                all_sampled_transitions.append(sampled_transitions)
                all_transitions.append(curr_trans)
                all_states.append(sampled_states)
                all_observations_sum.append(observations_sum)
                all_mues.append(curr_mus)
                all_ws.append(curr_w)
                pbar.update(1)
        return all_states, all_observations_sum, all_sampled_transitions, all_mues, all_ws, all_transitions

    def sample_guess_pc(self, all_relvent_observations, start_probs,
               known_mues,sigmas, Ng_iters, w_smapler_n_iter = 100,PC_guess=None,is_mh = False):
        if PC_guess == "unknown" :
            return self._sample_unknown_pc(all_relvent_observations, start_probs,
                                  known_mues, sigmas, Ng_iters)
        else :
            return self._new_sample_guess_pc(all_relvent_observations, start_probs,
                                            known_mues, sigmas, Ng_iters, PC_guess)

    def probability_over_known_transition(self,known_emissions, missing_sentences, curr_trans, start_probs,
                                          emmisions_params, Ng_iters, w_smapler_n_iter=100, N=None, is_mh=False):
        state_to_distrbution_param_mapping = {state:state for state in start_probs.keys()}

        possible_w = self.sample_W_options(known_emissions, missing_sentences, curr_trans, start_probs,
                                            emmisions_params, Ng_iters, w_smapler_n_iter, N, is_mh)

        full_likelihod = 0
        with tqdm(total=len(missing_sentences)) as pbar:
            for traj_i,missing_traj in enumerate(missing_sentences):
                _possible_w = [ws_option[traj_i] for ws_option in possible_w ]
                for _ws in _possible_w :
                    seq_probs = self._calculate_prob_single_sample(state_to_distrbution_param_mapping,
                                                                    start_probs, curr_trans,
                                                                    (missing_traj, _ws, N))
                    full_likelihod += np.log(seq_probs)
                        # pbar.update(1)
                pbar.update(1)

        return full_likelihod

    def sample_known_W(self, all_relvent_observations, start_probs,
               known_mues,sigmas, Ng_iters,curr_w, w_smapler_n_iter = 100,N=None,is_mh = False):
        print("start known W")
        N = self.N if N is None else N

        states = list(set(list(start_probs.keys()) + ['start','end']))
        start_probs = {state:(1/(len(states)-2)) for state in states}
        state_to_distrbution_param_mapping = self.__build_initial_state_to_distrbution_param_mapping(known_mues,sigmas,
                                                                                                     states)

        priors = self._calc_distributions_prior(all_relvent_observations, (len(states) -2) ) if not known_mues else None
        curr_mus = self.build_initial_mus(sigmas,priors,known_mues)
        curr_trans = self.build_initial_transitions(states)

        state_to_distrbution_param_mapping = self._update_distributions_params(state_to_distrbution_param_mapping, curr_mus)
        curr_walk,alpha= self.sample_walk_from_params(all_relvent_observations,N, state_to_distrbution_param_mapping,start_probs,
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

                curr_walk,_ = self.sample_walk_from_params(all_relvent_observations,N,
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
                all_states.append(curr_walk)
                all_observations_sum.append(observations_sum)
                all_mues.append(curr_mus)
                all_ws.append(curr_w)
                pbar.update(1)
        return all_states,all_observations_sum, all_sampled_transitions,all_mues,all_ws,all_transitions

    def sample_known_emissions(self, all_relvent_observations, start_probs,
                               emissions_table, Ng_iters, w_smapler_n_iter = 100,N = None,is_mh = True):
        # (all_relvent_observations, known_states) = all_relvent_observations
        emissions_table = self.impute_emissions_table_with_zeros(emissions_table,all_relvent_observations)
        N = self.N if N is None else N
        states = list(set(list(start_probs.keys()) + ['start', 'end']))
        GibbsSampler
        curr_trans = self.build_initial_transitions(states)

        if type(N) is list :
            curr_w = [sorted(np.random.choice(range(max(_N, len(obs))), len(obs), replace=False)) for obs,_N in
                      zip(all_relvent_observations,N)]
        else:
            curr_w = [sorted(np.random.choice(range(max(N, len(obs))), len(obs), replace=False)) for obs in
                  all_relvent_observations]

        curr_walk,alpha = self.sample_walk_from_params(all_relvent_observations, N,
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

                curr_walk,alpha2 = self.sample_walk_from_params(all_relvent_observations, N,
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

    def sample_known_emissions_with_pc_guess(self,all_relvent_observations, start_probs,emissions_table,
               Ng_iters, w_smapler_n_iter = 100,PC_guess=None,is_mh = False):
        N = [len(O) for O in all_relvent_observations]

        states = list(set(list(start_probs.keys()) + ['start', 'end']))
        curr_trans = self.build_initial_transitions(states)

        curr_w = [list(range(len(O))) for O in all_relvent_observations]
        curr_walk, alpha = self.sample_walk_from_params(all_relvent_observations, N, emissions_table,
                                                        start_probs,
                                                        curr_w, curr_trans)

        sampled_transitions = self._exrect_transitions_from_walk(curr_walk, states, curr_w)

        w_adj, N_adj = self.build_N_list(curr_walk, curr_w, curr_trans, PC_guess)

        all_sampled_transitions = [sampled_transitions]
        all_transitions = [curr_trans]
        all_ws = [curr_w]
        with tqdm(total=Ng_iters) as pbar:
            for i in range(Ng_iters):
                curr_trans, _ = self.sample_trans_from_params(sampled_transitions, states,
                                                              curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                           emissions_table],
                                                              stage_name="transitions" if is_mh else "no_mh",
                                                              observations=all_relvent_observations)

                curr_walk, _ = self.sample_walk_from_params(all_relvent_observations, N_adj,
                                                            emissions_table, start_probs,
                                                            w_adj, curr_trans,
                                                            curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                         emissions_table],
                                                            stage_name="walk" if is_mh else "no_mh",
                                                            observations=all_relvent_observations)

                curr_w, _ = self.sample_ws_from_params(all_relvent_observations, curr_walk,
                                                       emissions_table, N_adj,
                                                       n_iters=w_smapler_n_iter,
                                                       curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                    emissions_table],
                                                       stage_name="w" if is_mh else "no_mh",
                                                       observations=all_relvent_observations)

                sampled_transitions = self._exrect_transitions_from_walk(curr_walk, states, curr_w)

                w_adj, N_adj = self.build_N_list(curr_walk, curr_w, curr_trans, PC_guess)

                all_sampled_transitions.append(sampled_transitions)
                all_transitions.append(curr_trans)
                all_ws.append(curr_w)
                pbar.update(1)
        return None, None, all_sampled_transitions, None, all_ws, all_transitions

    def sample_known_emissions_known_W(self, all_relvent_observations, start_probs,
                               emissions_table,curr_w, Ng_iters, w_smapler_n_iter = 100,N = None,is_mh = True):
        emissions_table = self.impute_emissions_table_with_zeros(emissions_table,all_relvent_observations)
        N = self.N if N is None else N
        states = list(set(list(start_probs.keys()) + ['start', 'end']))

        curr_trans = self.build_initial_transitions(states)

        curr_walk,alpha = self.sample_walk_from_params(all_relvent_observations, N,
                                                 emissions_table, start_probs,
                                                 curr_w, curr_trans)
        _states_picked_by_w = [[seq[i] for i in ws] for ws, seq in zip(curr_w, curr_walk)]

        sampled_transitions = self._exrect_transitions_from_walk(curr_walk, states, curr_w)

        all_sampled_transitions = [sampled_transitions]
        all_transitions = [curr_trans]
        all_states_picked_by_w = [_states_picked_by_w]
        with tqdm(total=Ng_iters) as pbar:
            for i in range(Ng_iters):
                curr_trans,alpha0 = self.sample_trans_from_params(sampled_transitions, states,
                                                           curr_params =[curr_trans,curr_w,curr_walk,None,emissions_table],
                                                           stage_name="transitions"  if is_mh else "no_mh",
                                                           observations = all_relvent_observations)
                _states_picked_by_w = [[seq[i] for i in ws] for ws, seq in zip(curr_w, curr_walk)]

                curr_walk,alpha2 = self.sample_walk_from_params(all_relvent_observations, N,
                                                         emissions_table, start_probs,
                                                         curr_w, curr_trans,
                                                         curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                      emissions_table],
                                                         stage_name="walk"  if is_mh else "no_mh",
                                                         observations=all_relvent_observations)

                sampled_transitions = self._exrect_transitions_from_walk(curr_walk, states, curr_w)

                all_sampled_transitions.append(sampled_transitions)
                all_transitions.append(curr_trans)
                all_states_picked_by_w.append(_states_picked_by_w)
                pbar.update(1)
        return  all_sampled_transitions, None, all_transitions,all_states_picked_by_w ,None

    def reconstruction_using_pomegranate(self,all_relvent_observations,state_to_distrbution_param_mapping,known_w=None):
        simulator = Simulator_for_Gibbs(None,None,1)

        if known_w is not None:
            all_relvent_observations = [[(obs if idx in ws else None) for idx, obs in enumerate(traj)] for ws, traj in
                                        zip(known_w, all_relvent_observations)]

        states_names = state_to_distrbution_param_mapping.keys()
        uniform_transitions_matrix = {s: {ss: 1 for ss in states_names if ss != 'start'} for s in states_names}

        new_pome_model, all_model_pome_states = simulator._build_pome_model_from_params(
            state_to_distrbution_param_mapping, uniform_transitions_matrix)

        new_pome_model.fit(all_relvent_observations,n_jobs=base_config.n_cores)
        all_transitions = Utils._extrect_states_transitions_dict_from_pome_model(new_pome_model)[0]
        all_transitions = {str(k):{str(kk):vv for kk,vv in v.items()} for k,v in all_transitions.items()}
        return all_transitions

    def sequence_labeling_known_emissions(self,all_relvent_observations,transitions_probs, start_probs,
                               emissions_table, Ng_iters, w_smapler_n_iter = 100,N = None,is_mh = True):
        emissions_table = self.impute_emissions_table_with_zeros(emissions_table, all_relvent_observations)
        N = self.N if N is None else N

        curr_trans = transitions_probs

        if type(N) is list:
            curr_w = [sorted(np.random.choice(range(max(_N, len(obs))), len(obs), replace=False)) for obs, _N in
                      zip(all_relvent_observations, N)]
        else:
            curr_w = [sorted(np.random.choice(range(max(N, len(obs))), len(obs), replace=False)) for obs in
                      all_relvent_observations]

        curr_walk, alpha = self.sample_walk_from_params(all_relvent_observations, N,
                                                        emissions_table, start_probs,
                                                        curr_w, curr_trans)
        _states_picked_by_w = [[seq[i] for i in ws] for ws, seq in zip(curr_w, curr_walk)]

        all_alphas = [alpha]
        all_ws = [curr_w]
        all_states_picked_by_w = [_states_picked_by_w]
        all_walks = []
        with tqdm(total=Ng_iters) as pbar:
            for i in range(Ng_iters):
                curr_w, alpha1 = self.sample_ws_from_params(all_relvent_observations, curr_walk, emissions_table, N,
                                                            n_iters=w_smapler_n_iter,
                                                            curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                         emissions_table],
                                                            stage_name="w" if is_mh else "no_mh",
                                                            observations=all_relvent_observations)
                _states_picked_by_w = [[seq[i] for i in ws] for ws, seq in zip(curr_w, curr_walk)]

                curr_walk, alpha2 = self.sample_walk_from_params(all_relvent_observations, N,
                                                                 emissions_table, start_probs,
                                                                 curr_w, curr_trans,
                                                                 curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                              emissions_table],
                                                                 stage_name="walk" if is_mh else "no_mh",
                                                                 observations=all_relvent_observations)

                all_walks.append(curr_walk)
                all_alphas.append(np.mean([ alpha1, alpha2]))
                all_states_picked_by_w.append(_states_picked_by_w)
                all_ws.append(curr_w)
                pbar.update(1)
        return  all_ws, all_states_picked_by_w, all_alphas,all_walks

    # endregion

    # region Private

    def _sample_guess_pc(self, all_relvent_observations, start_probs,
                        known_mues, sigmas, Ng_iters, w_smapler_n_iter=100, PC_guess=None, is_mh=False):
        print("start sampler with sampling N")
        N = [len(O) for O in all_relvent_observations]

        states = list(set(list(start_probs.keys()) + ['start', 'end']))
        state_to_distrbution_param_mapping = self.__build_initial_state_to_distrbution_param_mapping(known_mues, sigmas,
                                                                                                     states)

        priors = self._calc_distributions_prior(all_relvent_observations, (len(states) - 2)) if not known_mues else None
        curr_mus = self.build_initial_mus(sigmas, priors, known_mues)
        curr_trans = self.build_initial_transitions(states)

        curr_w = [list(range(len(O))) for O in all_relvent_observations]

        state_to_distrbution_param_mapping = self._update_distributions_params(state_to_distrbution_param_mapping,
                                                                               curr_mus)
        curr_walk, alpha = self.sample_walk_from_params(all_relvent_observations, N, state_to_distrbution_param_mapping,
                                                        start_probs,
                                                        curr_w, curr_trans)

        sampled_states, observations_sum = self._exrect_samples_from_walk(curr_walk, all_relvent_observations, curr_w,
                                                                          state_to_distrbution_param_mapping,
                                                                          curr_mus, sigmas)
        sampled_transitions = self._exrect_transitions_from_walk(curr_walk, states, curr_w)

        w_adj, N_adj = self.build_N_list(curr_walk, curr_w, curr_trans, PC_guess)

        all_sampled_transitions = [sampled_transitions]
        all_transitions = [curr_trans]
        all_states = [sampled_states]
        all_observations_sum = [observations_sum]
        all_mues = [curr_mus]
        all_ws = [curr_w]
        with tqdm(total=Ng_iters) as pbar:
            for i in range(Ng_iters):
                curr_mus = self.sample_mus_from_params(sampled_states, observations_sum, priors, sigmas, known_mues)
                state_to_distrbution_param_mapping = self._update_distributions_params(
                    state_to_distrbution_param_mapping, curr_mus)

                curr_trans, _ = self.sample_trans_from_params(sampled_transitions, states,
                                                              curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                           state_to_distrbution_param_mapping],
                                                              stage_name="transitions" if is_mh else "no_mh",
                                                              observations=all_relvent_observations)

                curr_walk, _ = self.sample_walk_from_params(all_relvent_observations, N_adj,
                                                            state_to_distrbution_param_mapping, start_probs,
                                                            w_adj, curr_trans,
                                                            curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                         state_to_distrbution_param_mapping],
                                                            stage_name="walk" if is_mh else "no_mh",
                                                            observations=all_relvent_observations)

                curr_w, _ = self.sample_ws_from_params(all_relvent_observations, curr_walk,
                                                       state_to_distrbution_param_mapping, N_adj,
                                                       n_iters=w_smapler_n_iter,
                                                       curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                    state_to_distrbution_param_mapping],
                                                       stage_name="w" if is_mh else "no_mh",
                                                       observations=all_relvent_observations)

                sampled_transitions = self._exrect_transitions_from_walk(curr_walk, states, curr_w)
                sampled_states, observations_sum = self._exrect_samples_from_walk(curr_walk, all_relvent_observations,
                                                                                  curr_w,
                                                                                  state_to_distrbution_param_mapping,
                                                                                  curr_mus, sigmas)

                w_adj, N_adj = self.build_N_list(curr_walk, curr_w, curr_trans, PC_guess)

                all_sampled_transitions.append(sampled_transitions)
                all_transitions.append(curr_trans)
                all_states.append(curr_walk)
                all_observations_sum.append(observations_sum)
                all_mues.append(curr_mus)
                all_ws.append(curr_w)
                pbar.update(1)
        return all_states, all_observations_sum, all_sampled_transitions, all_mues, all_ws, all_transitions

    def _new_sample_guess_pc(self, all_relvent_observations, start_probs,
                        known_mues, sigmas, Ng_iters, PC_guess):
        #states mues initial
        states = list(set(list(start_probs.keys()) + ['start', 'end']))
        state_to_distrbution_param_mapping = self.__build_initial_state_to_distrbution_param_mapping(known_mues, sigmas,
                                                                                                     states)
        priors = self._calc_distributions_prior(all_relvent_observations, (len(states) - 2)) if not known_mues else None
        curr_mus = self.build_initial_mus(sigmas, priors, known_mues)
        state_to_distrbution_param_mapping = self._update_distributions_params(state_to_distrbution_param_mapping,
                                                                               curr_mus)

        # init transition matrix - with naive
        curr_trans = self._build_initial_transitions_from_naive(all_relvent_observations, start_probs,
                        known_mues, sigmas)

        #sample first walk - as naive. the length of X is as Obs
        N = [len(O) for O in all_relvent_observations]
        const_w = [list(range(len(O))) for O in all_relvent_observations]
        X_walk, _ = self.sample_walk_from_params(all_relvent_observations, N, state_to_distrbution_param_mapping,
                                                        start_probs,
                                                        const_w, curr_trans)

        sampled_states, observations_sum = self._exrect_samples_from_walk(X_walk, all_relvent_observations, const_w,
                                                                          state_to_distrbution_param_mapping,
                                                                          curr_mus, sigmas)

        #sample D
        w_adj, N_adj,d_lists = self.build_N_list(X_walk, const_w, curr_trans, PC_guess,return_d = True)


        all_sampled_transitions = [curr_trans]
        all_transitions = [curr_trans]
        all_states = [sampled_states]
        all_observations_sum = [observations_sum]
        all_mues = [curr_mus]
        all_ws = [const_w]
        with tqdm(total=Ng_iters) as pbar:
            for i in range(Ng_iters):
                curr_mus = self.sample_mus_from_params(sampled_states, observations_sum, priors, sigmas, known_mues)
                state_to_distrbution_param_mapping = self._update_distributions_params(
                    state_to_distrbution_param_mapping, curr_mus)

                if self.transition_sampling_profile == "all" :
                    # sample transitions from the adj W and N - first we Samle long walk and then calculate
                    for_T_walk, _ = self.sample_walk_from_params(all_relvent_observations, N_adj,
                                                                state_to_distrbution_param_mapping, start_probs,
                                                                w_adj, curr_trans)
                    sampled_transitions = self._exrect_transitions_from_walk(for_T_walk, states, const_w)
                else :
                    sampled_transitions = self._smaple_transitions_from_d_1(X_walk, d_lists, states)
                curr_trans, _ = self.sample_trans_from_params(sampled_transitions, states)
                n_steps_trans = self.build_n_steps_transitions_dicts(curr_trans)

                #sample current X walk from d lists
                X_walk, _ = self.sample_walk_from_params(all_relvent_observations, d_lists,
                                                            state_to_distrbution_param_mapping, start_probs,
                                                            None, n_steps_trans)

                w_adj, N_adj,d_lists = self.build_N_list(X_walk, const_w, curr_trans, PC_guess,return_d = True)


                sampled_states, observations_sum = self._exrect_samples_from_walk(X_walk, all_relvent_observations,
                                                                                  const_w,
                                                                                  state_to_distrbution_param_mapping,
                                                                                  curr_mus, sigmas)



                all_sampled_transitions.append(sampled_transitions)
                all_transitions.append(curr_trans)
                all_states.append(X_walk)
                all_observations_sum.append(observations_sum)
                all_mues.append(curr_mus)
                all_ws.append(const_w)
                pbar.update(1)
        return all_states, all_observations_sum, all_sampled_transitions, all_mues, all_ws, all_transitions

    def _sample_unknown_pc(self, all_relvent_observations, start_probs,
                        known_mues, sigmas, Ng_iters):
        print("start sampler with unknown N/PC")
        delta_emissions_table = {k:{kk:int(k==kk) for kk in start_probs.keys()} for k in start_probs.keys()}

        curr_w = [list(range(len(s))) for s in all_relvent_observations]

        states = list(set(list(start_probs.keys()) + ['start', 'end']))
        state_to_distrbution_param_mapping = self.__build_initial_state_to_distrbution_param_mapping(known_mues, sigmas,
                                                                                                     states)

        priors = self._calc_distributions_prior(all_relvent_observations, (len(states) - 2)) if not known_mues else None
        curr_mus = self.build_initial_mus(sigmas, priors, known_mues)
        curr_trans = self.build_initial_transitions(states)

        state_to_distrbution_param_mapping = self._update_distributions_params(state_to_distrbution_param_mapping,
                                                                               curr_mus)
        curr_walk, alpha = self.sample_walk_from_params(all_relvent_observations, 2, state_to_distrbution_param_mapping,
                                                        start_probs,
                                                        curr_w, curr_trans)

        sampled_states, observations_sum = self._exrect_samples_from_walk(curr_walk, all_relvent_observations, curr_w,
                                                                          state_to_distrbution_param_mapping,
                                                                          curr_mus, sigmas)
        sampled_transitions = self._exrect_transitions_from_walk(curr_walk, states, curr_w)


        all_sampled_transitions = [sampled_transitions]
        all_transitions = [curr_trans]
        all_states = [sampled_states]
        all_observations_sum = [observations_sum]
        all_mues = [curr_mus]

        with tqdm(total=Ng_iters) as pbar:
            for i in range(Ng_iters):
                curr_mus = self.sample_mus_from_params(sampled_states, observations_sum, priors, sigmas, known_mues)
                state_to_distrbution_param_mapping = self._update_distributions_params(
                    state_to_distrbution_param_mapping, curr_mus)

                curr_walk, _ = self.sample_walk_from_params(all_relvent_observations, 2,
                                                            state_to_distrbution_param_mapping, start_probs,
                                                            curr_w, curr_trans)

                trans_to_nstep = self.assign_transition_to_n_step(curr_walk,curr_trans)
                adj_w = list(map(lambda x: [0] + list(np.cumsum(x)),trans_to_nstep))
                _walk_for_trans_calc, _ = self.sample_walk_from_params(curr_walk, list(map(lambda x:x[-1] + 1,adj_w)),
                                                                            delta_emissions_table, start_probs,
                                                                            adj_w, curr_trans)

                sampled_transitions = self._exrect_transitions_from_walk(_walk_for_trans_calc, states, adj_w)
                curr_trans, _ = self.sample_trans_from_params(sampled_transitions, states)

                sampled_states, observations_sum = self._exrect_samples_from_walk(curr_walk, all_relvent_observations,
                                                                                  curr_w,
                                                                                  state_to_distrbution_param_mapping,
                                                                                  curr_mus, sigmas)

                all_sampled_transitions.append(sampled_transitions)
                all_transitions.append(curr_trans)
                all_states.append(curr_walk)
                all_observations_sum.append(observations_sum)
                all_mues.append(curr_mus)
                pbar.update(1)
        return all_states, all_observations_sum, all_sampled_transitions, all_mues, None, all_transitions

    def assign_transition_to_n_step(self,walks,transitions_dict,N_limit=25,accepted_error=0.01):
        n_steps_transitions = GibbsSampler.build_n_steps_transitions_dicts(transitions_dict)

        trans_to_nstep = []
        for walk in walks :
            transitions_in_walk = [(_f,_t) for _f,_t in zip(walk,walk[1:])]
            transition_to_best_n_to_error = self._build_transition_to_best_n_upto_error(transitions_in_walk,
                                                                                        n_steps_transitions, N_limit,
                                                                                        accepted_error)
            trans_to_nstep.append(transition_to_best_n_to_error)
        return trans_to_nstep

    @staticmethod
    def power_matrix_np(matrix, power):
        if power == 0: return np.eye(matrix.shape[0])
        final = matrix
        for i in range(1, power):
            final = final.dot(matrix)
        return final

    @staticmethod
    def build_n_steps_transitions_dicts(transition_dict, max_step=100):
        _transition_dict = {k:{kk:vv for kk,vv in v.items() if kk not in ['start','end']} for
                            k,v in transition_dict.items() if k not in ['start','end']}
        _transition_matrix = pd.DataFrame(_transition_dict).dropna(axis=1).sort_index(axis=1).sort_index(axis=0).T

        transitions = {}
        for i in range(1, max_step):
            new_transition_matrix = GibbsSampler.power_matrix_np(copy.copy(_transition_matrix), i)
            new_transition_dict = copy.copy(new_transition_matrix.T.to_dict())

            transitions[i] = new_transition_dict

        return transitions

    def _build_initial_transitions_from_naive(self,all_relvent_observations, start_probs,
                                          known_mues, sigmas) :
        _, _, _, _, _, all_transitions = self.sample(all_relvent_observations, start_probs,
        known_mues, sigmas, Ng_iters = 10, w_smapler_n_iter = 1, N = 2, is_mh = False)
        return all_transitions[-1]

    def build_N_list(self,walks,W,transitions_dict,N_factor,n_steps_transitions = None,return_d = False):
        if n_steps_transitions is None :
            n_steps_transitions = GibbsSampler.build_n_steps_transitions_dicts(transitions_dict)

        N_w_s_lists = []
        N_length_list = []
        ds_list = []
        for walk,w in zip(walks,W) :
            obs_walk = [obs for i,obs in enumerate(walk) if i in w ]

            first_state_time = self._calculate_first_state_time(obs_walk[0],n_steps_transitions,N_factor)
            last_time_from_state = self._calculate_last_time_from_state(N_factor)
            transitions_windows_time = [self._sample_N_window(_f,_t,n_steps_transitions,N_factor)
                                        for _f,_t in zip(obs_walk,obs_walk[1:])]

            N_list = [first_state_time] + transitions_windows_time + [last_time_from_state]
            ds_list.append(N_list)

            N_w_s_lists.append(list(np.cumsum(N_list)[:-1]))
            N_length_list.append(sum(N_list)+1)
        if not return_d :
            return N_w_s_lists,N_length_list
        else :
            return N_w_s_lists, N_length_list,ds_list

    def _build_transition_to_best_n_upto_error(self, transitions_in_walk, n_steps_transitions, N_limit, acc_error):
        transition_to_n_prob = []
        for trans in transitions_in_walk :
            tran_prob_per_n = list(map(lambda n : n_steps_transitions[n][trans[0]][trans[1]],range(1,N_limit)))

            #best n
            max_prob = max(tran_prob_per_n)
            #distance from best n
            dist_from_best_n = list(map(lambda x:max_prob-x,tran_prob_per_n))
            #accepted distance
            is_accepted_distance = list(map(lambda x:x<acc_error,dist_from_best_n))
            #first true
            smallest_n_accp = self.__return_first_True(is_accepted_distance)

            transition_to_n_prob.append(smallest_n_accp)

        return transition_to_n_prob

    def _smaple_transitions_from_d_1(self, Xs, ds_list, states) :
        trans_count = {k:{kk:0 for kk in states if not kk in ["start","end"]} for k in states if not k in ["start","end"] }
        for X,ds in zip(Xs,ds_list) :
            for i in range(1,len(X)-1):
                cu_x,nx_x = X[i-1],X[i]
                d = ds[i]
                if d==1 :
                    trans_count[cu_x][nx_x] += 1

        return trans_count

    def __return_first_True(self,l):
        for i,val in enumerate(l):
            if val : return (i+1)

    def _sample_N_window(self,from_state,to_state,n_steps_transitions,N_factor):
        prob_function = lambda n : n_steps_transitions[n][from_state][to_state]*(N_factor)*((1-N_factor)**(n-1))

        probs = np.array(list(map(prob_function,range(1,100))))
        norm_probs = probs/probs.sum()

        return np.random.choice(list(range(1,100)),p=norm_probs)

    def _calculate_last_time_from_state(self,N_factor):
        prob_function = lambda n: (N_factor) * ((1 - N_factor) ** (n - 1))
        probs = np.array(list(map(prob_function, range(1, 100))))
        norm_probs = probs / probs.sum()

        return np.random.choice(list(range(1, 100)), p=norm_probs)-1

    def _calculate_first_state_time(self,first_obs,n_steps_transitions,N_factor):
        all_states = n_steps_transitions[1].keys()

        probs_per_pos_state = []
        for _pos_orig_state in all_states :
            _probs_pos_state = self.calculate_probs_single_orig(_pos_orig_state,first_obs,n_steps_transitions,N_factor)
            probs_per_pos_state.append(np.array(_probs_pos_state))

        probs_per_N = reduce(lambda x,y:x+y,probs_per_pos_state)

        return np.random.choice(list(n_steps_transitions.keys()), p=probs_per_N/probs_per_N.sum()) -1

    def calculate_probs_single_orig(self,_pos_orig_state,first_obs,n_steps_transitions,N_factor):
        all_n_steps =[trans_dict[_pos_orig_state][first_obs] for trans_dict in n_steps_transitions.values()]
        P_ab_all_N = np.mean(all_n_steps)

        probs = [n_steps_transitions[i][_pos_orig_state][first_obs]*P_ab_all_N*(N_factor)*((1-N_factor)**(i-1)) for
                 i in n_steps_transitions.keys()]
        return probs

    def sample_W_options(self,known_emissions, all_relvent_observations, curr_trans, start_probs,
                         emmisions_params, Ng_iters, w_smapler_n_iter=100, N=None, is_mh=False):
        if known_emissions :
            emissions_table = emmisions_params
            return self.sample_W_options_known_emissions(all_relvent_observations, curr_trans, start_probs,
                         emissions_table, Ng_iters, w_smapler_n_iter, N, is_mh)
        else :
            known_mues, sigmas = emmisions_params
            return self.sample_W_options_normal_dist(all_relvent_observations, curr_trans, start_probs,
                         known_mues, sigmas, Ng_iters, w_smapler_n_iter, N, is_mh)

    def sample_W_options_normal_dist(self, all_relvent_observations, curr_trans, start_probs,
                         known_mues, sigmas, Ng_iters, w_smapler_n_iter=100, N=None, is_mh=False):
        N = self.N if N is None else N
        states = list(set(list(start_probs.keys()) + ['start', 'end']))
        state_to_distrbution_param_mapping = self.__build_initial_state_to_distrbution_param_mapping(known_mues, sigmas,
                                                                                                     states)

        priors = self._calc_distributions_prior(all_relvent_observations, (len(states) - 2)) if not known_mues else None
        curr_mus = self.build_initial_mus(sigmas, priors, known_mues)

        if type(N) is list:
            curr_w = [sorted(np.random.choice(range(max(_N, len(obs))), len(obs), replace=False)) for obs, _N in
                      zip(all_relvent_observations, N)]
        else:
            curr_w = [sorted(np.random.choice(range(max(N, len(obs))), len(obs), replace=False)) for obs in
                      all_relvent_observations]

        state_to_distrbution_param_mapping = self._update_distributions_params(state_to_distrbution_param_mapping,
                                                                               curr_mus)
        curr_walk, alpha = self.sample_walk_from_params(all_relvent_observations, N, state_to_distrbution_param_mapping,
                                                        start_probs,
                                                        curr_w, curr_trans)

        sampled_states, observations_sum = self._exrect_samples_from_walk(curr_walk, all_relvent_observations, curr_w,
                                                                          state_to_distrbution_param_mapping,
                                                                          curr_mus, sigmas)
        # Iterations for convergence
        with tqdm(total=Ng_iters) as pbar:
            for i in range(Ng_iters):
                curr_mus = self.sample_mus_from_params(sampled_states, observations_sum, priors, sigmas, known_mues)
                state_to_distrbution_param_mapping = self._update_distributions_params(
                    state_to_distrbution_param_mapping, curr_mus)

                curr_w, _ = self.sample_ws_from_params(all_relvent_observations, curr_walk,
                                                       state_to_distrbution_param_mapping, N,
                                                       n_iters=w_smapler_n_iter,
                                                       curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                    state_to_distrbution_param_mapping],
                                                       stage_name="w" if is_mh else "no_mh",
                                                       observations=all_relvent_observations)

                curr_walk, _ = self.sample_walk_from_params(all_relvent_observations, N,
                                                            state_to_distrbution_param_mapping, start_probs,
                                                            curr_w, curr_trans,
                                                            curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                         state_to_distrbution_param_mapping],
                                                            stage_name="walk" if is_mh else "no_mh",
                                                            observations=all_relvent_observations)

                sampled_states, observations_sum = self._exrect_samples_from_walk(curr_walk, all_relvent_observations,
                                                                                  curr_w,
                                                                                  state_to_distrbution_param_mapping,
                                                                                  curr_mus, sigmas)

        # Iterations for sampling W
        sampled_ws_options = []
        for i in range(20):
            curr_mus = self.sample_mus_from_params(sampled_states, observations_sum, priors, sigmas, known_mues)
            state_to_distrbution_param_mapping = self._update_distributions_params(
                state_to_distrbution_param_mapping, curr_mus)

            curr_w, _ = self.sample_ws_from_params(all_relvent_observations, curr_walk,
                                                   state_to_distrbution_param_mapping, N,
                                                   n_iters=w_smapler_n_iter,
                                                   curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                state_to_distrbution_param_mapping],
                                                   stage_name="w" if is_mh else "no_mh",
                                                   observations=all_relvent_observations)
            sampled_ws_options.append(curr_w)

            curr_walk, _ = self.sample_walk_from_params(all_relvent_observations, N,
                                                        state_to_distrbution_param_mapping, start_probs,
                                                        curr_w, curr_trans,
                                                        curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                     state_to_distrbution_param_mapping],
                                                        stage_name="walk" if is_mh else "no_mh",
                                                        observations=all_relvent_observations)

            sampled_states, observations_sum = self._exrect_samples_from_walk(curr_walk, all_relvent_observations,
                                                                              curr_w,
                                                                              state_to_distrbution_param_mapping,
                                                                              curr_mus, sigmas)

        return sampled_ws_options

    def sample_W_options_known_emissions(self, all_relvent_observations, curr_trans, start_probs,
                         emissions_table, Ng_iters, w_smapler_n_iter=100, N=None, is_mh=False):
        emissions_table = self.impute_emissions_table_with_zeros(emissions_table, all_relvent_observations)
        N = self.N if N is None else N

        if type(N) is list:
            curr_w = [sorted(np.random.choice(range(max(_N, len(obs))), len(obs), replace=False)) for obs, _N in
                      zip(all_relvent_observations, N)]
        else:
            curr_w = [sorted(np.random.choice(range(max(N, len(obs))), len(obs), replace=False)) for obs in
                      all_relvent_observations]

        curr_walk, alpha = self.sample_walk_from_params(all_relvent_observations, N,
                                                        emissions_table, start_probs,
                                                        curr_w, curr_trans)

        # Iterations for convergence
        with tqdm(total=Ng_iters) as pbar:
            for i in range(Ng_iters):
                curr_w, alpha1 = self.sample_ws_from_params(all_relvent_observations, curr_walk, emissions_table, N,
                                                            n_iters=w_smapler_n_iter,
                                                            curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                         emissions_table],
                                                            stage_name="w" if is_mh else "no_mh",
                                                            observations=all_relvent_observations)

                curr_walk, alpha2 = self.sample_walk_from_params(all_relvent_observations, N,
                                                                 emissions_table, start_probs,
                                                                 curr_w, curr_trans,
                                                                 curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                              emissions_table],
                                                                 stage_name="walk" if is_mh else "no_mh",
                                                                 observations=all_relvent_observations)
                pbar.update(1)


        # Iterations for sampling W
        sampled_ws_options = []
        with tqdm(total=Ng_iters) as pbar:
            for i in range(20):
                curr_w, alpha1 = self.sample_ws_from_params(all_relvent_observations, curr_walk, emissions_table, N,
                                                            n_iters=w_smapler_n_iter,
                                                            curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                         emissions_table],
                                                            stage_name="w" if is_mh else "no_mh",
                                                            observations=all_relvent_observations)
                sampled_ws_options.append(curr_w)

                curr_walk, alpha2 = self.sample_walk_from_params(all_relvent_observations, N,
                                                                 emissions_table, start_probs,
                                                                 curr_w, curr_trans,
                                                                 curr_params=[curr_trans, curr_w, curr_walk, None,
                                                                              emissions_table],
                                                                 stage_name="walk" if is_mh else "no_mh",
                                                                 observations=all_relvent_observations)
                pbar.update(1)

        return sampled_ws_options

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

    def _calc_distributions_prior(self,all_relvent_observations,N):
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

    @staticmethod
    def choice(options, probs):
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
        if _old_dim_vector == _curr_dim_vector:
            return 1

        old_prob = 1
        k = 0
        for w in _old_dim_vector:
            old_prob = old_prob * y_from_x_probs[(k, w)]
            k = k + 1

        curr_prob = 1
        k = 0
        for w in _curr_dim_vector:
            curr_prob = curr_prob * y_from_x_probs[(k, w)]
            k = k + 1

        if curr_prob == 0:
            if old_prob == 0:
                return 1
            else:
                return 0
        elif old_prob == 0:
            return 1

        return curr_prob / old_prob

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
    def __probability_from_dist_params(observation,params,is_transitions_dict) :
        if observation is None:
            return 1
        if not is_transitions_dict:
            return Utils.normpdf(observation, params[0], params[1])
        else:
            return params[observation] if observation in params.keys() else 0

    @staticmethod
    def _build_emmisions_for_sample(sample, w, state_to_distrbution_param_mapping,N,normalized_emm =  True,
                                    is_known_emm = False):
        states = list(state_to_distrbution_param_mapping.keys())

        __tmp_param = state_to_distrbution_param_mapping[states[0]]
        is_transitions_dict = not (type(__tmp_param) is tuple or type(__tmp_param) is list)

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
                    if not is_known_emm :
                        _emm = GibbsSampler.__probability_from_dist_params(observation,
                                                                       state_to_distrbution_param_mapping[state],
                                                                       is_transitions_dict)

                    else :
                        _emm = 1 if observation is None else int(state_to_distrbution_param_mapping[state] == observation)
                #if not cyclic case : if the start is out of time is zero

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
    def __sample_emmisions_matrix(emm_prob,state,observation_ind):
      return emm_prob[(state,observation_ind)]

    @staticmethod
    def _sample_trans_matrix( trans_prob, from_state, to_state):
        if from_state not in trans_prob.keys() : return 0
        if to_state not in trans_prob[from_state].keys() : return 0

        return trans_prob[from_state][to_state]

    @staticmethod
    def _sample_flat_trans(trans_prob, from_state, to_state,d=None):
        if d is None :
            return trans_prob[(from_state,to_state)]
        else :
            return trans_prob[(d,from_state, to_state)]

    @staticmethod
    def __sample_single_time(prev_state, walk, fwd, trans_prob, n,n_steps = None) :
        if n == -1 :
            return walk

        is_n_step = not (n_steps is None)

        states_of_time = []
        prob_of_states_of_time = []

        for state,prob in fwd[n].items() :
            if not is_n_step :
                prob_for_sample = prob * GibbsSampler._sample_flat_trans(trans_prob, state, prev_state)
            else :
                prob_for_sample = prob * GibbsSampler._sample_flat_trans(trans_prob, state, prev_state,n_steps[n+1])

            states_of_time.append(state)
            prob_of_states_of_time.append(prob_for_sample)

        #TODO: maybe we should remove zero chance transitions ?
        prob_of_states_of_time = prob_of_states_of_time if sum(prob_of_states_of_time) != 0 else [1 for i in prob_of_states_of_time]

        _sum = sum(prob_of_states_of_time)
        sampled_state_inde = GibbsSampler.choice(range(len(states_of_time)),[p/_sum for p in prob_of_states_of_time] )
        sampled_state = states_of_time[sampled_state_inde]

        walk.append(sampled_state)

        return GibbsSampler.__sample_single_time(sampled_state,walk,fwd, trans_prob, n - 1,n_steps=n_steps)

    @staticmethod
    def _build_single_walk_from_postrior(fwd,trans_prob,N,n_step = None) :
        walk = []
        states_of_time = []
        prob_of_states_of_time = []

        for state,prob in fwd[N-1].items() :
            states_of_time.append(state)
            prob_of_states_of_time.append(prob)

        prob_of_states_of_time = prob_of_states_of_time if sum(prob_of_states_of_time) != 0 else np.array([1 for i in prob_of_states_of_time])
        _sum_prob_of_states_of_time = sum(prob_of_states_of_time)
        sampled_state_inde = GibbsSampler.choice(range(len(states_of_time)),
                                    [p / _sum_prob_of_states_of_time for p in prob_of_states_of_time])
        sampled_state = states_of_time[sampled_state_inde]

        walk.append(sampled_state)

        _walk  = GibbsSampler.__sample_single_time(sampled_state,walk,fwd,trans_prob,N-2,n_steps=n_step)
        walk = list(reversed(_walk))

        return walk

    @staticmethod
    def _fwd_bkw(states, start_prob, trans_prob, emm_prob,N,only_forward = False,n_steps = None):
        """Forward–backward algorithm."""
        is_n_step = not (n_steps is None)
        # Forward part of the algorithm
        fwd = []
        for observation_i in range(N):
            f_curr = {}
            for st in states:
                if observation_i == 0:
                    if not is_n_step :
                        if st in start_prob.keys() :
                            prev_f_sum = start_prob[st]
                        else :
                            prev_f_sum = start_prob[str(st)]
                    else :
                        d = n_steps[observation_i]
                        if d == 0 :
                            prev_f_sum =  start_prob[st]
                        else :
                            prev_f_sum = sum([start_prob[k]*GibbsSampler._sample_flat_trans(trans_prob, k, st,d) for k in states])
                else:
                    if not is_n_step :
                        prev_f_sum = sum([f_prev[k] * GibbsSampler._sample_flat_trans(trans_prob, k, st)
                                      for k in states if f_prev[k] !=0])
                    else :
                        d = n_steps[observation_i]
                        prev_f_sum = sum([f_prev[k] * GibbsSampler._sample_flat_trans(trans_prob,k, st,d)
                                          for k in states if f_prev[k] != 0])

                if emm_prob is not None :
                    # F-B case
                    f_curr[st] = GibbsSampler.__sample_emmisions_matrix(emm_prob,st,observation_i) * prev_f_sum
                else :
                    # when we want to calculate only transitions (deviation of prob from expected sample count)
                    f_curr[st] =  prev_f_sum
            fwd.append(f_curr)
            f_prev = f_curr

        if only_forward :
            return fwd

        # Backward part of the algorithm
        return GibbsSampler._build_single_walk_from_postrior(fwd,trans_prob, N,n_step=n_steps)

    @staticmethod
    def _fwd_for_inference(states, start_prob, trans_prob, emm_prob, seq_with_nones):
        """Forward–backward algorithm."""
        # Forward part of the algorithm
        is_possible = not any([trans_prob[(_f,_t)] == 0 for  _f,_t in zip(seq_with_nones,seq_with_nones[1:]) if ((not _f is None) and  (not _t is None) )])
        if not is_possible : return [{"not_valid":0}]

        fwd = []
        for observation_i,sample in enumerate(seq_with_nones):
            f_curr = {}
            for st in states:
                if observation_i == 0:
                    # base case for the forward part
                    if st in start_prob.keys():
                        prev_f_sum = start_prob[st]
                    else:
                        prev_f_sum = start_prob[str(st)]
                else:
                    prev_f_sum = sum(
                    [f_prev[k] * GibbsSampler._sample_flat_trans(trans_prob, k, st) for k in states if
                     f_prev[k] != 0])

                if emm_prob is not None:
                    # F-B case
                    f_curr[st] = GibbsSampler.__sample_emmisions_matrix(emm_prob, st, observation_i) * prev_f_sum
                else:
                    # when we want to calculate only transitions (deviation of prob from expected sample count)
                    f_curr[st] = prev_f_sum
            fwd.append(f_curr)
            if sum(f_curr.values()) == 0: return fwd
            f_prev = f_curr
        return fwd

    def sample_mus_from_params(self,all_sampled_states, sum_relvent_observations, priors,  sigmas,known_mues):
        if known_mues is not None :
            return known_mues

        sis = sum_relvent_observations
        nis = all_sampled_states

        new_mues = {}
        for state in sis.keys() :
            time = int(state[1])
            ξ =  priors[time][0]
            κ =  priors[time][1]
            _mue = (sis[state] + ξ * κ * sigmas[state]) / (κ * sigmas[state] + nis[state])
            _sig = (sigmas[state]) / (κ * sigmas[state] + nis[state])
            new_mues[state] = np.random.normal(_mue, _sig)

        return new_mues

    @Utils.update_based_on_alpha
    def sample_trans_from_params(self,all_transitions,states):
        sampled_transitions = {state: {s:0 for s in states} for state in states}

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
            return Utils.normpdf(obs,
                                 state_to_distrbution_param_mapping[state][0],
                                 state_to_distrbution_param_mapping[state][1])
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

        if self.multi_process :
            with Pool(base_config.n_cores) as p :
                result_per_traj = p.map(_partial_sample_ws_from_params,samples_data)
        else :
            result_per_traj = list(map(_partial_sample_ws_from_params, samples_data))

        return result_per_traj

    def sample_walk_from_param_n_step_trans(self, state_to_distrbution_param_mapping,start_prob,
                                                            curr_trans,samples_data) :
        sample, _curr_ws,_ds = samples_data
        seq_length = len(sample)

        emmisions = GibbsSampler._build_emmisions_for_sample( sample,
                                                     _curr_ws, state_to_distrbution_param_mapping, seq_length)

        flat_trans_prob_n_steps = {(n,_f, _t): self._sample_trans_matrix(curr_trans[n], _f, _t) for n,_f, _t in
                           itertools.product(curr_trans.keys(),curr_trans[1].keys(), curr_trans[1].keys())}

        posterior = self._fwd_bkw( state_to_distrbution_param_mapping.keys(),
                                  start_prob, flat_trans_prob_n_steps, emmisions, seq_length,n_steps=_ds)
        return posterior

    def _sample_walk_from_params(self, N, state_to_distrbution_param_mapping,start_prob,  curr_trans,samples_data):
        if type(N) is list :
            if type(N[0]) is list:
                return self.sample_walk_from_param_n_step_trans( state_to_distrbution_param_mapping,start_prob,
                                                            curr_trans,samples_data)

        if type(N) is list :
            sample, _curr_ws,_N = samples_data
            seq_length = max(len(sample), _N)
        else :
            sample, _curr_ws = samples_data
            seq_length = max(len(sample), N)

        emmisions = GibbsSampler._build_emmisions_for_sample( sample,
                                                     _curr_ws, state_to_distrbution_param_mapping, seq_length)

        flat_trans_prob = {(_f, _t): self._sample_trans_matrix(curr_trans, _f, _t) for _f, _t in
                           itertools.product(curr_trans.keys(), curr_trans.keys())}
        posterior = self._fwd_bkw( state_to_distrbution_param_mapping.keys(),
                                  start_prob, flat_trans_prob, emmisions, seq_length)
        return posterior

    @Utils.update_based_on_alpha
    def sample_walk_from_params(self, sampled_trajs,N, state_to_distrbution_param_mapping,
                                start_prob, curr_ws, curr_trans):
        curr_ws = curr_ws if (not curr_ws is None) else [list(range(len(o))) for o in sampled_trajs]
        if type(N) is list :
            samples_data = [(sampled_trajs[i], curr_ws[i],N[i]) for i in range(len(sampled_trajs))]
        else :
            samples_data = list(zip(sampled_trajs , curr_ws))

        _partial_sample_walk_from_params = partial(self._sample_walk_from_params, N,
                                                   state_to_distrbution_param_mapping,start_prob,  curr_trans)
        if self.multi_process :
            with Pool(base_config.n_cores) as p:
                walks = p.map(_partial_sample_walk_from_params, samples_data)
        else :
            walks = list(map(_partial_sample_walk_from_params, samples_data))

        return walks

    def build_initial_transitions(self,states):
        initial_transitions = {state: {} for state in states}

        normalization_factors = {}
        for _s1 in initial_transitions.keys():
            if _s1 == 'end' : continue

            _sum = 0
            for _s2 in initial_transitions.keys():
                if _s2 == 'start' : continue

                _val = 0

                _val = np.random.rand()

                initial_transitions[_s1][_s2] = _val
                _sum += _val
            normalization_factors[_s1] = _sum

        initial_transitions = {_s1:{__s1:__s2/normalization_factors[_s1] for __s1,__s2 in s2.items()} for _s1,s2 in initial_transitions.items()}
        return initial_transitions

    def build_initial_mus(self,sigmas,priors,known_mues):
        if known_mues is not None:
            return known_mues

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

        if self.multi_process :
            with Pool(base_config.n_cores) as p :
                all_initial_observations_sum = p.map(_partial__exrect_samples_from_walk,samples_data)
        else :
            all_initial_observations_sum = list(map(_partial__exrect_samples_from_walk, samples_data))

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

        if self.multi_process :
            with Pool(base_config.n_cores) as p :
                all_initial_observations_sum = p.map(_partial__exrect_samples_from_walk,samples_data)
        else :
            all_initial_observations_sum = list(map(_partial__exrect_samples_from_walk, samples_data))

        observations_sum = self.__agg_list_of_dict(all_initial_observations_sum)

        return states_count,observations_sum

    def impute_emissions_table_with_zeros(self, emissions_table,all_relvent_observations,impute_zeros = False) :
        all_possible_emissions = set(chain(*all_relvent_observations))
        new_emissions_table = emissions_table
        for _emm in all_possible_emissions :
            for k,v in emissions_table.items():
                if _emm not in v.keys() :
                    new_emissions_table[k][_emm] = 0 if not impute_zeros else 1e-6

        if impute_zeros :
            return {k:{kk:(vv if vv != 0 else 1e-9) for kk,vv in v.items()} for k,v in new_emissions_table.items()}

        return new_emissions_table

    def _calculate_prob_single_sample(self, state_to_distrbution_param_mapping, start_prob, curr_trans, samples_data):
        sample, _curr_ws, _N = samples_data
        seq_length = max(len(sample), _N)

        emmisions = GibbsSampler._build_emmisions_for_sample(sample,
                                                             _curr_ws, state_to_distrbution_param_mapping,
                                                             seq_length, is_known_emm=True)
        seq_with_nones = []
        k=0
        for i in range(_N) :
            if i in _curr_ws :
                seq_with_nones.append(sample[k])
                k += 1
            else :
                seq_with_nones.append(None)

        flat_trans_prob = {(_f, _t): self._sample_trans_matrix(curr_trans, _f, _t) for _f, _t in
                           itertools.product(curr_trans.keys(), curr_trans.keys())}
        posterior = self._fwd_for_inference(state_to_distrbution_param_mapping.keys(),
                                            start_prob, flat_trans_prob, emmisions, seq_with_nones)
        return sum(posterior[-1].values())

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
