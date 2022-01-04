from pomegranate import *
import numpy as np
import itertools
from who_cell.Infras import Infras
from who_cell.simulation.known_transition_matrices import KnownTransition_matrices

class Simulator_for_Gibbs():
    def __init__(self,N,d,n_states,easy_mode=False,max_number_of_sampled_traj = None,sigma = 0.1 ):
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

        if isinstance(sigma, (int, float)) :
            self.sigmas = np.ones_like(self.mues)*sigma
        else :
            self.sigmas = np.array(sigma)

    @staticmethod
    def _sample_n_points_from_traj(full_sample,_n):
        ws = sorted(np.random.choice(range(len(full_sample)), _n, replace=False))
        return ([full_sample[i] for i in ws],ws)

    def _sample_single_traj_from_pome_model(self,pome_model,return_emissions = True,traj_length = None):
        if traj_length is None :
            long_sampled = pome_model.sample(path=True) # sample with inner loops
        else :
            long_sampled = pome_model.sample(length =traj_length
                                             ,path=True)  # sample with inner loops
        sampled_emissions = long_sampled[0]
        sampled_states = [ss.name for ss in long_sampled[1]]

        return sampled_emissions,sampled_states

        # if return_emissions :
        #     _sampled_emm = [sampled_emissions[i-1] for i in range(1,len(sampled_states)) if
        #                 sampled_states[i] != sampled_states[i-1]] # remove consucutive duplications
        #     _sampled_states = [sampled_states[i] for i in range(1, len(sampled_states)) if
        #                 sampled_states[i] != sampled_states[i - 1]]  # remove consucutive duplications
        #
        #     return _sampled_emm, _sampled_states
        #
        # _sampled = [sampled_states[i] for i in range(1, len(sampled_states) - 1) if
        #             sampled_states[i] != sampled_states[i - 1]]  # remove consucutive duplications
        # return _sampled

    def __build_known_mues_and_sigmes_to_state_mapping(self, mues,sigmas, state_to_distrbution_mapping):
        _known_mues_to_state = np.zeros((self.d,self.N))
        _known_sigmes_to_state = np.zeros((self.d, self.N))
        for in_time,time in itertools.product(range(self.d),range(self.N)) :
            _known_mues_to_state[in_time,time] = mues[state_to_distrbution_mapping[in_time,time]]
            _known_sigmes_to_state[in_time,time] = sigmas[state_to_distrbution_mapping[in_time,time]]

        return _known_mues_to_state,_known_sigmes_to_state

    def _sample_N_traj_from_pome_model(self,pome_model,N,traj_length = None):
        _traj = []
        _traj_states = []

        i=0
        while i < N :
            __t,__s = self._sample_single_traj_from_pome_model(pome_model,True,traj_length)

            if len(__t) <2 : continue

            _traj.append(__t)
            _traj_states.append(__s[1:])
            i += 1
        return _traj,_traj_states

    @staticmethod
    def _sample_n_points_from_traj_binom(vec, pc):
        relevent_obs = []
        w = []
        for i,obs in enumerate(vec) :
            if np.random.rand() < pc :
                relevent_obs.append(obs)
                w.append(i)

        if len(w) < 3 :
            return Simulator_for_Gibbs._sample_n_points_from_traj_binom(vec, pc)
        else :
            return (relevent_obs,w)

    @staticmethod
    def bernoulli_experiments(p_prob_of_observation, all_full_sampled_trajs):
        all_relvent_observations_and_ws = []
        for vec in all_full_sampled_trajs:
            # binom_dist = binom(len(vec), p_prob_of_observation)
            # n_of_obs = binom_dist.rvs(1)
            # n_of_obs = n_of_obs if n_of_obs > 2 else 2
            _new_vec = Simulator_for_Gibbs._sample_n_points_from_traj_binom(vec, p_prob_of_observation)
            all_relvent_observations_and_ws.append(_new_vec)

        all_relvent_observations = [ro[0] for ro in all_relvent_observations_and_ws]
        all_ws = [ro[1] for ro in all_relvent_observations_and_ws]

        return all_relvent_observations, all_ws

    @staticmethod
    def bernoulli_experiments_pc_changes(p_mean, p_std, all_full_sampled_trajs):
        all_relvent_observations_and_ws = []
        for vec in all_full_sampled_trajs:
            p_prob_of_observation = np.random.normal(p_mean, p_std, 1)
            p_prob_of_observation = p_prob_of_observation if p_prob_of_observation > 0.05 else 0.05
            p_prob_of_observation = p_prob_of_observation if p_prob_of_observation < 1 else 1

            # binom_dist = binom(len(vec), p_prob_of_observation)
            # n_of_obs = binom_dist.rvs(1)
            # n_of_obs = n_of_obs if n_of_obs > 2 else 2
            _new_vec = Simulator_for_Gibbs._sample_n_points_from_traj_binom(vec, p_prob_of_observation)
            all_relvent_observations_and_ws.append(_new_vec)

        all_relvent_observations = [ro[0] for ro in all_relvent_observations_and_ws]
        all_ws = [ro[1] for ro in all_relvent_observations_and_ws]

        return all_relvent_observations, all_ws

    @staticmethod
    def bernoulli_experiments_pc_mm(pp, pq, qp, qq, all_full_sampled_trajs):
        d1 = DiscreteDistribution({'p': 0.5, 'q': 0.5})
        d2 = ConditionalProbabilityTable([['p', 'p', pp],
                                          ['p', 'q', pq],
                                          ['q', 'p', qp],
                                          ['q', 'q', qq]], [d1])
        markov_model = MarkovChain([d1, d2])

        all_relvent_observations_and_ws = []
        for vec in all_full_sampled_trajs:
            removel_seq = markov_model.sample(len(vec))
            obs = [vec[i] for i, is_seen in enumerate(removel_seq) if is_seen == 'p']
            ws = [i for i, is_seen in enumerate(removel_seq) if is_seen == 'p']
            all_relvent_observations_and_ws.append((obs, ws))

        all_relvent_observations = [ro[0] for ro in all_relvent_observations_and_ws]
        all_ws = [ro[1] for ro in all_relvent_observations_and_ws]

        return all_relvent_observations, all_ws

    @staticmethod
    def  sample_traj_for_few_obs(p_params,all_full_sampled_trajs):
        if ((type(p_params) is float) or (type(p_params) is int)):
            p_prob_of_observation = p_params
            return Simulator_for_Gibbs.bernoulli_experiments(p_prob_of_observation,all_full_sampled_trajs)

        if len(p_params) == 1:
            p_prob_of_observation = p_params[0]
            return Simulator_for_Gibbs.bernoulli_experiments(p_prob_of_observation,all_full_sampled_trajs)

        if len(p_params) == 2 :
            p_mean,p_std  = p_params
            return Simulator_for_Gibbs.bernoulli_experiments_pc_changes(p_mean,p_std,all_full_sampled_trajs)

        if len(p_params) == 4 :
            pp,pq,qp,qq  = p_params
            return Simulator_for_Gibbs.bernoulli_experiments_pc_mm(pp,pq,qp,qq,all_full_sampled_trajs)


    @Infras.storage_cache
    def simulate_observations(self,pome_model,mutual_model_params_dict,params_signature,from_pre_sampled_traj = False):
        p_prob_of_observation = mutual_model_params_dict['p_prob_of_observation']
        number_of_smapled_traj = mutual_model_params_dict['number_of_smapled_traj']
        N = mutual_model_params_dict['N']

        if from_pre_sampled_traj :
            if self.pre_sampled_traj is not None :
                all_full_sampled_trajs_max_len = self.pre_sampled_traj
                all_full_sampled_trajs_states_max_len = self.pre_sampled_traj_states
                all_full_sampled_trajs = all_full_sampled_trajs_max_len[:number_of_smapled_traj]
                all_full_sampled_trajs_states = all_full_sampled_trajs_states_max_len[:number_of_smapled_traj]
            else :
                all_full_sampled_trajs_max_len,all_full_sampled_trajs_states_max_len = \
                    self._sample_N_traj_from_pome_model(pome_model,self.max_number_of_sampled_traj,N)
                self.pre_sampled_traj = all_full_sampled_trajs_max_len
                self.pre_sampled_traj_states = all_full_sampled_trajs_states_max_len

                all_full_sampled_trajs = all_full_sampled_trajs_max_len[:number_of_smapled_traj]
                all_full_sampled_trajs_states = all_full_sampled_trajs_states_max_len[:number_of_smapled_traj]
        else :
            all_full_sampled_trajs,all_full_sampled_trajs_states = self._sample_N_traj_from_pome_model(pome_model,
                                                                                                       number_of_smapled_traj,N)
        mutual_model_params_dict['non_cons_sim'] = mutual_model_params_dict['non_cons_sim'] if 'non_cons_sim'  in mutual_model_params_dict.keys() else False
        if not mutual_model_params_dict['non_cons_sim']   :
            all_relvent_observations,all_ws = Simulator_for_Gibbs.sample_traj_for_few_obs(p_prob_of_observation,all_full_sampled_trajs)
        else :
            _idx_to_smaple = [np.cumsum(np.random.randint(2,4,N)) for i in range(len(all_full_sampled_trajs))]
            all_relvent_observations = [[traj[w] for w in ws if w < len(traj)] for traj,ws in zip(all_full_sampled_trajs,_idx_to_smaple)]
            all_ws = [[w for w in ws if w < len(traj)] for traj,ws in zip(all_full_sampled_trajs,_idx_to_smaple)]
        all_relvent_sampled_trajs_states = list(map(lambda x: [x[0][i] for i in x[1]], zip(all_full_sampled_trajs_states, all_ws)))
        self.known_Ws = all_ws

        return all_relvent_observations,all_full_sampled_trajs,all_full_sampled_trajs_states,all_relvent_sampled_trajs_states,all_ws

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

    def _build_temporal_states_to_params(self,N,d,mues,sigmas):
        '''
        temporal_states are the states in the shape of (in time ind, time ind).(allow acyclic chain)
        :param N:
        :param d:
        :param mues:
        :param sigmas:
        :return:
        '''
        state_to_distrbution_mapping = {'start':('start'),'end':('end')}

        # build the distrbutions and shuffale them
        all_distrbutions_params = [(mu, sigmas[i]) for i, mu in enumerate(mues)]

        for time_ind in range(N):
            _picked_dists_inds_for_mutual_case = np.random.choice(range(len(all_distrbutions_params)), size=d,
                                                                  replace=False)
            for in_time_ind in range(d):
                _dist_params = all_distrbutions_params[_picked_dists_inds_for_mutual_case[in_time_ind]]
                state_to_distrbution_mapping[(in_time_ind, time_ind)] = _dist_params

        return state_to_distrbution_mapping

    def _build_unique_states_to_params(self,N,d,mues,sigmas):
        '''
        acyclic case
        :param N:
        :param d:
        :param mues:
        :param sigmas:
        :return:
        '''
        state_to_distrbution_mapping = {'start':('start')}

        _state_to_distrbution_mapping = {str((mu, sigmas[i])):(mu, sigmas[i]) for i, mu in enumerate(mues)}

        return {**state_to_distrbution_mapping,**_state_to_distrbution_mapping}

    def _generete_transition_matrix(self,N,d) :
        transition_matrix_sparse = {}

        for time_ind in range(N - 1):
            for in_time_ind in range(d):
                transition_matrix_sparse[(in_time_ind, time_ind)] = {}

                n_of_out_trans = np.random.randint(2, d - 1)
                out_trans = np.random.choice(d, size=n_of_out_trans, replace=False)

                for out_t in out_trans:
                    _trans_val = (np.random.rand() + 0.13) / 1.3
                    transition_matrix_sparse[(in_time_ind, time_ind)][(out_t, time_ind + 1)] = np.round(_trans_val,3)

        transition_matrix_sparse['start'] = {(in_time_ind, 0): 1 for in_time_ind in range(d)}
        for in_time_ind in range(d):
            transition_matrix_sparse[(in_time_ind, N - 1)] = {'end' : 1}

        return transition_matrix_sparse

    def _generete_acyclic_transition_matrix(self,states,d) :
        possible_states = [state for state in states if state  not in ['start','end']]
        transition_matrix_sparse = {}

        for state in states :
            if state == 'end' : continue
            transition_matrix_sparse[state] = {}

            n_of_out_trans =d# np.random.randint(d-1, d +1)
            _possible_states = [s for s in possible_states]
            out_trans = np.random.choice(_possible_states, size=n_of_out_trans, replace=False)

            for out_t in out_trans:
                _trans_val = (np.random.rand() )
                transition_matrix_sparse[state][out_t] = np.round(_trans_val,4)
            # transition_matrix_sparse[state]['end'] = np.round(1/len(states),3)

        transition_matrix_sparse['start'] = {state: np.round(1/len(states),3) for state in possible_states}

        return transition_matrix_sparse

    def __build_start_prob(self,state_to_distrbution_param_mapping):
        start_probs = {}

        count_starters = 0
        for state in state_to_distrbution_param_mapping.keys() :
            if (state == 'start') or (state == 'end') : continue
            is_start = int(state[1] == 0)
            count_starters += is_start

        for state in state_to_distrbution_param_mapping.keys() :
            if (state == 'start') or (state == 'end'): continue
            weight = int(state[1] == 0)/count_starters
            start_probs[state] = weight
        return start_probs

    @Infras.storage_cache
    def build_template_model_parameters(self,N,d,mues,sigmas):
        '''
        this function build the chain params (dists and transitions) in the shape of acyclic chain (state is :(in time ind, time ind)),
        we will add the cyclic part later if needed
        :param N:
        :param d:
        :param mues:
        :param sigmas:
        :return:
        '''

        state_to_distrbution_param_mapping = self._build_temporal_states_to_params(N,d,mues,sigmas)
        transition_matrix_sparse = self._generete_transition_matrix(N,d)
        start_probabilites = self.__build_start_prob(state_to_distrbution_param_mapping)

        return state_to_distrbution_param_mapping,transition_matrix_sparse,start_probabilites

    @Infras.storage_cache
    def build_acyclic_template_model_parameters(self,N,d,mues,sigmas):
        state_to_distrbution_param_mapping = self._build_unique_states_to_params(N,d,mues,sigmas)

        states = state_to_distrbution_param_mapping.keys()
        transition_matrix_sparse = self._generete_acyclic_transition_matrix(states,d)

        states = [state for state in states if state not in ['end','start'] ]
        start_probabilites = {state:1/len(states) for state in states}

        return state_to_distrbution_param_mapping,transition_matrix_sparse,start_probabilites

    def __rename_cyclic_neighbors_map(self,cyclic_neighbors_map,unique_name_to_name_mapping):
        return {unique_name_to_name_mapping[k]:
             {unique_name_to_name_mapping[kk]:vv for kk,vv in v.items()} for k,v in cyclic_neighbors_map.items() }

    def __filter_redundent_names(self,state_to_distrbution_mapping,unique_name_to_name_mapping):
        existing_names = list(unique_name_to_name_mapping.values()) + ['start','end']
        return {k:v for k,v in state_to_distrbution_mapping.items() if k in existing_names}

    def _merge_to_cyclic_chain(self,state_to_distrbution_mapping, transition_matrix_sparse):
        unique_name_to_name_mapping = {}
        _cyclic_neighbors_map = {}
        for state,neighbors in transition_matrix_sparse.items() :
            unique_state = state_to_distrbution_mapping[state]
            if unique_state not in _cyclic_neighbors_map.keys() :
                _cyclic_neighbors_map[unique_state] = {}
                unique_name_to_name_mapping[unique_state] = state

            for neighbor,trans_val in neighbors.items():
                unique_neighbor = state_to_distrbution_mapping[neighbor]
                if unique_neighbor in _cyclic_neighbors_map[unique_state].keys() :
                    continue
                _cyclic_neighbors_map[unique_state][unique_neighbor] = trans_val

        unique_name_to_name_mapping['end'] = 'end'
        _cyclic_neighbors_map_temporal_names = self.__rename_cyclic_neighbors_map(_cyclic_neighbors_map,
                                                                                  unique_name_to_name_mapping)
        #now we only have one temporal_name  to each uniaue state
        state_to_distrbution_mapping = self.__filter_redundent_names(state_to_distrbution_mapping,unique_name_to_name_mapping)

        return _cyclic_neighbors_map_temporal_names, state_to_distrbution_mapping

    def _build_pome_model_from_params(self,state_to_distrbution_param_mapping,
                                      transition_matrix_sparse):
        is_known_emissions = type(state_to_distrbution_param_mapping[list(state_to_distrbution_param_mapping.keys())[0]]) is dict

        if is_known_emissions :
            all_params_to_distrbutions = {st: DiscreteDistribution(dist) for st, dist in
                                          state_to_distrbution_param_mapping.items()}
            state_to_distrbution_param_mapping['start'] = {}
        else :
            # we need this because we need to share the dist instances for pomegranate
            all_params_to_distrbutions = {(_params[0], _params[1]):NormalDistribution(_params[0], _params[1]) for k,_params in
                                state_to_distrbution_param_mapping.items() if ((k != 'start') and (k != 'end'))}
            if 'start' not in state_to_distrbution_param_mapping.keys() :
                state_to_distrbution_param_mapping['start'] = (-1,-1)

        all_model_pome_states = {}
        model = HiddenMarkovModel()
        for state_name,_params in state_to_distrbution_param_mapping.items() :
            if state_name == 'start' :
                model.add_state(model.start)
                all_model_pome_states['start'] = model.start
                continue
            if state_name == 'end':
                model.add_state(model.end)
                all_model_pome_states['end'] = model.end
                continue
            if is_known_emissions :
                state = State(all_params_to_distrbutions[state_name], name=f"{state_name}")
            else :
                state = State(all_params_to_distrbutions[_params], name=f"{state_name}")
            model.add_state(state)
            all_model_pome_states[state_name] = state

        for _from_state_name,_to_states in transition_matrix_sparse.items():
            _from_state = all_model_pome_states[_from_state_name]
            for _to_state_name,_trans_prob in _to_states.items() :
                _to_state = all_model_pome_states[_to_state_name]
                model.add_transition(_from_state,_to_state,_trans_prob)
        model.bake()

        return model,all_model_pome_states

    def __remove_transitions_to_end(self,transition_matrix_sparse) :
        new_transition_matrix_sparse = {state:{} for state in transition_matrix_sparse.keys()}
        for _from,tos in transition_matrix_sparse.items() :
            for _to,weight in tos.items():
                if _to == 'end' : continue
                new_transition_matrix_sparse[_from][_to] = weight
        return new_transition_matrix_sparse

    def __extrect_end_states(self,transition_matrix_sparse) :
        from_states = set([state for state in transition_matrix_sparse.keys() if ((len(state) != 0) and (state != 'start')) ])
        to_states = set(itertools.chain(*[[state for state in states] for states in transition_matrix_sparse.values()]))

        return list(to_states.difference(from_states))

    def _normalize_transition_matrix(self,transition_matrix_sparse) :
        normalized_transition_matrix_sparse = {}
        for _from, to in transition_matrix_sparse.items():
            _sum = sum([v for k, v in to.items()])
            normalized_transition_matrix_sparse[_from] = {_to: (val / _sum) for _to, val in to.items()}
        return normalized_transition_matrix_sparse

    def build_pome_model(self,N, d, mues, sigmas,is_bipartite = False,inner_outer_trans_probs_ratio = 0,
                         mutual_model_params_dict = None):
        '''

        :param N:
        :param d:
        :param mues:
        :param sigmas:
        :param is_mutual_distrbutions: if True, the number of distrbutions is smaller then the number of states (tied states)
        :return:
        '''

        (state_to_distrbution_param_mapping, transition_matrix_sparse, start_probabilites), params_signature = \
            self.return_model_parameters(N, d, mues,sigmas,inner_outer_trans_probs_ratio,
                                                                is_bipartite,mutual_model_params_dict)

        transition_matrix_sparse = self._normalize_transition_matrix(transition_matrix_sparse)

        model, all_model_pome_states = self._build_pome_model_from_params(state_to_distrbution_param_mapping,
                                                                          transition_matrix_sparse)
        state_to_distrbution_mapping = {_s_name:p_model.distribution for _s_name,p_model in all_model_pome_states.items()}

        pome_results = {
            "model":model,
            "state_to_distrbution_mapping":state_to_distrbution_mapping,
            "transition_matrix_sparse":transition_matrix_sparse,
            "state_to_distrbution_param_mapping":state_to_distrbution_param_mapping,
            "start_probabilites":start_probabilites ,
            "params_signature":params_signature
        }

        return pome_results

    def return_model_parameters(self,N, d, mues,sigmas,inner_outer_trans_probs_ratio,
                                                                is_bipartite,mutual_model_params_dict):
        is_known_dataset = mutual_model_params_dict["known_dataset"] != -1 if "known_dataset" in mutual_model_params_dict.keys() else False

        if is_known_dataset :
            return self.build_known_model(mutual_model_params_dict["known_dataset"])

        if not is_bipartite:
            return self.build_acyclic_template_model_parameters(N, d, mues, sigmas)

        if is_bipartite:
            return self.build_bipartite_template_model_parameters(N, d, mues,sigmas,
                                                                  inner_outer_trans_probs_ratio)
        if is_bipartite == "DAG":
            return self.build_dag_template_model_parameters(N, d, mues,sigmas,inner_outer_trans_probs_ratio)

    @Infras.storage_cache
    def build_known_model(self,known_dataset):
        emmisions,transition_dict,start_probs = KnownTransition_matrices.load_data_set_by_name(known_dataset)
        return emmisions,transition_dict,start_probs

    def update_known_mues_and_sigmes_to_state_mapping(self,state_to_distrbution_param_mapping) :
        if type(state_to_distrbution_param_mapping[list(state_to_distrbution_param_mapping.keys())[0]]) is dict : return None
        self.states_known_mues = {state:params[0] for state,params in state_to_distrbution_param_mapping.items()}
        self.states_known_sigmas = {state:params[1] for state,params in state_to_distrbution_param_mapping.items()}

    @Infras.storage_cache
    def build_bipartite_template_model_parameters(self, N, d, mues, sigmas,inner_outer_trans_probs_ratio):
        n_states = len(mues)
        all_distrbutions_params_mapping = {str((mu, sigmas[i])): (mu, sigmas[i]) for i, mu in enumerate(mues)}

        # build groups and state -> group mapping
        groups = [list(all_distrbutions_params_mapping.keys())[i:i + d] for i in
                  range(0, len(all_distrbutions_params_mapping), d)]
        state_to_group = {}
        for i, group_list in enumerate(groups):
            for state in group_list:
                state_to_group[state] = i

        # draw transitions probability
        sparse_transition_matrix = {state: {} for state in all_distrbutions_params_mapping.keys()}
        sparse_transition_matrix['start'] = {}

        for s1, s2 in itertools.product(all_distrbutions_params_mapping, all_distrbutions_params_mapping):
            if state_to_group[s1] == state_to_group[s2]:
                _trans_prob = np.random.rand() / inner_outer_trans_probs_ratio
            elif (state_to_group[s2] - state_to_group[s1]) == 1:
                _trans_prob = inner_outer_trans_probs_ratio * np.random.rand()
            elif (state_to_group[s1] == (len(groups) - 1)) and (state_to_group[s2] == 0):
                _trans_prob = inner_outer_trans_probs_ratio * np.random.rand()
            elif (state_to_group[s2] - state_to_group[s1]) != 1:
                _trans_prob = np.random.rand()

            sparse_transition_matrix[s1][s2] = _trans_prob

            if state_to_group[s1] == 0:
                sparse_transition_matrix['start'][s1] = 1
            else :
                sparse_transition_matrix['start'][s1] = 0

        sparse_transition_matrix = self._normalize_transition_matrix(sparse_transition_matrix)

        return  all_distrbutions_params_mapping , sparse_transition_matrix, sparse_transition_matrix['start']

    @Infras.storage_cache
    def build_dag_template_model_parameters(self, N, d, mues, sigmas,inner_outer_trans_probs_ratio):
        all_distrbutions_params_mapping = {str((mu, sigmas[i])): (mu, sigmas[i]) for i, mu in enumerate(mues)}
        state_to_order = {state: i for i, state in enumerate(all_distrbutions_params_mapping.keys())}

        sparse_transitions_matrix = {}

        for s_from, s_from_order in state_to_order.items():
            _from_transitions = {}
            for s_to, s_to_order in state_to_order.items():
                if s_to_order > s_from_order:
                    _transition = np.random.rand() * inner_outer_trans_probs_ratio
                else:
                    _transition = np.random.rand()

                _from_transitions[s_to] = _transition
            sparse_transitions_matrix[s_from] = _from_transitions

        start_probs = {}
        start_states = []
        for s_from, s_from_order in state_to_order.items():
            if s_from_order < 3 :
                start_probs[s_from] = 1
                start_states.append(s_from)
            else :
                start_probs[s_from] = 0
        sparse_transitions_matrix['start'] = start_probs

        for s_from, s_from_order in state_to_order.items():
            if s_from_order > (max(state_to_order.values())-3) :
                for state in start_states :
                    sparse_transitions_matrix[s_from][state] = inner_outer_trans_probs_ratio

        sparse_transition_matrix = self._normalize_transition_matrix(sparse_transitions_matrix)

        return all_distrbutions_params_mapping, sparse_transition_matrix, sparse_transition_matrix['start']

