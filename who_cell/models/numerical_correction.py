import pandas as pd
import numpy as np
from who_cell.models.gibbs_sampler import GibbsSampler
from collections import Counter
import itertools
import copy

class NumericalCorrection():
    def __init__(self,multi_process):
        self.multi_process = multi_process

    def em_gibbs_numerical_reconstruction(self, all_relvent_observations, start_probs,
                                          known_mues, sigmas, Ng_iters, w_smapler_n_iter=100, N=None, is_mh=False):
        known_emissions = True
        if known_emissions :
            all_trans_count = Counter(itertools.chain(*[[(_f,_t) for _f,_t in zip(sentence,sentence[1:]) ] for sentence in all_relvent_observations]))
            all_transitions = {k:{kk:all_trans_count[(k,kk)] for kk in start_probs.keys()} for k in start_probs.keys()}
            naive_transitions_matrix = {k:{kk:vv/sum(v.values()) for kk,vv in v.items()} for k,v in all_transitions.items()}

            emmisions_params = {k:{kk:int(k == kk) for kk in start_probs.keys()} for k in start_probs.keys()}
        else :
            gs = GibbsSampler(2,multi_process=self.multi_process)
            _, _, all_transitions,_,_ = gs.sample( all_relvent_observations, start_probs,
               known_mues,sigmas, 10, w_smapler_n_iter,N=2,is_mh = is_mh)
            naive_transitions_matrix = all_transitions[-1]
            emmisions_params = (known_mues, sigmas)

        results = []
        gs = GibbsSampler(N,multi_process=self.multi_process,transition_sampling_profile="observed")
        avg_seq_len = np.mean(list(map(len, all_relvent_observations)))
        for pc in np.linspace(0.1,1,10) :
            gussed_reconstructed_transitions = NumericalCorrection.reconstruct_full_transitions_dict_from_few(naive_transitions_matrix,
                                                                                                              pc,start_probs,avg_seq_len)
            seq_probs= gs.probability_over_known_transition(known_emissions,all_relvent_observations,
                                                            gussed_reconstructed_transitions, start_probs,
                                                            emmisions_params, Ng_iters,
                                                            w_smapler_n_iter=w_smapler_n_iter, N=N, is_mh=is_mh)
            print((pc,seq_probs))
            results.append((pc,seq_probs))

        best_results = None
        best_prob = 0

        l_pc,u_pc = self.return_best_window(results)
        for pc in np.linspace(l_pc, u_pc, 5):
            gussed_reconstructed_transitions = NumericalCorrection.reconstruct_full_transitions_dict_from_few(
                naive_transitions_matrix,
                pc, start_probs)
            _, _, seq_probs, _, _, _ = gs.probability_over_known_transition(known_emissions,all_relvent_observations,
                                                                            gussed_reconstructed_transitions, start_probs,
                                                                            emmisions_params, Ng_iters, w_smapler_n_iter=w_smapler_n_iter,
                                                                            is_mh=is_mh)
            prob = sum(seq_probs[-3:-1])
            if prob > best_prob :
                best_prob = prob
                best_results = gussed_reconstructed_transitions

        return best_results

    def return_best_window(self,results_list):
        probs = [p for _,p in results_list]
        pcs = [pc for pc, _ in results_list]

        best_pc_idx = np.argmax(probs)
        if best_pc_idx == 0 :
            return pcs[0],pcs[1]
        if best_pc_idx == (len(results_list)-1) :
            return pcs[-2],pcs[-1]
        if probs[best_pc_idx-1] > probs[best_pc_idx + 1] :
            return pcs[best_pc_idx-1],pcs[best_pc_idx]
        else :
            return pcs[best_pc_idx], pcs[best_pc_idx + 1]


    @staticmethod
    def rebuild_transitions_dict(transitions, all_states):
        _transitions = {str(k): {str(kk): vv for kk, vv in v.items() if not kk in ["start", 'end']} for
                        k, v in transitions.items() if not k in ["start", 'end']}
        new_transitions = _transitions.copy()

        for state in all_states:
            if state in ['start', 'end']: continue
            for k, v in _transitions.items():
                if state not in new_transitions[str(k)].keys():
                    new_transitions[str(k)][state] = 0
        return new_transitions

    @staticmethod
    def buil_np_matrix(transition_dict):
        df = pd.DataFrame(transition_dict).dropna(axis=1).sort_index(axis=1).sort_index(axis=0).T
        return df.values, df

    @staticmethod
    def power_matrix_np(matrix, power):
        if power == 0: return np.eye(matrix.shape[0])
        final = matrix
        for i in range(1, power):
            final = final.dot(matrix)
        return final

    @staticmethod
    def reconstruct_full_transitions_matrix_from_few(few_transition_matrix, pc,avg_seq_len):
        _few_transition_matrix = copy.copy(few_transition_matrix)

        _N = int(((1-pc)/pc)*avg_seq_len) + 1
        stats_transitions = NumericalCorrection.power_matrix_np(_few_transition_matrix, 100)
        adj_few_transition_matrix = _few_transition_matrix.dot(
            np.linalg.inv(NumericalCorrection.power_matrix_np(_few_transition_matrix, 0) - ((1 - pc) ** _N) * stats_transitions))

        return np.linalg.inv(
            (pc) * NumericalCorrection.power_matrix_np(adj_few_transition_matrix, 0) + (1 - pc) * adj_few_transition_matrix).dot(
            adj_few_transition_matrix)

    @staticmethod
    def reconstruct_full_transitions_dict_from_few(few_transition_dict, pc_guess,start_probabilites,avg_seq_len):
        all_states = list(start_probabilites.keys())
        few_transition_dict = NumericalCorrection.rebuild_transitions_dict(few_transition_dict,all_states)
        few_transition_matrix, df = NumericalCorrection.buil_np_matrix(few_transition_dict)
        numrical_full_matrix = NumericalCorrection.reconstruct_full_transitions_matrix_from_few(few_transition_matrix, pc_guess,avg_seq_len)
        numrical_full_matrix_df = pd.DataFrame(columns=df.columns, index=df.index, data=numrical_full_matrix)

        numrical_full_matrix_dict = numrical_full_matrix_df.T.to_dict()
        numrical_full_matrix_dict = {k:{kk:(vv if vv >0 else 10**(-4)) for kk,vv in v.items()} for k,v in numrical_full_matrix_dict.items()}
        return  {k:{kk:(vv / sum(v.values())) for kk,vv in v.items()} for k,v in numrical_full_matrix_dict.items()}
