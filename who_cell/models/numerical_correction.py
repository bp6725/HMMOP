import pandas as pd
import numpy as np



class NumericalCorrection():
    def __init__(self):
        pass

    def em_gibbs_numerical_reconstruction(self, all_relvent_observations, start_probs,
                                          known_mues, sigmas, Ng_iters, w_smapler_n_iter=100, N=None, is_mh=False):
        pass

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
    def reconstruct_full_transitions_matrix_from_few(few_transition_matrix, pc):
        return np.linalg.inv((pc) * NumericalCorrection.power_matrix_np(few_transition_matrix, 0) + (1 - pc) * few_transition_matrix).dot(
            few_transition_matrix)

    @staticmethod
    def reconstruct_full_transitions_dict_from_few(few_transition_dict, pc_guess,start_probabilites):
        all_states = list(start_probabilites.keys())
        few_transition_dict = NumericalCorrection.rebuild_transitions_dict(few_transition_dict,all_states)
        few_transition_matrix, df = NumericalCorrection.buil_np_matrix(few_transition_dict)
        numrical_full_matrix = NumericalCorrection.reconstruct_full_transitions_matrix_from_few(few_transition_matrix, pc_guess)
        numrical_full_matrix_df = pd.DataFrame(columns=df.columns, index=df.index, data=numrical_full_matrix)
        #     print(df - numrical_full_matrix_df)
        return numrical_full_matrix_df.T.to_dict()
