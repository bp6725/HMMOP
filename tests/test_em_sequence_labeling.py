from unittest import TestCase
from who_cell.models.em_sequence_labeling import EmSequenceLabeling
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pomegranate import State,DiscreteDistribution,HiddenMarkovModel
from who_cell.models.gibbs_sampler import GibbsSampler
from itertools import chain
import pandas as pd

class TestEmSequenceLabeling(TestCase):
    @staticmethod
    def build_pome_model_from_trnaisiotns(predicted_transitions, words_emms_probs, start_probs, all_states):
        states_name_to_state_mapping = {state: State(DiscreteDistribution(words_emms_probs[state]), state) for state in
                                        all_states}

        _model = HiddenMarkovModel()
        for _from, _tos in predicted_transitions.items():
            if _from == "start":
                for state in all_states:
                    _to_state = states_name_to_state_mapping[state]
                    _model.add_transition(_model.start, _to_state, start_probs[state])
                continue
            for _to, val in _tos.items():
                if _to == 'end': continue
                _to_state = states_name_to_state_mapping[_to]

                _from_state = states_name_to_state_mapping[_from]
                _model.add_transition(_from_state, _to_state, val)

        _model.bake()
        return _model

    @staticmethod
    def calculate_error_pome(predicted_transitions, test_set_words, test_set_tags, words_emms_probs, start_probs,
                             all_states, unknown_words):
        pome_model = TestEmSequenceLabeling.build_pome_model_from_trnaisiotns(predicted_transitions, words_emms_probs, start_probs, all_states)
        states_name_list = [state.name for state in pome_model.states]

        errors = []
        for sent_words, sent_tags in zip(test_set_words, test_set_tags):
            sent_words = [(word if word not in unknown_words else None) for word in sent_words]
            _predicted = pome_model.predict(sent_words)
            predicted_tags = [states_name_list[i] for i in _predicted]
            error = sum([i != j for i, j in zip(sent_tags, predicted_tags)])
            errors.append(error)

        amount_of_tags = sum(list(map(len, test_set_tags)))
        return np.sum(errors) / amount_of_tags

    @staticmethod
    def calculate_error_gibbs(N, predicted_transitions, test_set_words,
                              test_set_tags, words_emms_probs, start_probs,
                              all_states):

        _test_set_words = list(filter(lambda x: len(x) < 60, test_set_words))
        _test_set_tags = list(filter(lambda x: len(x) < 60, test_set_tags))
        _N = list(filter(lambda x: x < 60, N)) if type(N) is list else N

        all_states_picked_by_w = EmSequenceLabeling.sequence_labeling_known_emissions(_test_set_words,
                                                                                      predicted_transitions,
                                                                                      start_probs,
                                                                                      words_emms_probs, 5, N=_N)

        errors = []
        for known_tags, predicted_tags in zip(_test_set_tags, all_states_picked_by_w[-1]):
            error = sum([i != j for i, j in zip(known_tags, predicted_tags)])
            errors.append(error)

        amount_of_tags = sum(list(map(len, _test_set_tags)))
        return np.sum(errors) / amount_of_tags

    @staticmethod
    def plot_df_results(model_results_df, title):
        fig, sub = plt.subplots(1, 1, figsize=(8, 8))

        sns.lineplot(data=model_results_df, ax=sub, legend='full', dashes=False)
        sub.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        sub.set_title(title)
        sub.set_xlabel("iter")
        sub.set_ylabel(f"Hamming distance over test set")

        plt.subplots_adjust(hspace=0.8)
        plt.subplots_adjust(wspace=0.8)

        plt.show()


    def test_sequence_labeling_known_emissions(self):
        with open('../benchmarks/POS_tagging/build_and_plot_data.pkl','rb') as f :
            (experiments_results,experiments_params,
             test_set_words,test_set_tags,all_states,
             words_emms_probs,all_words_in_test_set,
             words_emms_probs,start_probs) = pickle.load(f)


        missing_obs_errors = {}
        gibbs_no_missing_errors = {}

        n_iter_exp = len(experiments_results[0]['transitions'])
        transitions_to_pick = sorted(list(set(list(range(0, n_iter_exp, 15)) + [n_iter_exp - 1])))
        print(transitions_to_pick)
        for i, (exp_res, exp_args) in enumerate(zip(experiments_results, experiments_params)):
            transitions = exp_res['transitions']
            fewer_transitions = [transitions[i] for i in transitions_to_pick]

            title = exp_args['title']

            gibbs_missing_error_per_experiment = [TestEmSequenceLabeling.calculate_error_gibbs(exp_args["N"],
                                                                                               _trans,
                                                                                               test_set_words,
                                                                                               test_set_tags,
                                                                                               words_emms_probs,
                                                                                               start_probs,
                                                                                               all_states) for _trans in
                                                  fewer_transitions]
            missing_obs_errors[title] = gibbs_missing_error_per_experiment

            gibbs_full_error_per_experiment = [
                TestEmSequenceLabeling.calculate_error_gibbs(2, _trans, test_set_words, test_set_tags,
                                                             words_emms_probs, start_probs, all_states)
                for _trans in fewer_transitions]
            gibbs_no_missing_errors[title] = gibbs_full_error_per_experiment

        gibbs_mising_results_df = pd.DataFrame(missing_obs_errors)
        gibbs_full_results_df = pd.DataFrame(gibbs_no_missing_errors)

        TestEmSequenceLabeling.plot_df_results(gibbs_mising_results_df,
                                               "inference Gibbs counting for missing observations ")
        TestEmSequenceLabeling.plot_df_results(gibbs_full_results_df,
                                               "inference Gibbs not counting for missing observations")

