from unittest import TestCase
from who_cell.experiments.gibbs_experiments import GibbsExperiment


class TestGibbsExperiment(TestCase):
    def test_run_multi_params_and_plot_report(self):
        model_defining_params_pre = ['N', "d", "n_states"]
        params_dict = {
            'N': [6, 12],
            'd': [8],
            'n_states': [20],
            'number_of_smapled_traj': [50, 30],
            'p_prob_of_observation': [0.7, 1],
            'N_itres': [10],
            'w_smapler_n_iter': [10]}

        model_defining_params_pre = ['N', "d", "n_states"]
        params_dict = {
            'N': [12],
            'd': [8],
            'n_states': [20],
            'number_of_smapled_traj': [500],
            'p_prob_of_observation': [0.7],
            'N_itres': [100],
            'w_smapler_n_iter': [20]}


        GibbsExperiment.run_multi_params_and_plot_report(params_dict,model_defining_params_pre)

    def test_run_multi_params_and_return_results(self):
        model_defining_params_pre = ['N', "d", "n_states"]
        params_dict = {
            'N': [6,12],
            'd': [8],
            'n_states': [20],
            'number_of_smapled_traj': [50, 30],
            'p_prob_of_observation': [0.7, 1],
            'N_itres': [10],
            'w_smapler_n_iter': [10]}

        GibbsExperiment.run_multi_params_and_return_results(params_dict,model_defining_params_pre)
