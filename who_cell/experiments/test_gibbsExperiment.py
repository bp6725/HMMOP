from unittest import TestCase
from who_cell.experiments.gibbs_experiments import GibbsExperiment
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

class TestGibbsExperiment(TestCase):
    def test_run_multi_params_and_plot_report(self):
        model_defining_params_pre = ['N', "d", "n_states",'is_acyclic','sigma']
        params_dict = {
            'is_acyclic': [True],
            'known_mues' : [False],
            'N': [15],
            'd': [5],
            'n_states': [10],
            'sigma' : [0.1],
            'number_of_smapled_traj': [100],
            'p_prob_of_observation': [1],
            'N_itres': [10],
            'w_smapler_n_iter': [20]}

        GibbsExperiment.run_multi_params_and_plot_report(params_dict,model_defining_params_pre,skip_sampler = False)

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
