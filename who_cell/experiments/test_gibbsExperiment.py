from unittest import TestCase
from who_cell.experiments.gibbs_experiments import GibbsExperiment
from who_cell.models.gibbs_sampler import GibbsSampler
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

class TestGibbsExperiment(TestCase):
    def test_run_multi_params_and_plot_report(self):
        model_defining_params_pre = ['N', "d", "n_states",'is_acyclic','sigma','bipartite']
        params_dict = {
            'is_acyclic': [True],
            'known_mues' : [False],
            "is_few_observation_model":[False,True],
            "bipartite" : [True],
            "inner_outer_trans_probs_ratio":[5],
            'N': [50,15],
            'd': [5],
            'n_states': [15],
            'sigma' : [0.1,0.7],
            'number_of_smapled_traj': [300,1500],
            'p_prob_of_observation': [1,0.4,0.1],
            'N_itres': [35],
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

    def test_POS(self):
        import pickle
        with open(r"../../benchmarks/POS_tagging/all_you_need_for_sampling.pkl", 'rb') as f:
            [few_obs_test_set_words, start_probs, emms_probs, number_of_iters, _known_N] = pickle.load( f)

        gs = GibbsSampler(2,5)
        res = gs.sample_known_emissions(few_obs_test_set_words, start_probs, emms_probs, number_of_iters, N=_known_N)


    def test_Penn_POS(self):
        import pickle
        with open(r"C:\Repos\WhoCell\benchmarks\POS_tagging\tmp_cache.pkl", 'rb') as f:
            [test_set_words,start_probs,emms_probs,number_of_iters] = pickle.load( f)

        gs = GibbsSampler(2,5)
        all_sampled_transitions, all_ws, all_transitions,_states_picked_by_w = gs.sample_known_emissions(test_set_words,start_probs,emms_probs,number_of_iters)