from unittest import TestCase
from who_cell.experiments.gibbs_experiments import GibbsExperiment
from who_cell.models.gibbs_sampler import GibbsSampler
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

class TestGibbsExperiment(TestCase):
    def test_run_multi_params_and_plot_report(self):
        model_defining_params_pre = ['N', "d", "n_states",'is_acyclic','sigma','bipartite',"known_dataset"]
        params_dict = {
            'known_mues': [True],
            "is_few_observation_model": [False],
            "is_only_seen": ['all'],
            'N': [45],
            'd': [5],
            "non_cons_sim": [False],
            "bipartite": [False],
            "inner_outer_trans_probs_ratio": [300],
            'n_states': [10],
            "known_dataset":["SCHIZX1_drug"],
            'sigma': [0.1],
            'number_of_smapled_traj': [1000],
            # 'p_prob_of_observation': [0.5, (0.5, 0.1), (0.55, 0.45, 0.45, 0.55)],
            'p_prob_of_observation': [0.5],
            'N_itres': [2],
            'is_mh': [False],
            "is_known_W": [False],
            'w_smapler_n_iter': [80],
            'is_multi_process': [False],
            "use_pomegranate": [False],
            "PC_guess" : ["unknown",-1,0.75,0.8],
            "N_guess" :[5,-1,3],
            "numerical_reconstruction_pc": [-1],
            "is_numerical_reconstruction_method" : [False]
        }

        model_defining_params_pre = ['N', "d", "n_states", 'is_acyclic', 'sigma', 'bipartite', 'known_dataset']
        params_dict = {
            'is_acyclic': [True],
            'known_mues': [True],
            #     "is_few_observation_model":[True,False],
            "is_few_observation_model": [False],
            "is_only_seen": ['observed', "all"],
            'N': [50],
            'd': [5],
            "bipartite": [False],
            "inner_outer_trans_probs_ratio": [50],
            'n_states': [10],
            # "known_dataset": ["SCHIZX1_plcebo", "SCHIZX1_drug"],
            'sigma': [0.1],
            'number_of_smapled_traj': [1000],
            #     'p_prob_of_observation': [0.5],
            'p_prob_of_observation': [1],
            'N_itres': [100],
            'is_mh': [False],
            'w_smapler_n_iter': [120],
            #     'is_known_W':[True,False],
            'is_known_W': [False],
            "is_multi_process": [True],
            "PC_guess": [0.25, -1],
            "N_guess": [-1, 4]
        }

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

        all_models_results = GibbsExperiment.run_multi_params_and_return_results(params_dict,model_defining_params_pre)

    def test_POS(self):
        import pickle
        with open(r"../../benchmarks/POS_tagging/all_you_need_for_sampling.pkl", 'rb') as f:
            [few_obs_test_set_words, start_probs, emms_probs, number_of_iters, _known_N] = pickle.load( f)

        gs = GibbsSampler(2,5)
        res = gs.sample_known_emissions(few_obs_test_set_words, start_probs, emms_probs, number_of_iters, N=30)


    def test_Penn_POS(self):
        import pickle
        with open(r"C:\Repos\WhoCell\benchmarks\POS_tagging\tmp_cache.pkl", 'rb') as f:
            [test_set_words,start_probs,emms_probs,number_of_iters] = pickle.load( f)

        gs = GibbsSampler(2,5)
        all_sampled_transitions, all_ws, all_transitions,_states_picked_by_w = gs.sample_known_emissions(test_set_words,start_probs,emms_probs,number_of_iters)