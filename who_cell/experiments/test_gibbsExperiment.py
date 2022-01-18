from unittest import TestCase
from who_cell.experiments.gibbs_experiments import GibbsExperiment
from who_cell.models.gibbs_sampler import GibbsSampler
import sys
from who_cell.simulation.simulation_for_gibbs import Simulator_for_Gibbs

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


class TestGibbsExperiment(TestCase):
    def test_run_multi_params_and_plot_report(self):
        model_defining_params_pre = ['N', "d", "n_states", 'is_acyclic', 'sigma', 'bipartite', "known_dataset"]
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
            "known_dataset": [-1, "SCHIZX1_drug"],
            'sigma': [0.001],
            'number_of_smapled_traj': [2],
            # 'p_prob_of_observation': [0.5, (0.5, 0.1), (0.55, 0.45, 0.45, 0.55)],
            'p_prob_of_observation': [0.5],
            'N_itres': [2],
            'is_mh': [False],
            "is_known_W": [False],
            'w_smapler_n_iter': [80],
            'is_multi_process': [False],
            "use_pomegranate": [False],
            "PC_guess": ["unknown", -1, 0.75, 0.8],
            "N_guess": [5, -1, 3],
            "numerical_reconstruction_pc": [-1],
            "is_numerical_reconstruction_method": [False]
        }

        model_defining_params_pre = ['N', "d", "n_states", 'is_acyclic', 'sigma', 'bipartite', 'known_dataset']
        params_dict = {
            'is_acyclic': [True],
            'known_mues': [True],
            "is_few_observation_model": [False, True],
            "is_only_seen": ["all"],
            'N': [50],
            'd': [5],
            "bipartite": [False],
            "inner_outer_trans_probs_ratio": [50],
            'n_states': [10],
            "known_dataset": ["POS", -1],
            'sigma': [0.1],
            'number_of_smapled_traj': [500],
            'p_prob_of_observation': [(0.55, 0.45, 0.45, 0.55), (0.65, 0.35, 0.35, 0.65), (0.7, 0.3, 0.45, 0.55),0.5],
            'N_itres': [1],
            'is_mh': [False],
            'w_smapler_n_iter': [100],
            'is_known_W': [True, False],
            "is_multi_process": [False],
            "PC_guess": ["known"],
            "numerical_reconstruction_pc": [0.5],
            "is_numerical_reconstruction_method": [True, False],
            "exp_name" :["test"]
        }
        # all_models_results_known_sets = GibbsExperiment.run_multi_params_and_return_results(params_dict,
        #                                                                                     model_defining_params_pre,
        #                                                                                     skip_sampler=True)

        N = 30
        n_traj = 150
        pcs = [ 0.5, 0.65, 0.8, 1]
        n_iters = 200
        model_defining_params_pre = ['N', "d", "n_states", 'is_acyclic', 'sigma', 'bipartite', 'known_dataset']
        params_dict = {
            'is_acyclic': [True],
            'known_mues': [True],
            "is_few_observation_model": [True, False],
            "is_only_seen": ["all", "d1"],
            'N': [50],
            'd': [5, 3],
            "bipartite": [False],
            "inner_outer_trans_probs_ratio": [50],
            'n_states': [10],
            'sigma': [0.1],
            'number_of_smapled_traj': [n_traj],
            'p_prob_of_observation': pcs,
            'N_itres': [1],
            'is_mh': [False],
            'w_smapler_n_iter': [120],
            'is_known_W': [False, True],
            "is_multi_process": [False],
            "PC_guess": ["known"],
            "exp_name": ["firth figure"]
        }
        all_models_results_syntetic = GibbsExperiment.run_multi_params_and_return_results(params_dict,
                                                                                          model_defining_params_pre,
                                                                                          skip_sampler=True)

    def test_run_multi_params_and_return_results(self):
        mutual_model_params_dict = {
            'is_acyclic': True,
            'known_mues': True,
            "is_few_observation_model": True,
            "is_only_seen": 'observed',
            'N': 50,
            'd': 5,
            "bipartite": False,
            "inner_outer_trans_probs_ratio": 50,
            'n_states': 10,
            'sigma': 0.001,
            'number_of_smapled_traj': 200,
            'p_prob_of_observation': 0.5,
            'N_itres': 2,
            'is_mh': False,
            "known_dataset": -1,
            'w_smapler_n_iter': 100}
        run_name = "P(C) = 0.5"

        simulator = Simulator_for_Gibbs(mutual_model_params_dict['N'], mutual_model_params_dict['d'],
                                        mutual_model_params_dict['n_states'], easy_mode=True,
                                        max_number_of_sampled_traj=mutual_model_params_dict["number_of_smapled_traj"],
                                        sigma=mutual_model_params_dict[
                                            'sigma'])  # we need max_number_of_sampled_traj to know how much traj to pre sample so the traj will be mutual

        pome_results = simulator.build_pome_model(mutual_model_params_dict['N'], mutual_model_params_dict['d'],
                                                  simulator.mues, simulator.sigmas, False,
                                                  inner_outer_trans_probs_ratio=mutual_model_params_dict[
                                                      'inner_outer_trans_probs_ratio'],
                                                  mutual_model_params_dict=mutual_model_params_dict)

        simulator.update_known_mues_and_sigmes_to_state_mapping(pome_results["state_to_distrbution_param_mapping"])

        (all_relvent_observations, all_full_sampled_trajs, all_full_sampled_trajs_states,
         all_relvent_sampled_trajs_states, all_ws), _ = \
            simulator.simulate_observations(pome_results["model"], mutual_model_params_dict,
                                            pome_results['params_signature'], from_pre_sampled_traj=True)

        sampler = GibbsSampler(mutual_model_params_dict['N'], mutual_model_params_dict['d'],
                               transition_sampling_profile=mutual_model_params_dict["is_only_seen"],
                               multi_process=False)
        all_states, _, all_sampled_transitions, _, all_ws, all_transitions = sampler._new_sample_guess_pc(
            all_relvent_observations, pome_results['start_probabilites'],
            simulator.states_known_mues, simulator.states_known_sigmas, mutual_model_params_dict['N_itres'], 0.5)

    def test_POS(self):
        import pickle
        with open(r"../../benchmarks/POS_tagging/all_you_need_for_sampling.pkl", 'rb') as f:
            [few_obs_test_set_words, start_probs, emms_probs, number_of_iters, _known_N] = pickle.load(f)

        gs = GibbsSampler(2, 5)
        res = gs.sample_known_emissions(few_obs_test_set_words, start_probs, emms_probs, number_of_iters, N=30)

    def test_Penn_POS(self):
        import pickle
        with open(r"C:\Repos\WhoCell\benchmarks\POS_tagging\tmp_cache.pkl", 'rb') as f:
            [test_set_words, start_probs, emms_probs, number_of_iters] = pickle.load(f)

        gs = GibbsSampler(2, 5)
        all_sampled_transitions, all_ws, all_transitions, _states_picked_by_w = gs.sample_known_emissions(
            test_set_words, start_probs, emms_probs, number_of_iters)