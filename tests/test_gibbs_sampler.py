from unittest import TestCase
from who_cell.models.gibbs_sampler import GibbsSampler

from who_cell.models.gibbs_sampler import GibbsSampler
from who_cell.simulation.simulation_for_gibbs import Simulator_for_Gibbs
from who_cell.experiments.experiment_report import ExperimentReport
from who_cell.experiments.gibbs_experiments import GibbsExperiment



class TestGibbsSampler(TestCase):
    def test_sample_known_emissions(self):
        length_of_chain = 2
        number_of_states_in_time = 5

        gs = GibbsSampler(length_of_chain, number_of_states_in_time)

        start_probs = {f"state_{i}": 0.2 for i in range(5)}
        emissions_table = {}
        emissions_table['state_0'] = {'0': 0.6, '1': 0.1, '2': 0.1, '3': 0.1, '4': 0.1}
        emissions_table['state_1'] = {'0': 0.1, '1': 0.6, '2': 0.1, '3': 0.1, '4': 0.1}
        emissions_table['state_2'] = {'0': 0.1, '1': 0.1, '2': 0.6, '3': 0.1, '4': 0.1}
        emissions_table['state_3'] = {'0': 0.1, '1': 0.1, '2': 0.1, '3': 0.6, '4': 0.1}
        emissions_table['state_4'] = {'0': 0.1, '1': 0.1, '2': 0.1, '3': 0.1, '4': 0.6}

        all_relvent_observations = [['0', '0', '1', '2', '2', '1', '0', '2', '4', '1', '0', '3', '1', '4', '0'],
                                    ['1', '1', '2', '4', '3', '4', '2', '2', '1', '1', '0', '3', '1', '4', '0'],
                                    ['0', '0', '1', '2', '2', '1', '0', '2', '4', '1', '0', '3', '1', '4', '0'],
                                    ['0', '0', '1', '2', '2', '1', '0', '2', '4', '1', '0', '3', '1', '4', '0'],
                                    ['4', '1', '1', '3', '2', '1', '2', '3', '1', '0', '0', '3', '1', '4', '0'],
                                    ['0', '1', '4', '2', '2', '1', '0', '3', '4', '1', '1', '3', '1', '4', '0']]
        N = [len(seq) + 1 for seq in all_relvent_observations]
        res = gs.sample_known_emissions(all_relvent_observations, start_probs, emissions_table, 10, 20, N=N)


class TestGibbsSampler(TestCase):
    def test__sample_guess_pc(self):
        model_defining_params_pre = ['N', "d", "n_states", 'is_acyclic', 'sigma', 'bipartite', 'known_dataset']
        params_dict = {
            'is_acyclic': [True],
            'known_mues': [True],
            "is_few_observation_model": [True],
            "is_only_seen": ["all"],
            'N': [60],
            'd': [5],
            "bipartite": [False],
            "inner_outer_trans_probs_ratio": [50],
            'n_states': [10],
            'sigma': [0.1],
            "known_dataset": [-1],
            'number_of_smapled_traj': [1500],
            'p_prob_of_observation': [0.5],
            'N_itres': [0],
            'is_mh': [False],
            'w_smapler_n_iter': [100],
            'is_known_W': [False],
            "is_multi_process": [True],
            "PC_guess": [0.5]
        }

        er = ExperimentReport()
        all_models_results = GibbsExperiment.run_multi_params_and_return_results(params_dict, model_defining_params_pre,
                                                                                 skip_sampler=True)

        all_relvent_observations = all_models_results[0][0]["all_relvent_sampled_trajs_states"][0:200]
        known_ws = all_models_results[0][0]["known_ws"][0:200]
        all_full_sampled_trajs_states = all_models_results[0][0]["all_full_sampled_trajs_states"][:200]
        trans = er._extrect_states_transitions_dict_from_pome_model(all_models_results[0][0]["original_pome_model"])[0]
        trans = {str(k): {str(kk): vv for kk, vv in v.items()} for k, v in trans.items()}
        start_probs = all_models_results[0][0]["start_probabilites"]

        emm_table = {s: {ss: 1 if s==ss else 0.001 for ss in start_probs.keys()} for s in start_probs.keys()}
        emm_table = {k:{kk:vv/sum(v.values()) for kk,vv in v.items()} for k,v in emm_table.items()}
        gs = GibbsSampler(params_dict["N"][0], multi_process=False)

        all_states_n, all_ws_n,picked_by_w = gs.infer_w_known_T(all_relvent_observations, start_probs, trans,
                                                    emm_table, 40, params_dict["N"][0], w_smapler_n_iter=150)

        all_states_d, all_ws_d, all_wadjs_d = gs.infer_ds_known_T(all_relvent_observations, start_probs, trans,
                                                                  emm_table, 40, params_dict["PC_guess"][0])
