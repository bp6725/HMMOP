from unittest import TestCase
from who_cell.experiments.gibbs_experiments import GibbsExperiment
from who_cell.models.gibbs_sampler import GibbsSampler
import sys
from who_cell.simulation.simulation_for_gibbs import Simulator_for_Gibbs
from who_cell.experiments.experiment_report import ExperimentReport
from who_cell.models.em_sequence_labeling import EmSequenceLabeling
import numpy as np
import pandas as pd
import pickle
import itertools

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


class TestGibbsExperiment(TestCase):
    def test_run_multi_params_and_plot_report_remote_aws(self):
        model_defining_params_pre = ['N', "d", "n_states", 'is_acyclic', 'sigma', 'bipartite']
        model_defining_params_pre = ['N', "d", "n_states", 'is_acyclic', 'sigma', 'bipartite']
        params_dict = {
            'is_acyclic': [True],
            'known_mues': [True],
            "is_few_observation_model": [True],
            "is_only_seen": ["all"],
            'N': [70],
            'd': [5, 3],
            "bipartite": [False],
            "inner_outer_trans_probs_ratio": [50],
            'n_states': [10],
            'sigma': [0.1],
            'number_of_smapled_traj': [1500],
            'p_prob_of_observation': [0.5],
            'N_itres': [51],
            'is_mh': [False],
            'w_smapler_n_iter': [120],
            'is_known_W': [False],
            "is_multi_process": [True],
            "PC_guess": ["known"],
            "numerical_reconstruction_pc": [-1],
            "is_numerical_reconstruction_method": [False],
            "exp_name": ["wrong pc guess"]
        }
        all_models_results_syntetic_fohmm = GibbsExperiment.run_multi_params_and_return_results(params_dict,
                                                                                                model_defining_params_pre,
                                                                                                skip_sampler=True)

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

    def test_seq_labels(self):
        all_models_results = {}
        er = ExperimentReport()
        def get_few_obs_idxs(model_results):
            if (model_results[0]["hyper_params"]["is_few_observation_model"]) and (
            not model_results[1]["hyper_params"]["is_few_observation_model"]):
                return 0, 1
            if (model_results[1]["hyper_params"]["is_few_observation_model"]) and (
            not model_results[0]["hyper_params"]["is_few_observation_model"]):
                return 1, 0
            raise Exception()

        def rebuild_transitions_dict(transitions, all_states, start_probs):
            new_transitions = {str(k): {str(kk): vv for kk, vv in v.items() if kk not in ["start", "end"]}
                               for k, v in transitions.items() if k != "end"}

            for state in all_states:
                for k, v in transitions.items():
                    if k == "end": continue
                    if state not in new_transitions[str(k)].keys():
                        new_transitions[str(k)][state] = 0

            new_transitions["start"] = start_probs
            return new_transitions

        def prediction_scores_all_sens(predicted, known):
            if len(predicted) != len(known): raise Exception()
            if len(list(filter(lambda x: len(x[0]) != len(x[1]), zip(predicted, known)))) > 0: raise Exception()

            all_errors = []
            for _known, _predicted in zip(known, predicted):
                error = sum([a == b for a, b in zip(_known, _predicted)]) / len(_known)
                all_errors.append(error)
            return all_errors

        def return_exp_name(unique_model_results):
            d = unique_model_results[0]['mutual_params']['d']
            s = unique_model_results[0]['mutual_params']['sigma']
            is_bip = unique_model_results[0]['mutual_params']['bipartite']
            bip = f"Multi Partite" if is_bip else f"d = {d}"

            return f"Multi Partite ; sigma = {s}" if is_bip else f"d = {d} ; sigma = {s}", (bip, s)

        def run_prediction_return_score(unique_model_results, n_sentences=10):
            # extrect experiments
            #     miss_ex_idx,naive_ex_idx =   get_few_obs_idxs(unique_model_results)
            #     miss_ex_results, naive_ex_results = unique_model_results[miss_ex_idx],unique_model_results[naive_ex_idx]
            naive_ex_results = unique_model_results[0]

            # params
            start_probabilites = {k:(1/len(naive_ex_results["start_probabilites"].keys())) for k in naive_ex_results["start_probabilites"].keys()}
            state_to_distrbution_param_mapping = {s: (m, naive_ex_results['simulator'].states_known_sigmas[s]) for s, m
                                                  in naive_ex_results['simulator'].states_known_mues.items() if
                                                  s != "start"}
            state_to_distrbution_param_mapping['start'] = 'start'

            # build transition matrics
            all_states = start_probabilites.keys()

            _T = er._extrect_states_transitions_dict_from_pome_model(naive_ex_results["original_pome_model"])[0]
            T = rebuild_transitions_dict(_T, all_states, naive_ex_results["start_probabilites"])

            T_naive = rebuild_transitions_dict(naive_ex_results['all_transitions'][-1], all_states,
                                               naive_ex_results["start_probabilites"])
            #     T_miss = rebuild_transitions_dict(miss_ex_results['all_transitions'][-1],all_states,miss_ex_results["start_probabilites"])

            # relvent data
            miss_sens_obs = naive_ex_results["all_relvent_observations"][:n_sentences]
            miss_sens_labels = naive_ex_results["all_relvent_sampled_trajs_states"][:n_sentences]

            full_sens_obs = naive_ex_results["all_full_sampled_trajs"][:n_sentences]
            full_sens_labels = naive_ex_results["all_full_sampled_trajs_states"][:n_sentences]

            # run prediction
            pc = naive_ex_results["mutual_params"]["p_prob_of_observation"]
            _N_known = list(map(lambda x: naive_ex_results["mutual_params"]['N'], full_sens_obs))
            return_N_naive = lambda x: list(map(len, x))

            # known T, full sentences, naive infer
            # xml_T_full_naive = EmSequenceLabeling.most_likely_path(full_sens_obs, T,
            #                                                        start_probabilites,
            #                                                        state_to_distrbution_param_mapping, Ng_iters=15,
            #                                                        N=return_N_naive(full_sens_obs))
            # # known T, missing sentences, naive infer
            # xml_T_missing_naive = EmSequenceLabeling.most_likely_path(miss_sens_obs, T,
            #                                                           start_probabilites,
            #                                                           state_to_distrbution_param_mapping, Ng_iters=15,
            #                                                           N=return_N_naive(miss_sens_obs))
            # known T, missing sentences, missing infer
            xml_T_missing_miss = EmSequenceLabeling.most_likely_path(miss_sens_obs, T,
                                                                     start_probabilites,
                                                                     state_to_distrbution_param_mapping, Ng_iters=15,
                                                                     N=_N_known)
            # T naive, full sentences, naive infer
            #     xml_Tnaive_full_naive = EmSequenceLabeling.most_likely_path(full_sens_obs,T_naive,
            #                                         start_probabilites,
            #                                         state_to_distrbution_param_mapping, Ng_iters=15,
            #                                                                N=return_N_naive(full_sens_obs))
            #     #T missing, full sentences, naive infer
            #     xml_Tmiss_full_naive = EmSequenceLabeling.most_likely_path(full_sens_obs,T_miss,
            #                                        start_probabilites,
            #                                         state_to_distrbution_param_mapping, Ng_iters=15,
            #                                                                N=return_N_naive(full_sens_obs))

            # scores
            # scores_list_T_full_naive = prediction_scores_all_sens(xml_T_full_naive, full_sens_labels)
            # scores_list_T_missing_naive = prediction_scores_all_sens(xml_T_missing_naive, miss_sens_labels)
            scores_list_T_missing_miss = prediction_scores_all_sens(xml_T_missing_miss, miss_sens_labels)
            #     scores_list_Tnaive_full_naive = prediction_scores_all_sens(xml_Tnaive_full_naive,full_sens_labels)
            #     scores_list_Tmiss_full_naive  = prediction_scores_all_sens(xml_Tmiss_full_naive,full_sens_labels)

            res = {}
            # res["scores_list_T_full_naive"] = scores_list_T_full_naive
            # res["scores_list_T_missing_naive"] = scores_list_T_missing_naive
            res["scores_list_T_missing_miss"] = scores_list_T_missing_miss
            #     res["scores_list_Tnaive_full_naive"] = scores_list_Tnaive_full_naive
            #     res["scores_list_Tmiss_full_naive"] = scores_list_Tmiss_full_naive

            return res

        return_result_string = lambda x: f"{np.round(np.mean(x), 4)} +- ({np.round(np.std(x), 4)})"

        model_defining_params_pre = ['N', "d", "n_states", 'is_acyclic', 'sigma', 'bipartite', 'known_dataset']
        params_dict = {
            'is_acyclic': [True],
            'known_mues': [True],
            "is_few_observation_model": [False],
            "is_only_seen": ["all"],
            'N': [50],
            'd': [5],
            "bipartite": [True],
            "inner_outer_trans_probs_ratio": [50],
            'n_states': [25],
            #             "known_dataset": ["SCHIZX1_plcebo", "SCHIZX1_drug","POS"],
            'sigma': [0.1, 0.3, 0.5],
            'number_of_smapled_traj': [100],
            'p_prob_of_observation': [0.5],
            'N_itres': [10],
            'is_mh': [False],
            'w_smapler_n_iter': [100],
            'is_known_W': [False],
            "is_multi_process": [False],
        }

        all_models_results_bipartite = GibbsExperiment.run_multi_params_and_return_results(params_dict,
                                                                                           model_defining_params_pre,
                                                                                           skip_sampler=False)

        all_res = {}
        all_df_scores = []
        for unique_model_results in all_models_results_bipartite.values():
            res = run_prediction_return_score(unique_model_results)
            exp_name, exp_params = return_exp_name(unique_model_results)

            all_res[exp_name] = res

            df_score = pd.DataFrame({exp_params: {
                ('T', "full sentences", "naive"): return_result_string(res["scores_list_T_full_naive"]),
                ('T', "missing sentences", "naive"): return_result_string(res["scores_list_T_missing_naive"]),
                ('T', "missing sentences", "ours"): return_result_string(res["scores_list_T_missing_miss"]),
                #     ('T reconstructed',"full sentences","naive") :return_result_string(res["scores_list_Tnaive_full_naive"]),
                #     ('T reconstructed',"full sentences","ours"):return_result_string(res["scores_list_Tmiss_full_naive"])
            }})

            all_df_scores.append(df_score)
        pd.concat(all_df_scores, 1)

    def test_seq_labels_pos(self):
        er = ExperimentReport()

        def rebuild_transitions_dict(transitions, all_states, start_probs):
            new_transitions = {str(k): {str(kk): vv for kk, vv in v.items() if kk not in ["start", "end"]}
                               for k, v in transitions.items() if k != "end"}

            for state in all_states:
                for k, v in transitions.items():
                    if k == "end": continue
                    if state not in new_transitions[str(k)].keys():
                        new_transitions[str(k)][state] = 0

            new_transitions["start"] = start_probs
            return new_transitions

        def prediction_scores_all_sens(predicted, known):
            if len(predicted) != len(known): raise Exception()
            if len(list(filter(lambda x: len(x[0]) != len(x[1]), zip(predicted, known)))) > 0: raise Exception()

            all_errors = []
            for _known, _predicted in zip(known, predicted):
                error = sum([a == b for a, b in zip(_known, _predicted)]) / len(_known)
                all_errors.append(error)
            return all_errors

        def run_pos_prediction_return_score(unique_model_results, n_sentences=2000):

            naive_ex_results = unique_model_results[0]

            with open(r"../../benchmarks/POS_tagging/all_you_need_for_sampling.pkl", 'rb') as f:
                [_, start_probabilites, state_to_distrbution_param_mapping, _, _] = pickle.load(f)

                # build transition matrics
            all_states = start_probabilites.keys()

            _T = er._extrect_states_transitions_dict_from_pome_model(naive_ex_results["original_pome_model"])[0]
            T = rebuild_transitions_dict(_T, all_states, naive_ex_results["start_probabilites"])

            T_naive = rebuild_transitions_dict(naive_ex_results['all_transitions'][-1], all_states,
                                               naive_ex_results["start_probabilites"])
            #     T_miss = rebuild_transitions_dict(miss_ex_results['all_transitions'][-1],all_states,miss_ex_results["start_probabilites"])

            all_words = set(
                itertools.chain(*[[k for k in v.keys()] for v in state_to_distrbution_param_mapping.values()]))
            miss_sens_obs = []
            miss_sens_labels = []
            full_sens_obs = []
            full_sens_labels = []
            known_ws = []
            k = 0
            for i in range(len(naive_ex_results["all_relvent_observations"])):
                if k == (n_sentences + 1): break
                rel_sen = naive_ex_results["all_relvent_observations"][i]
                rel_sen_state = naive_ex_results["all_relvent_sampled_trajs_states"][i]
                ful_sen = naive_ex_results["all_full_sampled_trajs"][i]
                ful_sen_state = naive_ex_results["all_full_sampled_trajs_states"][i]
                known_w = naive_ex_results["known_ws"][i]

                if all([k in all_words for k in ful_sen]):
                    miss_sens_obs.append(rel_sen)
                    miss_sens_labels.append(rel_sen_state)
                    full_sens_obs.append(ful_sen)
                    full_sens_labels.append(ful_sen_state)
                    known_ws.append(known_w)
                    k = k + 1
            #     print(rel_sen_state)
            #     print(ful_sen_state)

            # relvent data
            miss_sens_obs = miss_sens_obs[:n_sentences]
            miss_sens_labels = miss_sens_labels[:n_sentences]

            full_sens_obs = full_sens_obs[:n_sentences]
            full_sens_labels = full_sens_labels[:n_sentences]
            known_ws = known_ws[:n_sentences]

            # run prediction
            pc = naive_ex_results["mutual_params"]["p_prob_of_observation"]
            _N_known = list(map(lambda x: naive_ex_results["mutual_params"]['N'], full_sens_obs))
            return_N_naive = lambda x: list(map(len, x))

            # known T, full sentences, naive infer
            xml_T_full_naive = EmSequenceLabeling.most_likely_path(full_sens_obs, T,
                                                                   start_probabilites,
                                                                   state_to_distrbution_param_mapping, Ng_iters=1,
                                                                   N=return_N_naive(full_sens_obs))
            #     #known T, missing sentences, naive infer
            xml_T_missing_naive = EmSequenceLabeling.most_likely_path(miss_sens_obs, T,
                                                                      start_probabilites,
                                                                      state_to_distrbution_param_mapping, Ng_iters=1,
                                                                      N=return_N_naive(miss_sens_obs))
            # known T, missing sentences, missing infer
            xml_T_missing_miss = EmSequenceLabeling.most_likely_path(miss_sens_obs, T,
                                                                     start_probabilites,
                                                                     state_to_distrbution_param_mapping, Ng_iters=100,
                                                                     N=_N_known)
            # known T, missing sentences, missing infer,known W
            xml_T_missing_miss_known_w = EmSequenceLabeling.most_likely_path(miss_sens_obs, T,
                                                                             start_probabilites,
                                                                             state_to_distrbution_param_mapping,
                                                                             Ng_iters=100,
                                                                             N=_N_known, W=known_ws)
            xml_T_emm_only = [[EmSequenceLabeling._closest_state(state_to_distrbution_param_mapping, obs, True) for
                               obs in sen] for sen in miss_sens_obs]

            # scores
            scores_list_T_full_naive = prediction_scores_all_sens(xml_T_full_naive, full_sens_labels)
            scores_list_T_missing_naive = prediction_scores_all_sens(xml_T_missing_naive, miss_sens_labels)
            scores_list_T_missing_miss = prediction_scores_all_sens(xml_T_missing_miss, miss_sens_labels)
            scores_list_T_missing_miss_known_w = prediction_scores_all_sens(xml_T_missing_miss_known_w,
                                                                            miss_sens_labels)
            score_only_emmis = prediction_scores_all_sens(xml_T_emm_only, miss_sens_labels)
            #
            res = {}
            res["scores_list_T_full_naive"] = scores_list_T_full_naive
            res["scores_list_T_missing_naive"] = scores_list_T_missing_naive
            res["scores_list_T_missing_miss"] = scores_list_T_missing_miss
            res["scores_list_T_missing_miss_known_w"] = scores_list_T_missing_miss_known_w
            res["score_only_emmis"] = score_only_emmis
            #
            return res, miss_sens_labels, full_sens_labels

        return_result_string = lambda x: f"{np.round(np.mean(x), 4)} +- ({np.round(np.std(x), 4)})"

        return_result_string = lambda x: f"{np.round(np.mean(x), 4)} +- ({np.round(np.std(x), 4)})"

        model_defining_params_pre = ['N', "d", "n_states", 'is_acyclic', 'sigma', 'bipartite', 'known_dataset']
        params_dict = {
                    'is_acyclic': [True],
                    'known_mues': [True],
                    "is_few_observation_model": [False],
                    "is_only_seen": [ "all"],
                    'N': [25],
                    'd': [-1],
                    "bipartite": [True],
                    "inner_outer_trans_probs_ratio": [10],
                    'n_states': [-1],
                    "known_dataset": ["POS"],
                    'sigma':  [0],
                    'number_of_smapled_traj': [2000],
                    'p_prob_of_observation': [0.5],
                    'N_itres': [0],
                    'is_mh': [False],
                    'w_smapler_n_iter': [100],
                    'is_known_W': [False],
                    "is_multi_process": [False],
                }

        all_models_results_pos = GibbsExperiment.run_multi_params_and_return_results(params_dict,model_defining_params_pre,skip_sampler=False)


        for unique_model_results in all_models_results_pos.values() :
            res_pos = run_pos_prediction_return_score(unique_model_results)

    def test_multiple_choice_knapsack(self):
        # data_df = pd.DataFrame(columns=["id", "bin", "Cost", "Points"],
        #                        data=[(0, 0, 15, 5),
        #                              (1, 0, 20, 10),
        #                              (2, 0, 20, 11),
        #                              (3, 1, 9, 7),
        #                              (4, 1, 8, 6),
        #                              (5, 1, 12, 20),
        #                              (6, 2, 12, 7),
        #                              (7, 2, 11, 6),
        #                              (8, 2, 1, 6)])

        data_df = pd.DataFrame(columns=["id", "bin", "Cost", "Points"],
                               data=[(0, 0, 7, 2),
                                     (1, 0, 3, 1),
                                     (2, 0, 5, 9),
                                     (3, 0, 8, 13),
                                     (4, 0, 17, 17),
                                     (5, 1, 21, 18),
                                     (6, 1, 31, 14),
                                     (7, 1, 12, 11),
                                     (8, 1, 29, 27),
                                     (9, 1, 11, 30),
                                     (10, 2, 1, 22),
                                     (11, 2, 33, 16),
                                     (12, 2, 30, 22),
                                     (13, 2, 21, 40),
                                     (14, 2, 1, 1),
                                     (15, 3, 2, 2),
                                     (16, 3, 3, 3),
                                     (17, 3, 4, 4),
                                     (18, 3, 5, 5),
                                     (19, 3, 20, 10),
                                     (20, 4, 20, 11),
                                     (21, 4, 9, 7),
                                     (22, 4, 8, 6),
                                     (23, 4, 12, 20),
                                     (24, 4, 12, 7)])

        def multi_choice_ksp(data_df, max_weight):
            n_bins = (len(data_df.bin.unique()))

            K = np.zeros((n_bins, max_weight + 1)) + np.finfo(float).eps
            # track = {i:{ii:[] for ii in range(max_weight+1)} for i in range(-1,n_bins)}
            track = np.zeros((data_df.shape[0], max_weight + 1))

            min_w = 0#data_df.groupby("bin").min().sum()['Cost']
            for i in range(0,n_bins):
                for w in range(min_w, max_weight + 1):
                    op_in_bin = data_df[data_df["bin"] == i]
                    options = []
                    options.append(K[i - 1][w])
                    for j in range(op_in_bin.shape[0]):
                        adj_w = w - op_in_bin.iloc[j]["Cost"]
                        rel_point = op_in_bin.iloc[j]["Points"]
                        opt = K[i - 1][adj_w] + rel_point if (
                                    (op_in_bin.iloc[j]["Cost"] < w) and (K[i - 1][adj_w] != 0)) else 0
                        options.append(opt)
                    best_option_j = np.argmax(options)
                    K[i][w] = options[best_option_j]

                    if best_option_j != 0 :
                        track[op_in_bin.iloc[best_option_j-1]["id"]][w] = op_in_bin.iloc[best_option_j-1]["Cost"]

            alloc_ind = [i for i, w in enumerate(track.sum(axis=0) < max_weight) if w][-1]
            alloc = [i for i in range(track.shape[0]) if track[i][alloc_ind] > 0]
            return K[-1][-1], alloc

        res = multi_choice_ksp(data_df, 64)
        data_df["picked"] = data_df['id'].isin(res[1])
        print(res)
        print("pass")



































