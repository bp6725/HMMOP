from functools import reduce
from collections import Counter
import pomegranate
from pomegranate import *
import numpy as np
import math
import matplotlib.pyplot as plt
from functools import partial
import itertools
import random
import copy
import pandas as pd
import itertools
import networkx as nx
import holoviews as hv
from scipy.stats import binom
import pickle as pkl
from ast import literal_eval as make_tuple
from IPython.display import display
import matplotlib.pyplot as plt
from toolz import unique
from who_cell.models.numerical_correction import NumericalCorrection

from numba import jit
import numba

from tqdm import tqdm
from who_cell.models.numerical_correction import NumericalCorrection
from who_cell.simulation.simulation_for_gibbs import Simulator_for_Gibbs
from who_cell.models.gibbs_sampler import GibbsSampler
from who_cell.experiments.experiment_report import ExperimentReport
import pickle

class GibbsExperiment() :
    def __init__(self,N,d,n_states,number_of_smapled_traj,p_prob_of_observation,N_itres):
        self.N = N
        self.d = d
        self.n_states = n_states
        self.number_of_smapled_traj = number_of_smapled_traj
        self.p_prob_of_observation = p_prob_of_observation
        self.N_itres = N_itres

    @staticmethod
    def run_multi_params_and_return_results(params_dict,model_defining_params_pre,skip_sampler = False) :
        defining_model_params = GibbsExperiment._build_defining_model_params(params_dict, model_defining_params_pre)

        sets_of_params_dicts = GibbsExperiment._build_sets_of_params_dicts(params_dict,defining_model_params)

        all_models_results = {}
        for model_idx,(mutual_model_params_dict, hyper_params_dict) in enumerate(sets_of_params_dicts) :
            results_of_mutual_model = GibbsExperiment.run_multi_params_mutual_model_and_return_results(mutual_model_params_dict,
                                                                                       hyper_params_dict,skip_sampler)
            all_models_results[model_idx] = results_of_mutual_model

        print('finish solving')
        return all_models_results

    @staticmethod
    def _return_params_sets(mutual_model_params_dict,hyper_params_dict):
        hyper_params_sets = list(itertools.product(*[[(k, vv) for vv in v] for k, v in hyper_params_dict.items()]))

        new_hyper_sets = []
        for set in hyper_params_sets:
            if (any(list(map(lambda x: (('is_few_observation_model' == x[0]) and (False == x[1])), set)))):
                new_set = (list(map(lambda x: (x[0], x[1] if x[0] != 'N_guess' else 1), set)))
            else:
                new_set = set
            new_hyper_sets.append(new_set)
        new_hyper_sets = list(map(list, unique(map(tuple, new_hyper_sets))))

        if 'number_of_smapled_traj' in hyper_params_dict.keys():
            max_number_of_sampled_traj = max([[vv for kk, vv in v if kk == 'number_of_smapled_traj'][0] for v in
                                              new_hyper_sets])
        else:
            max_number_of_sampled_traj = mutual_model_params_dict['number_of_smapled_traj']


        return new_hyper_sets, max_number_of_sampled_traj

    @staticmethod
    def _return_hyper_params_set(hyper_param_set, mutual_model_params_dict, simulator) :
        _hyper_param_set = {k: v for (k, v) in hyper_param_set}
        combined_params = {**_hyper_param_set, **mutual_model_params_dict}
        mues_for_sampler = simulator.states_known_mues if combined_params['known_mues'] else None
        sigmas_for_sampler = simulator.states_known_sigmas

        return _hyper_param_set, combined_params, mues_for_sampler, sigmas_for_sampler

    @staticmethod
    def run_multi_params_mutual_model_and_return_results(mutual_model_params_dict, hyper_params_dict,skip_sampler):
        all_results_of_model = {}

        hyper_params_sets, max_number_of_sampled_traj = GibbsExperiment._return_params_sets(mutual_model_params_dict,
                                                                                           hyper_params_dict)

        # simulate
        simulator = Simulator_for_Gibbs(mutual_model_params_dict['N'], mutual_model_params_dict['d'],
                                        mutual_model_params_dict['n_states'], easy_mode=True,
                                        max_number_of_sampled_traj =max_number_of_sampled_traj,
                                        sigma=mutual_model_params_dict['sigma']) # we need max_number_of_sampled_traj to know how much traj to pre sample so the traj will be mutual

        pome_results = simulator.build_pome_model(mutual_model_params_dict['N'], mutual_model_params_dict['d'],
                                       simulator.mues, simulator.sigmas,
                                       is_bipartite = mutual_model_params_dict['bipartite'],
                                       inner_outer_trans_probs_ratio = mutual_model_params_dict['inner_outer_trans_probs_ratio'],
                                                  mutual_model_params_dict = mutual_model_params_dict)

        simulator.update_known_mues_and_sigmes_to_state_mapping(pome_results["state_to_distrbution_param_mapping"])

        for exp_idx,hyper_param_set in enumerate(hyper_params_sets) :
            _hyper_param_set, combined_params, mues_for_sampler, sigmas_for_sampler = GibbsExperiment._return_hyper_params_set(hyper_param_set,
                                                                                                           mutual_model_params_dict,simulator )

            (all_relvent_observations, all_full_sampled_trajs, all_full_sampled_trajs_states,\
            all_relvent_sampled_trajs_states,known_ws),_ = \
                simulator.simulate_observations(pome_results["model"],combined_params,
                                                pome_results['params_signature'],from_pre_sampled_traj = True)

            _cache_path = ''.join([f"{str(k)[:3]}{str(v)[:3]}" for k,v in combined_params.items()])
            is_from_cache = skip_sampler and os.path.exists(_cache_path)
            if is_from_cache :
                with open(_cache_path,'rb') as f :
                    result = pickle.load(f)
            else :
                result = GibbsExperiment.solve_return_results_mutual_model(combined_params,
                                    pome_results,all_relvent_observations,mues_for_sampler,sigmas_for_sampler,
                                    w_smapler_n_iter = combined_params['w_smapler_n_iter'],known_w=known_ws)
                if skip_sampler :
                    with open(_cache_path,'wb') as f :
                        pickle.dump(result,f)
            if result is None : continue

            #region update result param
            _result = copy.copy(result)
            _result['mutual_params'] = mutual_model_params_dict
            _result['hyper_params'] = _hyper_param_set
            _result["all_full_sampled_trajs"] = all_full_sampled_trajs
            _result["all_full_sampled_trajs_states"] =  all_full_sampled_trajs_states
            _result["all_relvent_sampled_trajs_states"] = all_relvent_sampled_trajs_states
            _result['known_ws'] = known_ws
            _result["simulator"] = simulator
            _result["sigmas"] = simulator.sigmas
            _result["original_pome_model"] =  pome_results["model"]
            _result["state_to_distrbution_mapping"] =  pome_results["state_to_distrbution_mapping"]
            _result['start_probabilites']= pome_results['start_probabilites']
            #endregion

            all_results_of_model[exp_idx] = _result

            #region update results params if numerical reconstruction
            # if "numerical_reconstruction_pc" in combined_params.keys() :
            #     if combined_params["numerical_reconstruction_pc"] != -1 :
            #         _result = copy.copy(result)
            #
            #         _result['all_transitions'] = list(
            #             map(lambda x: NumericalCorrection.reconstruct_full_transitions_dict_from_few(x.copy(),
            #                                                                                          combined_params[
            #                                                                                              'numerical_reconstruction_pc'],
            #                                                                                          pome_results['start_probabilites']),
            #                 _result['all_transitions']))
            #
            #         _result['mutual_params'] = copy.copy(mutual_model_params_dict)
            #         _result['hyper_params'] = copy.copy(_hyper_param_set)
            #         _result["all_full_sampled_trajs"] = copy.copy(all_full_sampled_trajs)
            #         _result["all_full_sampled_trajs_states"] = copy.copy(all_full_sampled_trajs_states)
            #         _result["all_relvent_sampled_trajs_states"] = copy.copy(all_relvent_sampled_trajs_states)
            #         _result['known_ws'] = copy.copy(known_ws)
            #         _result["simulator"] = copy.copy(simulator)
            #         _result["sigmas"] = copy.copy(simulator.sigmas)
            #         _result["original_pome_model"] = copy.copy(pome_results["model"])
            #         _result["state_to_distrbution_mapping"] = copy.copy(pome_results["state_to_distrbution_mapping"])
            #         _result['start_probabilites'] = copy.copy(pome_results['start_probabilites'])
            #
            #         all_results_of_model[len(hyper_params_sets) + exp_idx] = _result
            #         # endregion

        return all_results_of_model

    @staticmethod
    def run_multi_params_and_plot_report(params_dict,model_defining_params_pre,skip_sampler,
                                         params_titles = None, mutual_model_params_titles = None):
        er = ExperimentReport()

        all_models_results = GibbsExperiment.run_multi_params_and_return_results(params_dict,model_defining_params_pre,skip_sampler)

        # er.build_report_from_multiparam_exp_results(all_models_results)

    @staticmethod
    def _is_valid_experiment(params):
        if ((params['is_few_observation_model'] == True) and (params['p_prob_of_observation'] == 1)):
            return False
        if ((params['is_few_observation_model'] == False) and (
                params['is_only_seen'] == "observed" or params['is_only_seen'] == "extended")):
            return False
        if ((params['is_few_observation_model'] == False) and (params["is_known_W"] == True)):
            return False
        if "numerical_reconstruction_pc" in params.keys() :
            if (((params['is_known_W'] == True) or (params['is_few_observation_model'] == False)) and (params["is_numerical_reconstruction_method"] )):
                return False
        if "PC_guess" in params.keys() and "N_guess" in params.keys() :
            if (params["PC_guess"] != -1) and (params["N_guess"] != -1):
                return False

        # is_few_obse_exp = (params["PC_guess"] != -1 if "PC_guess" in params.keys() else False) or \
        #                   (params["is_numerical_reconstruction_method"] if "is_numerical_reconstruction_method" in
        #                                                                    params.keys() else False) or (
        #                       params["N_guess"] != -1 if "N_guess" in params.keys() else False)
        # if is_few_obse_exp and (not params['is_few_observation_model']) :
        #     return False
        # if is_few_obse_exp and (params["is_known_W"]) :
            # return False

        return True

    @staticmethod
    def extrect_params(params,all_relvent_observations):
        transition_sampling_profile = params["is_only_seen"]

        if "N_guess" not in params.keys() :
            N = params['N']
        else :
            if params['N_guess'] != -1 :
                N = [int(len(v) * params["N_guess"]) for v in all_relvent_observations]
            else :
                N = params['N']

        N = N if  params['is_few_observation_model'] else 2
        sample_missing_with_prior  = params["sample_missing_with_prior"] if "sample_missing_with_prior" in params.keys() else False
        is_known_W = params['is_known_W'] if "is_known_W" in params.keys() else False
        is_multi_process = params['is_multi_process'] if "is_multi_process" in params.keys() else True
        use_pomegranate = params['use_pomegranate'] if "use_pomegranate" in params.keys() else False
        is_numerical_reconstruction = params[
            "is_numerical_reconstruction_method"] if "is_numerical_reconstruction_method" in params.keys() else False
        is_pc_guess = False if  "PC_guess" not in params.keys() else params["PC_guess"] != -1
        is_known_dataset = params["known_dataset"] != -1 if "known_dataset" in params.keys() else False

        return transition_sampling_profile,N,sample_missing_with_prior,is_known_W,\
               is_multi_process,use_pomegranate,is_numerical_reconstruction,is_pc_guess,is_known_dataset

    @staticmethod
    def _rechoose_n_iters(params):
        if type(params["p_prob_of_observation"]) is tuple : return params["N_itres"]
        if params["is_few_observation_model"] == False : return 5
        if params["p_prob_of_observation"] > 0.55 : return 50
        return params["N_itres"]

    @staticmethod
    def solve_return_results_mutual_model(params,pome_results,
                                          all_relvent_observations,mues_for_sampler,sigmas_for_sampler,
                                          w_smapler_n_iter = 100,known_w = None):

        transition_sampling_profile, N, sample_missing_with_prior,\
        is_known_W,is_multi_process,use_pomegranate,\
        is_numerical_reconstruction,is_pc_guess,is_known_dataset = GibbsExperiment.extrect_params(params,all_relvent_observations)

        st_par_map = pome_results['state_to_distrbution_param_mapping']
        is_known_emmi = type(st_par_map[list(st_par_map.keys())[0]]) is dict

        if not GibbsExperiment._is_valid_experiment(params) : return None
        print(params)

        # solve

        params["N_itres"] = GibbsExperiment._rechoose_n_iters(params)

        sampler = GibbsSampler(N, params['d'],transition_sampling_profile= transition_sampling_profile,
                               multi_process= is_multi_process)

        relevent_sampling_method = GibbsExperiment.__sampling_method_from_params(use_pomegranate,is_known_W,
                                                                                 is_numerical_reconstruction,is_pc_guess
                                                                                 )

        if relevent_sampling_method == "pomegranate" :
            _transitions = sampler.reconstruction_using_pomegranate(all_relvent_observations,
                                                                    pome_results["state_to_distrbution_param_mapping"],
                                                                    known_w=known_w)
            all_transitions = [_transitions for i in range(params['N_itres']+1)]
            all_states, all_observations_sum, all_sampled_transitions, all_mues, all_ws = (None, None, None, None, None)

            sampled_transitions_dict = None
            sampled_mues = None

        if relevent_sampling_method == "N from outside" :
            if not is_known_emmi :
                all_states, all_observations_sum, all_sampled_transitions, all_mues, all_ws, all_transitions = \
                    sampler.sample(all_relvent_observations, pome_results['start_probabilites'],
                                   mues_for_sampler, sigmas_for_sampler, params['N_itres'],
                                   w_smapler_n_iter=w_smapler_n_iter,
                                   is_mh=params["is_mh"])
            else :
                all_sampled_transitions, all_ws, all_transitions, all_states_picked_by_w, all_alphas = \
                    sampler.sample_known_emissions(all_relvent_observations, pome_results['start_probabilites'],
                                                   {k: v for k, v in
                                                    pome_results['state_to_distrbution_param_mapping'].items() if
                                                    k != 'start'},
                                                   Ng_iters=params['N_itres'],
                                                   w_smapler_n_iter=w_smapler_n_iter, is_mh=params["is_mh"])

            sampled_transitions_dict = all_sampled_transitions[-1]
            sampled_mues = None
            all_states = None
            all_observations_sum = None
            all_mues = None

        if relevent_sampling_method == "PC from outside":
            if not is_known_emmi :
                all_states, all_observations_sum, all_sampled_transitions, all_mues, all_ws, all_transitions = \
                    sampler.sample_guess_pc(all_relvent_observations, pome_results['start_probabilites'],
                                            mues_for_sampler, sigmas_for_sampler, params['N_itres'],
                                            PC_guess=params["PC_guess"],
                                            w_smapler_n_iter=w_smapler_n_iter,
                                            is_mh=params["is_mh"])
                sampled_transitions_dict = all_sampled_transitions[-1]
                sampled_mues = all_mues[-1]
            else :
                _, _, all_sampled_transitions, _, all_ws, all_transitions = \
                    sampler.sample_known_emissions_with_pc_guess(all_relvent_observations, pome_results['start_probabilites'],
                                                   {k: v for k, v in
                                                    pome_results['state_to_distrbution_param_mapping'].items() if
                                                    k != 'start'},
                                                   Ng_iters=params['N_itres'],
                                                   w_smapler_n_iter=w_smapler_n_iter,PC_guess=params["PC_guess"],
                                                                 is_mh=params["is_mh"])
                sampled_transitions_dict = all_sampled_transitions[-1]
                sampled_mues = None
                all_states = None
                all_observations_sum = None
                all_mues = None

        if relevent_sampling_method == "numerical reconstruction":
            sampler = GibbsSampler(2, params['d'], transition_sampling_profile="all",
                                   multi_process=is_multi_process)

            if not is_known_emmi :
                all_states, all_observations_sum, all_sampled_transitions, all_mues, all_ws, all_transitions = \
                        sampler.sample(all_relvent_observations, pome_results['start_probabilites'],
                                       mues_for_sampler, sigmas_for_sampler, 20,
                                       w_smapler_n_iter=1,
                                       is_mh=params["is_mh"])
            else :
                all_sampled_transitions, all_ws, all_transitions, all_states_picked_by_w, all_alphas = \
                    sampler.sample_known_emissions(all_relvent_observations, pome_results['start_probabilites'],
                                                   {k: v for k, v in
                                                    pome_results['state_to_distrbution_param_mapping'].items() if
                                                    k != 'start'},
                                                   Ng_iters=20,
                                                   w_smapler_n_iter=1, is_mh=params["is_mh"])

            sampled_transitions_dict = all_sampled_transitions[-1]
            sampled_mues = None
            all_states = None
            all_observations_sum = None
            all_mues = None

            all_transitions = list(
                        map(lambda x: NumericalCorrection.reconstruct_full_transitions_dict_from_few(x.copy(),
                                                                                                     params[
                                                                                                         'numerical_reconstruction_pc'],
                                                                                                     pome_results['start_probabilites']),
                            all_transitions))

        if relevent_sampling_method == "Known W":
            if not  is_known_emmi :
                all_states, all_observations_sum, all_sampled_transitions, all_mues, all_ws, all_transitions = \
                    sampler.sample_known_W(all_relvent_observations, pome_results['start_probabilites'],
                                           mues_for_sampler, sigmas_for_sampler, params['N_itres'], known_w,
                                           w_smapler_n_iter=w_smapler_n_iter,
                                           is_mh=params["is_mh"])
                sampled_transitions_dict = all_sampled_transitions[-1]
                sampled_mues = all_mues[-1]
            else :
                all_sampled_transitions, _, all_transitions, all_states_picked_by_w, _ = \
                    sampler.sample_known_emissions_known_W( all_relvent_observations, pome_results['start_probabilites'],
                                                {k: v for k, v in
                                                 pome_results['state_to_distrbution_param_mapping'].items() if
                                                 k != 'start'},known_w, params['N_itres'],
                                                w_smapler_n_iter = w_smapler_n_iter,N = None,is_mh=params["is_mh"])
                all_states = None
                all_observations_sum = None
                all_mues=None
                all_ws = None
                sampled_transitions_dict = all_sampled_transitions[-1]
                sampled_mues = None

        results = {"all_relvent_observations": all_relvent_observations,
                   "sampler": sampler,
                   "all_states": all_states,
                   "all_observations_sum": all_observations_sum,
                   "all_sampled_transitions": all_sampled_transitions,
                   "all_transitions": all_transitions,
                   "all_mues": all_mues,
                   "all_ws": all_ws,
                   "sampled_transitions_dict": sampled_transitions_dict,
                   "sampled_mues": sampled_mues
                    }
        print("finish")
        return results

    @staticmethod
    def _build_defining_model_params(params_dict, model_defining_params_pre):
        single_option_parmas = [param for param, vals in params_dict.items() if len(vals) == 1]
        model_defining_params_pre += single_option_parmas
        return list(set(model_defining_params_pre))

    @staticmethod
    def _build_sets_of_params_dicts(params_dict, defining_model_params) :
        defining_params_dict = {k: v for k, v in params_dict.items() if k in defining_model_params}
        defining_params_sets = list(
            itertools.product(*[[(k, vv) for vv in v] for k, v in defining_params_dict.items()]))

        non_defining_params_dict = {k: v for k, v in params_dict.items() if k not in defining_model_params}

        sets_of_params_dicts = list(itertools.product(list(map(lambda x: {k: v for k, v in x},defining_params_sets)),
                                                      [non_defining_params_dict]))

        return sets_of_params_dicts

    @staticmethod
    def __sampling_method_from_params(use_pomegranate,is_known_W,is_numerical_reconstruction,is_pc_guess):
        if is_numerical_reconstruction :
            return "numerical reconstruction"

        if use_pomegranate:
            return "pomegranate"

        if (not is_known_W) and (not is_numerical_reconstruction) :
            return "N from outside" if (not is_pc_guess) else "PC from outside"

        if is_numerical_reconstruction :
            return "numerical reconstruction"

        if is_known_W:
            return "Known W"

    @staticmethod
    def measure_ws_correction(sampled_ws, known_W,N) :
        def mapping_distance(first,second) :
            res = [aa==bb for aa,bb in zip(first,second)]
            return (2*sum(res)-len(res))/len(res)

        all_correction = []
        all_random = []
        for sampled_w in sampled_ws :
            correction_measure = [mapping_distance(sampled, known) for sampled, known in zip(sampled_w, known_W)]
            random_measure = [ mapping_distance(list(sorted(np.random.choice(range(N), len(known), replace=False))),known)for known in  known_W]

            all_correction.append(correction_measure)
            all_random.append(random_measure)

        # last w histogram
        plt.hist(correction_measure)
        plt.hist(random_measure)
        plt.show()

        # convergence
        convergence_track_mean = [np.mean(w_measure) for w_measure in all_correction]
        convergence_track_std =  [ np.std(w_measure) for w_measure in all_correction]
        random_measures_mean = np.mean([np.mean(rand_measure) for rand_measure in all_random])
        random_measures_std  = np.mean([np.std(rand_measure) for rand_measure in all_random])

        plt.errorbar(list(range(len(convergence_track_mean))), convergence_track_mean, convergence_track_std,
                     linestyle='None', marker='^')
        plt.errorbar(list(range(len(convergence_track_mean))),
                     random_measures_mean * np.ones_like(convergence_track_mean),
                     random_measures_std * np.ones_like(convergence_track_std), linestyle='None', marker='^')
        plt.ylim([-1, 1])
        plt.show()

        plt.errorbar(list(range(len(convergence_track_mean))), convergence_track_mean, convergence_track_std,
                     linestyle='None', marker='^')

        return convergence_track_mean

    @staticmethod
    def build_full_mues_comper_df(all_mues,full_mues_from_known):
        full_mues_from_sampler = np.mean(all_mues, axis=0)

        full_mues_comper = []
        for state, time in itertools.product(range(full_mues_from_known.shape[0]),
                                             range(full_mues_from_known.shape[1])):
            full_mues_comper.append(
                [f"{(state, time)}", full_mues_from_known[state, time], full_mues_from_sampler[state, time]])

        return pd.DataFrame(columns=["state", "known", "sampled"], data=full_mues_comper)

    @staticmethod
    def build_final_transitions_comper_using_pome(all_transitions, pome_model) :
        _full_transitions_from_sampler = reduce(lambda a, b: {k: (Counter(a[k]) + Counter(b[k])) for k, _ in a.items()},
                                                all_transitions)
        full_transitions_known = pd.DataFrame(index=[s.name for s in pome_model.states],
                                              columns=[s.name for s in pome_model.states],
                                              data=pome_model.dense_transition_matrix())
        full_transitions_known = full_transitions_known.round(2)

        al_states = list(
            set([s for s in full_transitions_known.index] + [str(s) for s in _full_transitions_from_sampler.keys()]))
        final_transitions_comper = []

        for from_state in al_states:
            for to_state in al_states:
                is_in_known = ((from_state in full_transitions_known.index) and (
                            to_state in full_transitions_known.columns))
                known_count = full_transitions_known.loc[from_state].loc[to_state] if is_in_known else 0

                if ("start" in from_state) or ("end" in to_state) or ("start" in to_state) or ("end" in from_state):
                    continue

                # normalize sampler counts :
                _sum = sum(_full_transitions_from_sampler[make_tuple(from_state)].values())
                _sum = _sum if _sum != 0 else 1
                sampled_count = _full_transitions_from_sampler[make_tuple(from_state)][make_tuple(to_state)] / _sum

                if (known_count + sampled_count) == 0:
                    continue

                final_transitions_comper.append([from_state, to_state, known_count, sampled_count])

        final_transitions_comper_df = pd.DataFrame(columns=["from_state", "to_state", "known_count", "sampled_count"],
                                                   data=final_transitions_comper)
        final_transitions_comper_df.sort_values(by="from_state")
        return final_transitions_comper_df

    @staticmethod
    def kl_distances_over_original(original_model, sampled_models):
        samples_for_comperison = original_model.sample(300)

        results = []
        for _model_for_compr in sampled_models:
            _kl_function = lambda x: (original_model.log_probability(x) - _model_for_compr.log_probability(
                x)) * original_model.probability(x)
            kl_distnace = reduce(lambda x, y: y + _kl_function(x), samples_for_comperison + [0])
            results.append(kl_distnace)
        return results

    @staticmethod
    def _build_transition_df(_full_transitions_from_sampler, _known_transitions_summary):
        sampled_transition_df = pd.melt(pd.DataFrame(_full_transitions_from_sampler).reset_index(),
                                        id_vars="index"). \
            rename(columns={'index': "to", "variable": "from"})  # .fillna(0)
        known_transition_df = pd.melt(pd.DataFrame(_known_transitions_summary).reset_index(),
                                      id_vars="index"). \
            rename(columns={'index': "to", "variable": "from"})  # .fillna(0)

        final_transitions_comper_df = sampled_transition_df.merge(known_transition_df, left_on=("from", "to"),
                                                                  right_on=("from", "to"),
                                                                  suffixes=("_sampled", "_known"), how="outer").fillna(
            0)
        final_transitions_comper_df = final_transitions_comper_df[['from', 'to', 'value_known', 'value_sampled']]
        final_transitions_comper_df = final_transitions_comper_df[
            final_transitions_comper_df.apply(lambda row: (eval(row["to"])[1] - eval(row["from"])[1]) == 1, axis=1)]

        final_transitions_comper_df["from_time_reverse_tuple"] = final_transitions_comper_df.apply(
            lambda row: (eval(row["from"])[1], eval(row["from"])[0]),
            axis=1)
        final_transitions_comper_df["from_time_reverse_tuple"] = final_transitions_comper_df[
            "from_time_reverse_tuple"].astype(str)
        final_transitions_comper_df = final_transitions_comper_df.sort_values("from_time_reverse_tuple")

        final_transitions_comper_df = final_transitions_comper_df[final_transitions_comper_df.apply(
            lambda row: not (np.isnan(row["value_known"]) and np.isnan(row["value_sampled"])), axis=1)]

        return final_transitions_comper_df

    @staticmethod
    def build_final_transitions_compr(all_transitions, known_transitions_summary,return_df = True) :
        if type(all_transitions) is list :
            _all_transitions_states_as_string = [{str(k).replace(' ',''):{str(kk).replace(' ',''):vv for kk,vv in v.items()}
                                                  for k,v in _transition.items()} for _transition in all_transitions]
            _full_transitions_from_sampler = reduce(lambda a, b: {k: (Counter(a[k]) + Counter(b[k])) for k, _ in a.items()},
                                                    _all_transitions_states_as_string)
        else :
            _full_transitions_from_sampler = {
                str(k).replace(' ', ''): {str(kk).replace(' ', ''): vv for kk, vv in v.items()} for k, v in
                all_transitions.items()}

        _normalized_full_transitions_from_sampler = \
            {k: {kk: (vv / sum(v.values())) for kk, vv in v.items()} for k, v in _full_transitions_from_sampler.items()}
        _normalized_known_transitions_summary = \
            {k: {kk: (vv / sum(v.values())) for kk, vv in v.items()} for k, v in known_transitions_summary.items()}

        _full_transitions_from_sampler = \
            {k: {kk: vv for kk, vv in v.items()} for k, v in _full_transitions_from_sampler.items()}
        _known_transitions_summary = \
            {k: {kk: vv  for kk, vv in v.items()} for k, v in known_transitions_summary.items()}

        if not return_df :
            return _normalized_full_transitions_from_sampler, _full_transitions_from_sampler

        final_normalized_transitions_comper_df = GibbsExperiment._build_transition_df(
            _normalized_full_transitions_from_sampler,
            _normalized_known_transitions_summary)

        final_transitions_comper_df = GibbsExperiment._build_transition_df(
            _full_transitions_from_sampler,
            _known_transitions_summary)


        return final_normalized_transitions_comper_df,final_transitions_comper_df

if __name__ == '__main__':

    mode = "single"

