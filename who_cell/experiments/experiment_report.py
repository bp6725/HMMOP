import itertools
import numpy as np
from tqdm import tqdm
from functools import reduce
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

sys.path.append('C:\Repos\pomegranate')

from pomegranate import *
import who_cell.transitions_dict
import who_cell
from who_cell.models.gibbs_sampler import GibbsSampler

class ExperimentReport() :

    def __init__(self):
        pass

    def __is_transition_exist(self, _from, _to, trans):
        if _from in trans.keys():
            return _to in trans[_from].keys()
        return False

    def _build_pome_model_from_collections(self, _mues, _trans, _params):
        if type(_params) is not dict :
            _params = {k: v for k, v in _params}
        states_track = {}
        _model = HiddenMarkovModel("test")
        for _d, _t in itertools.product(list(range(_params['d'])),
                                        list(range(_params['N']))):
            mu = _mues[_d, _t]
            #TODO: get std from outside
            _dist = NormalDistribution(mu, 1 / 10)
            _state = State(_dist, name=f"({_d},{_t})")
            _model.add_state(_state)
            states_track[f"({_d},{_t})"] = _state

        _model.add_state(_model.start)
        _model.add_state(_model.end)

        for _d, _t in itertools.product(list(range(_params['d'])),
                                        list(range(_params['N']))):
            for _dd, _tt in itertools.product(list(range(_params['d'])),
                                              list(range(_params['N']))):
                if (_tt - _t) != 1:
                    continue

                if self.__is_transition_exist((_d, _t), (_dd, _tt), _trans):
                    _from = states_track[f"({_d},{_t})"]
                    _to = states_track[f"({_dd},{_tt})"]
                    _weight = _trans[(_d, _t)][(_dd, _tt)]
                    #             print(f"({_d},{_t})" + '==>' + f"({_dd},{_tt})")
                    _model.add_transition(_from, _to, _weight)

        for _d, _t in itertools.product(list(range(_params['d'])),
                                        list(range(_params['N']))):

            if _t == 0:
                _model.add_transition(_model.start, states_track[f"({_d},{_t})"], 1)
            if _t == _params['N'] - 1:
                _model.add_transition(states_track[f"({_d},{_t})"], _model.end, 1)

        _model.bake(merge=None)

        return _model

    def _extrect_states_transitions_dict_from_pome_model(self, model, states=None):
        if states is None:
            states = model.get_params()['states']

        edges = model.get_params()['edges']
        transition_dict = {}
        final_states = []
        for e in edges:
            if ('start' in states[e[0]].name):
                continue
            if ('end' in states[e[1]].name):
                final_states.append(eval(states[e[0]].name))
                continue

            _from = eval(states[e[0]].name)
            _to = eval(states[e[1]].name)
            _weight = e[2]

            if _from not in transition_dict.keys():
                transition_dict[_from] = {_to: _weight}
            else:
                transition_dict[_from][_to] = _weight
        return transition_dict, final_states

    def _build_states_from_mues_matrix(self, mues, N, d):
        print("warning : this code assumes std = mu/10")
        states = {}
        for _d, _t in itertools.product(list(range(d)), list(range(N))):
            mu = mues[_d, _t]
            _dist = NormalDistribution(mu, mu / 10)
            _state = State(_dist, name=f"({_d},{_t})")
            states[(_d, _t)] = _state
        return states

    def _calculate_prob_of_sample(self, model, samples, param):
        if type(param) is not dict :
            _params = {k: v for k, v in param}
        else :
            _params = param
        if type(model) is HiddenMarkovModel:
            states = model.get_params()['states']
            transition_dict, final_states = self._extrect_states_transitions_dict_from_pome_model(model, states)

            _states_dict = {eval(s.name): s for s in states if not (('start' in s.name) or ('end' in s.name))}
            _states = list(transition_dict.keys()) + final_states

        else:
            print('hello')
            transition_dict = model[1]
            states = build_states_from_mues_matrix(model[0], _params['N'], _params['d'])

        gs = GibbsSampler(_params['N'], _params['d'])
        start_prob = gs._build_start_prob(transition_dict, _params['N'], _params['d'])
        results = []
        for sample in samples:
            emm = gs._build_emmisions_for_sample(sample, list(range(_params['N'])), _states_dict, _params['d'],
                                                 _params['N'], normalized_emm=False)
            prior = gs._fwd_bkw(_states, start_prob, transition_dict, emm, _params['N'], _params['d'],
                                only_forward=True)
            res = sum([sum(time_prior.values()) for time_prior in prior])
            results.append(res)
        return results

    def _kl_distances_over_original(self, original_model, sampled_models, params):
        samples_for_comperison = original_model.sample(1000)
        _org_prob = self._calculate_prob_of_sample(original_model, samples_for_comperison, params)

        results = []
        for _model_for_compr in sampled_models:
            _comp_prob = self._calculate_prob_of_sample(_model_for_compr, samples_for_comperison, params)
            _org_prob_log = np.log(_org_prob)
            _comp_prob_log = np.log(_comp_prob)

            kl_distnace = sum((_org_prob_log - _comp_prob_log) * _org_prob) / (
                        len(samples_for_comperison) * len(samples_for_comperison[0]))
            results.append(kl_distnace)
        return results

    def _insertion_error_over_original(self,transitions_all_iters ,original_model,params,return_comp_df = False) :
        comp_dfs = []
        insertion_error_for_iter = []
        for _iter in range(len(transitions_all_iters)):
            sampled_res = transitions_all_iters[_iter]
            real_res = self._extrect_states_transitions_dict_from_pome_model(original_model)[0]
            res = []
            for (_d, _n) in self.__iterate_over_states_idxs(params):
                for (_dd, _nn) in self.__iterate_over_states_idxs(params):
                    _from = (_d, _n)
                    _to = (_dd, _nn)

                    if _from in sampled_res.keys():
                        if _to in sampled_res[_from].keys():
                            _val = sampled_res[_from][_to]
                        else:
                            _val = 0
                    else:
                        _val = 0

                    if _from in real_res.keys():
                        if _to in real_res[_from].keys():
                            _val_real = real_res[_from][_to]
                        else:
                            _val_real = 0
                    else:
                        _val_real = 0

                    res.append([_from, _to, _val, _val_real])

            final_df_for_iter = pd.DataFrame(columns=['from', 'to', "sampled", 'real'], data=res)
            if return_comp_df :
                comp_dfs.append(final_df_for_iter)

            _insertion_error_for_iter = sum(
                ((final_df_for_iter["sampled"] != 0) & (final_df_for_iter["real"] == 0)))
            insertion_error_for_iter.append(_insertion_error_for_iter)

        if return_comp_df:
            return insertion_error_for_iter , comp_dfs
        return insertion_error_for_iter


    def __iterate_over_states_idxs(self,params):
        if type(params) is dict :
            return itertools.product(range(params["d"]), range(params["N"]))
        else :
            return itertools.product(range(params[1][1]), range(params[0][1]))


    def _plot_results(self,results_dict,sup_title = '',x_axis = '',y_axis = ''):
        _dim = int(len(results_dict.items()))
        fig, subs = plt.subplots(_dim, 1, figsize=(12, 12))
        fig.suptitle(sup_title)

        for (model_name, single_modle_reults), sub in zip(results_dict.items(), itertools.chain(subs)):
            model_results_df = pd.DataFrame(single_modle_reults)

            sns.lineplot(data=model_results_df, ax=sub, legend='full', dashes=False)
            sub.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            sub.set_title(model_name)
            sub.set_xlabel(x_axis)
            sub.set_ylabel(y_axis)

        plt.subplots_adjust(hspace=0.8)
        plt.subplots_adjust(wspace=0.8)

        plt.show()

    def __build_exp_name(self,params):
        if type(params) is tuple :
            return reduce(lambda x, y: str(x)+ '\n' + str(y), params)
        else :
            return reduce(lambda x, y: str(x) + str(y), params.items())


    def calculate_measure_over_all_results(self, all_results):
        all_measure_results = {}
        with tqdm(total=len(all_results.keys())) as pbar:
            for single_model_permuts in all_results.values():
                single_model_results = {}
                for unique_permut in single_model_permuts.values() :
                    params = {**unique_permut['mutual_params'], **unique_permut['hyper_params']}
                    original_model = unique_permut['original_pome_model']
                    sampled_models = [self._build_pome_model_from_collections(mues, trans, params) for
                                      mues, trans in zip(unique_permut['all_mues'], unique_permut['all_transitions'])]

                    measures_over_iters = self._kl_distances_over_original(original_model, sampled_models, params)

                    exp_name = self.__build_exp_name(unique_permut['hyper_params'])
                    single_model_results[exp_name] = measures_over_iters
                    pbar.update(1)

                model_name = self.__build_exp_name(unique_permut['mutual_params'])
                all_measure_results[model_name] = single_model_results

        return all_measure_results

    def calculate_insertion_error_over_all_permuts(self, all_results):
        all_measure_results = {}
        for single_model_permuts in all_results.values():
            single_model_results = {}
            for unique_permut in single_model_permuts.values():
                transitions_all_iters = unique_permut["all_transitions"]
                original_model = unique_permut["original_pome_model"]
                params = {**unique_permut['mutual_params'], **unique_permut['hyper_params']}

                insertion_error_for_iter = self._insertion_error_over_original(transitions_all_iters ,original_model,params)

                exp_name = self.__build_exp_name(unique_permut['hyper_params'])
                single_model_results[exp_name] = insertion_error_for_iter

            model_name = self.__build_exp_name(unique_permut['mutual_params'])
            all_measure_results[model_name] = single_model_results
        return all_measure_results

    def build_report_from_multiparam_exp_results(self,all_results,params_titles = None):
        all_measure_results = self.calculate_measure_over_all_results(all_results)
        insertion_error_per_permut = self.calculate_insertion_error_over_all_permuts(all_results)

        self._plot_results(all_measure_results, sup_title = 'kl distance')
        self._plot_results(insertion_error_per_permut, sup_title = 'insertion error')

        # self._plot_results(all_measure_results, params_titles['kl_distance']['title'],
        #                        params_titles['kl_distance']['x'],params_titles['kl_distance']['y'])
        # self._plot_results(insertion_error_per_permut , params_titles['insertion_error']['title'],
        #                        params_titles['insertion_error']['x'],params_titles['insertion_error']['y'])