import itertools
import numpy as np
from tqdm import tqdm
from functools import reduce
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from multiprocessing import Pool
from functools import partial
import who_cell.config as base_config

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

    def __return_first_last_states(self,_trans,N) :
        firsts = []
        lasts = []
        for _from,tos in _trans.items() :
            for _to,_ in tos.items() :
                if _from[1]  == 0 :
                    firsts.append(_from)
                if _to[1] == 0 :
                    firsts.append(_to)
                if _from[1] == N-1:
                    firsts.append(_from)
                if _to[1] == N-1:
                    firsts.append(_to)
        return firsts,lasts

    def _build_pome_model_from_collections(self, _mues, _trans, _params,is_acyclic,unique_permut):
        if type(_params) is not dict :
            _params = {k: v for k, v in _params}

        states = list(_mues.keys())
        states_track = {}
        _model = HiddenMarkovModel("test")
        for state in states:
            if state in ['start','end'] : continue
            mu = _mues[state]
            #TODO: get std from outside - now its only the constent std case
            _dist = NormalDistribution(mu,unique_permut['sigmas'][0])
            _state = State(_dist, name=f"({state})")
            _model.add_state(_state)
            states_track[f"{state}"] = _state

        _model.add_state(_model.start)

        if not is_acyclic :
            _model.add_state(_model.end)

        for _from_state in states:
            for _to_state in states:
                if ((_from_state in ['start','end']) or (_to_state in ['start','end'])) : continue

                if (not is_acyclic) and ((_to_state[1] - _from_state[1]) != 1) :
                    continue

                if self.__is_transition_exist(_from_state, _to_state, _trans):
                    _from = states_track[f"{_from_state}"]
                    _to = states_track[f"{_to_state}"]
                    _weight = _trans[_from_state][_to_state]

                    _model.add_transition(_from, _to, _weight)

        if not is_acyclic :
            start_states,end_states = self.__return_first_last_states(_trans,_params['N'])
            for state in start_states:
                _model.add_transition(_model.start, states_track[f"{state}"], 1)
            for state in end_states:
                _model.add_transition(states_track[f"{state}"], _model.end, 1)
        else :
            for state in states :
                if state in ['start','end'] : continue
                _model.add_transition(_model.start ,states_track[f"{state}"], 1)

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
                try:
                    final_states.append(eval(states[e[0]].name))
                except:
                    final_states.append(states[e[0]].name)
                continue

            try :
                _from = eval(states[e[0]].name)
                _to = eval(states[e[1]].name)
            except :
                _from = states[e[0]].name
                _to = states[e[1]].name

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

    @staticmethod
    def _calculate_prob_single_sample(start_prob,state_to_distrbution_param_mapping,transition_dict,
                                      param,_params,_states,is_known_emm,sample):
        N = max(_params['N'], len(sample))
        emm = GibbsSampler._build_emmisions_for_sample(param['is_acyclic'], sample, list(range(len(sample))),
                                             state_to_distrbution_param_mapping, N, normalized_emm=False,
                                                       is_known_emm = is_known_emm)
        prior = GibbsSampler._fwd_bkw(param['is_acyclic'], _states, start_prob, transition_dict, emm, N, only_forward=True)

        return sum([sum(time_prior.values()) for time_prior in prior])

    def _calculate_prob_of_sample(self, model, samples, param,start_probabilites,is_known_emm = False):
        if type(param) is not dict :
            _params = {k: v for k, v in param}
        else :
            _params = param
        if type(model) is HiddenMarkovModel:
            states = model.get_params()['states']
            transition_dict, final_states = self._extrect_states_transitions_dict_from_pome_model(model, states)

            try :
                _states_dict = {eval(s.name): s for s in states if not (('start' in s.name) or ('end' in s.name))}
                _states = [eval(s.name) for s in states if not (('start' in s.name) or ('end' in s.name))]
                state_to_distrbution_param_mapping = {eval(s.name): s.distribution.parameters for s in states
                                                      if not (('start' in s.name) or ('end' in s.name))}
            except :
                _states_dict = {(s.name): s for s in states if not (('start' in s.name) or ('end' in s.name))}
                _states = [(s.name) for s in states if not (('start' in s.name) or ('end' in s.name))]
                state_to_distrbution_param_mapping = {(s.name): s.distribution.parameters[0] for s in states
                                                      if not (('start' in s.name) or ('end' in s.name))}
        else:
            print('hello')
            #transition_dict = model[1]
            #states = build_states_from_mues_matrix(model[0], _params['N'], _params['d'])

        gs = GibbsSampler(_params['N'])
        start_prob = gs._build_start_prob(_states_dict,start_probabilites)
        _partial_calculate_prob_single_sample = partial(ExperimentReport._calculate_prob_single_sample,start_prob,
                                                        state_to_distrbution_param_mapping,transition_dict,
                                                        param,_params,_states,is_known_emm)

        with Pool(base_config.n_cores) as p :
            results = p.map(_partial_calculate_prob_single_sample,samples)

        return results

    def _kl_distances_over_original(self, original_model, sampled_models, params,start_probabilites,is_known_emm = False,
                                    number_of_samples = 500,external_samples = None):
        if external_samples is None :

            if params['is_acyclic']:
                samples_for_comperison = original_model.sample(number_of_samples,length=params['N'])
            else :
                samples_for_comperison = original_model.sample(number_of_samples)
        else :
            samples_for_comperison = external_samples

        _org_prob = self._calculate_prob_of_sample(original_model, samples_for_comperison,
                                                   params, start_probabilites, is_known_emm=is_known_emm)

        results = []
        for _model_for_compr in sampled_models:
            _comp_prob = self._calculate_prob_of_sample(_model_for_compr, samples_for_comperison,
                                                        params,start_probabilites)
            _org_prob_log = np.log(_org_prob)
            _comp_prob_log = np.log(_comp_prob)

            kl_distnace = sum((_org_prob_log - _comp_prob_log) * _org_prob) / (
                        len(samples_for_comperison) * len(samples_for_comperison[0]))
            results.append(kl_distnace)
        return results

    def _insertion_error_over_original(self,transitions_all_iters ,original_model,params,states,return_comp_df = False) :
        real_res = self._extrect_states_transitions_dict_from_pome_model(original_model)[0]
        comp_dfs = []
        insertion_error_for_iter = []

        for _iter in range(len(transitions_all_iters)):
            sampled_res = transitions_all_iters[_iter]
            # real_res = self._extrect_states_transitions_dict_from_pome_model(original_model)[0]
            res = []
            for _s in states:
                for __s in states:
                    if _s in ['start','end'] or __s in ['start','end'] : continue
                    _from = _s
                    _to = __s

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

    def _plot_batch(self,results_dict,sup_title,x_axis,y_axis):
        _dim = int(len(results_dict.items()))
        fig, subs = plt.subplots(_dim, 1, figsize=(12, 12))
        fig.suptitle(sup_title)

        subs = subs if _dim > 1 else [subs]
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

    def _plot_results(self,results_dict,sup_title = '',x_axis = '',y_axis = ''):
        models_list = list(results_dict.keys())
        batches = np.array_split(models_list, np.ceil(len(models_list) / 3))

        for batch in batches :
            sub_result_dict = {k:results_dict[k] for k in batch}
            self._plot_batch(sub_result_dict,sup_title,x_axis,y_axis)

    def __build_exp_name(self,params):
        if len(params) == 0 :
            return 'only'
        if type(params) is tuple :
            return reduce(lambda x, y: str(x)+ '\n' + str(y), params)
        else :
            return reduce(lambda x, y: str(x) + str(y), params.items())

    def calculate_measure_over_all_results(self, all_results):
        all_measure_results = {}
        with tqdm(total=sum([len(v) for k,v in all_results.items()])) as pbar:
            for single_model_permuts in all_results.values():
                original_model = single_model_permuts[0]['original_pome_model']
                external_samples = original_model.sample(300,single_model_permuts[0]['mutual_params']["N"])

                single_model_results = {}
                for unique_permut in single_model_permuts.values() :
                    params = {**unique_permut['mutual_params'], **unique_permut['hyper_params']}
                    sampled_models = [self._build_pome_model_from_collections(mues, trans, params,unique_permut['is_acyclic'],unique_permut) for
                                        mues, trans in zip(unique_permut['all_mues'], unique_permut['all_transitions'])]

                    measures_over_iters = self._kl_distances_over_original(original_model, sampled_models,
                                                                           params,unique_permut['start_probabilites'],
                                                                           external_samples=external_samples)

                    exp_name = self.__build_exp_name(unique_permut['hyper_params'])
                    single_model_results[str(exp_name)] = measures_over_iters
                    pbar.update(1)

                model_name = self.__build_exp_name(unique_permut['mutual_params'])
                all_measure_results[str(model_name)] = single_model_results

        return all_measure_results

    def calculate_insertion_error_over_all_permuts(self, all_results):
        all_measure_results = {}
        for single_model_permuts in all_results.values():
            single_model_results = {}
            for unique_permut in single_model_permuts.values():
                transitions_all_iters = unique_permut["all_transitions"]
                original_model = unique_permut["original_pome_model"]
                params = {**unique_permut['mutual_params'], **unique_permut['hyper_params']}

                insertion_error_for_iter = self._insertion_error_over_original(transitions_all_iters ,original_model,
                                                                               params,list(unique_permut['all_states'][-1].keys()))

                exp_name = self.__build_exp_name(unique_permut['hyper_params'])
                single_model_results[str(exp_name)] = insertion_error_for_iter

            model_name = self.__build_exp_name(unique_permut['mutual_params'])
            all_measure_results[str(model_name)] = single_model_results
        return all_measure_results

    def _plot_transitions_compare(self,first_trans,second_trans,title,first_title,second_title):
        if type(first_trans) is not dict :
            first_trans = self._extrect_states_transitions_dict_from_pome_model(first_trans)[0]
            first_trans = {str(k): {str(kk): vv for kk, vv in v.items()} for k, v in first_trans.items()}
        first_trans = {str(k): {str(kk):vv for kk,vv in v.items()} for k, v in first_trans.items() if k not in ['start','end']}
        if type(second_trans) is not dict :
            second_trans = self._extrect_states_transitions_dict_from_pome_model(second_trans)[0]
            second_trans = {str(k): {str(kk): vv for kk, vv in v.items()} for k, v in second_trans.items()}
        second_trans = {str(k): {str(kk):vv for kk,vv in v.items()} for k, v in second_trans.items() if k not in ['start','end']}

        first_trans_df = pd.DataFrame(first_trans)
        second_trans_df = pd.DataFrame(second_trans)

        first_trans_df.index.name = 'from'
        first_trans_df.columns.name = 'to'
        second_trans_df.columns.name = 'to'
        second_trans_df.index.name = 'from'

        first_trans_df_stack = first_trans_df.stack().to_frame()
        second_trans_df_stack = second_trans_df.stack().to_frame()

        # first_trans_df = first_trans_df.sort_index().sort_index(1)
        # second_trans_df = second_trans_df.sort_index().sort_index(1)

        mutual_df_stack = first_trans_df_stack.merge(second_trans_df_stack, left_index=True, right_index=True,
                                                     how='outer').fillna(0)

        fig,subs = plt.subplots(2,1,figsize=(12, 12))
        fig.suptitle(title)

        sns.heatmap(mutual_df_stack['0_x'].unstack().fillna(0),ax=subs[0])
        sns.heatmap(mutual_df_stack['0_y'].unstack().fillna(0),ax=subs[1])

        subs[0].set_title(first_title)
        subs[1].set_title(second_title)

        plt.show()

        print('stop')

    def _plot_transitions_compare_over_all_permuts(self,all_results,params_to_plot) :
        for single_model_permuts in all_results.values():
            for unique_permut in single_model_permuts.values():
                combined_params = {**unique_permut['mutual_params'],**unique_permut['hyper_params']}
                _to_plot = {k:v for k,v in combined_params.items() if k in params_to_plot}
                self._plot_transitions_compare(unique_permut['original_pome_model'],
                                               unique_permut['all_transitions'][-1],str(_to_plot),
                                               "original","predicted")
               
    def build_report_from_multiparam_exp_results(self,all_results,params_titles = None):
        all_measure_results = self.calculate_measure_over_all_results(all_results)
        insertion_error_per_permut = self.calculate_insertion_error_over_all_permuts(all_results)

        self._plot_results(all_measure_results, sup_title = 'kl distance')
        self._plot_results(insertion_error_per_permut, sup_title = 'insertion error')
        self._plot_transitions_compare_over_all_permuts(all_results)
