import sys
sys.path.insert(0,'C:\Repos\pomegranate')
from who_cell.simulation.meta_network_simulator import MetaNetworkSimulator
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import random
import functools
from functools import reduce
import copy

import networkx as nx
import holoviews as hv
from scipy.stats import spearmanr

import pomegranate
from pomegranate import HiddenMarkovModel ,State
from pomegranate.distributions import IndependentComponentsDistribution
from pomegranate.distributions import NormalDistribution,DiscreteDistribution

from who_cell.Utils import PomegranateUtils
from who_cell.simulation.pomegranate_network_model_builder import PomegranateNetworkModelBuilder

class SilentModel() :

    #region Public

    def return_all_possible_G_without_cycles(self,__G,pn_data_to_possible_and_connected_states_trans,all_walks_of_pn):
        lexicographical_order = self.build_data_based_lexicographical_order(all_walks_of_pn)

        def lexicographical_order_func(x):
            res = None
            if x in lexicographical_order.keys():
                res = lexicographical_order[x] + 1
            else:
                #         print("blo")
                res = max([v for v in lexicographical_order.values()]) + 2
                res = 0

            return res

        all_Gs = []
        return self.all_possible_G_without_cycles(__G, pn_data_to_possible_and_connected_states_trans,
                                            lexicographical_order_func, all_Gs, verbose=True)

    def build_silent_pomemodel(self,global_G,DAG_G,state_name_to_state_mapping,network_proprties = {"prob_of_dim" : 0.7, "n_dim_of_chain" : 10, "n_of_chains" :1 },with_silent_mode = True):
        pmb = PomegranateNetworkModelBuilder()
        model = pmb.build_silent_model(global_G,DAG_G,state_name_to_state_mapping,network_proprties,True)

        return model


    #endregion

    #region private

    def have_cycle(self,G):
        try:
            nx.algorithms.cycles.find_cycle(G)
            return True
        except:
            return False

    def calc_spearman_correlation_with_org_transitions(self,G, pn_data_to_possible_and_connected_states_trans,
                                                       lexicographical_order_func):
        res = random.choice([0.1, 0.2, 0.3, 0.4, 1])
        ordered_nodes = nx.algorithms.dag.lexicographical_topological_sort(G.copy(), key=lexicographical_order_func)

        node_to_order = {node: i for i, node in enumerate(ordered_nodes)}
        spearmans = [spearmanr(range(len(trans)), [node_to_order[str(sorted(tuple(st)))] for st in trans])[0] for trans
                     in pn_data_to_possible_and_connected_states_trans]

        #     print([[node_to_order[str(sorted(tuple(st)))] for st in trans] for trans in pn_data_to_possible_and_connected_states_trans])
        # print(np.average(spearmans))
        return np.average(spearmans)

    def is_sc_best(self,new_G, is_best, pn_data_to_possible_and_connected_states_trans, lexicographical_order_func, verbose = False):
        if is_best:
            if verbose:
                print("is best")
            return new_G, True

        sc = self.calc_spearman_correlation_with_org_transitions(new_G, pn_data_to_possible_and_connected_states_trans,
                                                            lexicographical_order_func)
        if sc > 0.8:
            if verbose:
                print("sc 1")
            return new_G, True
        else:
            if verbose:
                print("not sc1")
            return new_G, False

    def all_possible_G_without_cycles(self,_G, pn_data_to_possible_and_connected_states_trans, lexicographical_order_func,
                                      all_Gs=[], verbose=False, depth=0):
        #     print(depth)
        local_G = _G.copy()

        if not self.have_cycle(local_G):
            if verbose:
                print("finish")
            return local_G, False

        # all trns of cycles in current G
        cycles = nx.algorithms.cycles.find_cycle(local_G)

        for trns in cycles:
            if verbose:
                print(f"trns :{trns}")
            G = local_G.copy()

            # removing the nth trans
            G.remove_edge(trns[0], trns[1])

            # keep the recursition . is best = True if we have sc =1 alredy
            new_G, is_best = self.all_possible_G_without_cycles(G, pn_data_to_possible_and_connected_states_trans,
                                                           lexicographical_order_func, all_Gs, depth=depth + 1)

            #       which mean that no G was best in the pre loop
            if (not self.have_cycle(new_G)):
                new_G, is_best = self.is_sc_best(new_G, is_best, pn_data_to_possible_and_connected_states_trans,
                                            lexicographical_order_func, verbose)
                if is_best:
                    return new_G, is_best
                all_Gs.append(new_G)
            if verbose:
                print("?")

        return local_G, False

    def build_data_based_lexicographical_order(self,all_walks_of_pn):
        # node_to_order_point - count when source came before target in pn_data_to_possible_and_connected_states_trans
        node_to_order_point = []
        for walk in all_walks_of_pn:
            for i in range(len(walk)):
                for j in range(i, len(walk)):
                    if i != j:
                        node_to_order_point.append([str(sorted(tuple(walk[i]))), str(sorted(tuple(walk[j]))), 1])

        node_to_order_point_df = pd.DataFrame(data=node_to_order_point, columns=["source", "target", "count"])
        node_to_order_point_sum_df = node_to_order_point_df.groupby(by=["source", "target"])["count"].sum()
        pivot_node_to_order_point_sum_df = node_to_order_point_sum_df.reset_index().pivot_table(index="source",
                                                                                                columns="target",
                                                                                                values="count")

        all_columns = pd.Index(
            list(map(lambda x: str(sorted(tuple(x))), PomegranateUtils.make_unique(list(itertools.chain(*all_walks_of_pn))))))
        pivot_node_to_order_point_sum_df = pd.DataFrame(columns=all_columns, index=all_columns, data=None).fillna(
            pivot_node_to_order_point_sum_df).fillna(0)

        diff_order_point_df = pivot_node_to_order_point_sum_df - pivot_node_to_order_point_sum_df.T

        state_to_order = {}
        order = 0
        _diff_order_point_df = diff_order_point_df.copy(deep=True)
        while (not _diff_order_point_df.empty):
            always_largest = _diff_order_point_df.loc[(_diff_order_point_df >= 0).all(axis=1)].index.tolist()
            _diff_order_point_df = _diff_order_point_df.drop(index=always_largest, columns=always_largest)

            for st in always_largest:
                state_to_order[st] = order
            order += 1

            if len(always_largest) == 0:
                #             print(always_largest)
                #             print(_diff_order_point_df[(_diff_order_point_df<0)].max().max())
                _diff_order_point_df[(_diff_order_point_df < 0)] -= _diff_order_point_df[
                    (_diff_order_point_df < 0)].max().max()

            if order > 10000:
                raise Exception()
                print(_diff_order_point_df)

        return state_to_order

    #endregion