import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import random


class MetaNetworkSimulator():

    def __init__(self):
        pass

    def build_meta_network(self,args):
        #region args
        n_dim_of_chain = args["n_dim_of_chain"]
        n_of_chains = args["n_of_chains"]
        possible_number_of_walks = args["possible_number_of_walks"]
        number_of_possible_states_limit = args["number_of_possible_states_limit"]
        chance_to_be_in_path = args["chance_to_be_in_path"]
        prob_of_dim = args["prob_of_dim"]
        #endregion

        self.all_binarys = []
        self._generate_all_binarys(n_dim_of_chain, [None] * n_dim_of_chain, 0)
        imx_possible_walks = self._build_IX_mock(n_dim_of_chain, possible_number_of_walks)
        model_networks = self._build_model_networks(n_of_chains, n_dim_of_chain, self.all_binarys, imx_possible_walks)

        # all the naive comb of states across chains
        all_curr_state_comb_between_networks = itertools.product(*[network for network in model_networks.values()])

        # we belive we can filter base on the real data
        filtered_curr_state_comb_between_networks = (
            itertools.islice(all_curr_state_comb_between_networks, number_of_possible_states_limit))
        filtered_curr_state_comb_between_networks_1, filtered_curr_state_comb_between_networks_2 = itertools.tee(
            filtered_curr_state_comb_between_networks)

        # now we need to find all possible combinations of walks :
        # we start by building all comb of walks across chains for all comb of states across chain
        all_walks_across_chains = (
            map(lambda comb: [model_networks[net_idx][chain_state] for net_idx, chain_state in enumerate(comb)],
                filtered_curr_state_comb_between_networks_1))

        filt_comb_walks_across_chains = (map(
            lambda _walks_per_curr_comb: self._filter_comb_of_walks_across_chains(_walks_per_curr_comb, chance_to_be_in_path),
            all_walks_across_chains))

        # zip the combinations of states with the comb of walks
        state_comb_to_walks_comb = zip(filtered_curr_state_comb_between_networks_2, filt_comb_walks_across_chains)

        # now we filter out comb of walks and combs of states where there is no possible walk from this current state
        meta_state_comb_to_walks_comb = filter(lambda _walks: len(_walks[1]) > 0, state_comb_to_walks_comb)

        return meta_state_comb_to_walks_comb

    #region private methods

    def _generate_all_binarys(self,n, arr, i):
        if i == n:
            self.all_binarys.append(np.nonzero(arr.copy())[0].tolist())
            return

        arr[i] = 0
        self._generate_all_binarys(n, arr, i + 1)

        arr[i] = 1
        self._generate_all_binarys(n, arr, i + 1)

    def _build_IX_mock(self,n, possible_number_of_walks):
        imx_possible_walks = {}

        if (max(possible_number_of_walks) + 1 ) > n :
            raise Exception( f"there is no {max(possible_number_of_walks)} possible")

        cell_idx = 0
        while cell_idx < n :
            number_of_walks = np.random.choice(possible_number_of_walks)
            walks_idx = np.random.choice(range(n), number_of_walks, False)
            walks_idx = np.append(walks_idx, np.array(cell_idx))
            walks_idx = np.unique(walks_idx)

            if len(walks_idx) != (number_of_walks + 1) :
                continue

            imx_possible_walks[cell_idx] = walks_idx
            cell_idx = cell_idx + 1
        return imx_possible_walks

    def _build_network_from_IX(self,n, all_binarys, imx_possible_walks):
        network_dic = {}  # key:cell vec, value : all conncted cell vecs

        for cell_vec in all_binarys:
            if len(cell_vec) == 0:
                continue

            possible_walk_for_idx_matrix = [imx_possible_walks[cell_idx] for cell_idx in cell_vec]
            all_possible_walks_from_cell = [set(comb) for comb in itertools.product(*possible_walk_for_idx_matrix)]

            network_dic[tuple(cell_vec)] = all_possible_walks_from_cell
        return network_dic

    def _build_model_networks(self,n_of_chains, n, all_binarys, imx_possible_walks):
        model_networks = {}
        for i in range(n_of_chains):
            _net_walks = self._build_network_from_IX(n, all_binarys, imx_possible_walks)
            model_networks[i] = _net_walks
        return model_networks

    def _build_pathways_mock(self,imx_possible_walks):
        hard_walk_to_pathways_map = {}
        for vec in imx_possible_walks:
            path = np.random.choice([1, 2, 3, 4])
            self.hard_cells_to_pathways_map[tuple(vec)] = path
        return hard_walk_to_pathways_map

    def _is_walk_in_path(self , x, _path, chance_to_be_in_path):
        return random.random() < chance_to_be_in_path

    def _return_filtered_walks_per_curr_chain(self,_walks_per_curr_chain, path, chance_to_be_in_path):
        return list(map(lambda walk: list(filter(lambda x: self._is_walk_in_path(x, path, chance_to_be_in_path), walk)),
                        _walks_per_curr_chain))

    def _filter_comb_of_walks_across_chains(self,_walks_per_curr_comb,chance_to_be_in_path):
        all_combs_of_walks_all_paths = []
        for _path in [1, 2, 3, 4, 5]:

            # keep only walks in the pathway
            _walks_per_curr_comb_in_path = self._return_filtered_walks_per_curr_chain(_walks_per_curr_comb, _path,
                                                                                chance_to_be_in_path)

            # its smart to filter out first comb of walks where there is at least one chain with no walk in this pathway :
            if any([len(_walks) == 0 for _walks in _walks_per_curr_comb_in_path]):
                continue

            _combs_of_walks = list(itertools.product(*_walks_per_curr_comb_in_path))
            for _comb in _combs_of_walks:
                if _comb not in all_combs_of_walks_all_paths:
                    all_combs_of_walks_all_paths = all_combs_of_walks_all_paths + [_comb]
        return all_combs_of_walks_all_paths

    #endregion





if __name__ == '__main__':
    args = {"n_dim_of_chain":3,"n_of_chains":2,"possible_number_of_walks":[1, 2],
    "number_of_possible_states_limit":10000000,"chance_to_be_in_path":1.1,"prob_of_dim":0.7}

    mns = MetaNetworkSimulator()

    meta_network = mns.build_meta_network(args)

    print("finish")