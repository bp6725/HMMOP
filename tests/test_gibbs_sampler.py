from unittest import TestCase
from who_cell.models.gibbs_sampler import GibbsSampler

class TestGibbsSampler(TestCase):
    def test_sample_known_emissions(self):
        length_of_chain = 2
        number_of_states_in_time = 5

        gs = GibbsSampler(length_of_chain,number_of_states_in_time)

        start_probs = {f"state_{i}":0.2 for i in range(5)}
        emissions_table = {}
        emissions_table['state_0'] = {'0': 0.6, '1': 0.1, '2': 0.1, '3': 0.1, '4': 0.1}
        emissions_table['state_1'] = {'0': 0.1, '1': 0.6, '2': 0.1, '3': 0.1, '4': 0.1}
        emissions_table['state_2'] = {'0': 0.1, '1': 0.1, '2': 0.6, '3': 0.1, '4': 0.1}
        emissions_table['state_3'] = {'0': 0.1, '1': 0.1, '2': 0.1, '3': 0.6, '4': 0.1}
        emissions_table['state_4'] = {'0': 0.1, '1': 0.1, '2': 0.1, '3': 0.1, '4': 0.6}

        all_relvent_observations = [['0','0','1','2','2','1','0','2','4','1','0','3','1','4','0'],
                                    ['1','1','2','4','3','4','2','2','1','1','0','3','1','4','0'],
                                    ['0','0','1','2','2','1','0','2','4','1','0','3','1','4','0'],
                                    ['0', '0', '1', '2', '2', '1', '0', '2', '4', '1', '0','3', '1', '4', '0'],
                                    ['4','1','1','3','2','1','2','3','1','0','0','3','1','4','0'],
                                    ['0','1','4','2','2','1','0','3','4','1','1','3','1','4','0']]
        N = [len(seq) +1 for seq in all_relvent_observations]
        res = gs.sample_known_emissions(all_relvent_observations,start_probs,emissions_table,10,20,N=N)
