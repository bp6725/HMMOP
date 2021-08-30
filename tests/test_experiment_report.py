from unittest import TestCase
from who_cell.simulation.simulation_for_gibbs import Simulator_for_Gibbs
from who_cell.models.gibbs_sampler import GibbsSampler
from who_cell.experiments.experiment_report import ExperimentReport
from who_cell.experiments.gibbs_experiments import GibbsExperiment

import matplotlib.pyplot as plt
import numpy as np
import string

from collections import Counter
from IPython.core.display import HTML
from itertools import chain
from collections import Counter, defaultdict
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution
import pandas as pd
import seaborn as sns
import itertools



class TestExperimentReport(TestCase):
    def test__kl_distances_over_original(self):
        def build_pome_for_pos_exp(trnasitions, start_probs, emms_probs):

            states_track = {}

            pome_model = HiddenMarkovModel()

            for pos, trans in emms_probs.items():
                dist = DiscreteDistribution(trans)
                state = State(dist, pos)

                pome_model.add_state(state)
                states_track[pos] = state

            n_states = len(states_track)
            for _from_pos, _from_s in states_track.items():
                for _to_pos, _to_s in states_track.items():
                    if _to_pos in trnasitions[_from_pos].keys():
                        pome_model.add_transition(_from_s, _to_s, trnasitions[_from_pos][_to_pos])

            for _pos, _s in states_track.items():
                pome_model.add_transition(pome_model.start, _s, start_probs[_pos])

            pome_model.bake()
            return pome_model
        er = ExperimentReport()

        import pickle

        with open('../benchmarks/POS_tagging/cache.pkl', 'rb') as f:
            [transitions_probs, start_probs, emms_probs, all_transitions] = pickle.load(f)

        original_model = build_pome_for_pos_exp(transitions_probs, start_probs, emms_probs)

        hmms_of_exp = [build_pome_for_pos_exp(_t, start_probs, emms_probs) for _t in all_transitions]
        results_dist = er._kl_distances_over_original(original_model, hmms_of_exp, {"is_acyclic": True, "N": 70},
                                                      start_probs, is_known_emm=True,number_of_samples=10)
        raise

        print("hello")
        self.fail()

    def test__calculate_measure_over_all_results(self):
        import pickle
        with open('../../WhoCell/few_observations/notebook/temp_cache', 'rb') as f:
            all_models_results = pickle.load(f)
        er = ExperimentReport()
        all_measure_results = er.calculate_measure_over_all_results(all_models_results)