from unittest import TestCase
from who_cell.models.em_sequence_labeling import EmSequenceLabeling
import pickle

class TestEmSequenceLabeling(TestCase):
    def test_sequence_labeling_known_emissions(self):
        with open('all you need for pos infernce.pkl','rb') as f:
            (N,
             transitions,
             test_set_words,
             test_set_tags,
             words_emms_probs,
             start_probs, all_states) = pickle.load(f)

        test_set_words = list(filter(lambda x: len(x)< 100,test_set_words))
        N = list(map(lambda x:2*len(x),test_set_words))
        predicted = EmSequenceLabeling.sequence_labeling_known_emissions(test_set_words[:200],transitions, start_probs,
                               words_emms_probs, 5,N = N)

