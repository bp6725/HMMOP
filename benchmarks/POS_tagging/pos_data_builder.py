import sys
sys.path.append('../')
from helpers import show_model, Dataset
from collections import Counter
import string
from itertools import chain
import pandas as pd
from who_cell.models.gibbs_sampler import Simulator_for_Gibbs
from pomegranate import HiddenMarkovModel,DiscreteDistribution,State
import os
import pickle
import nltk
import snowballstemmer
import itertools


class PosDataBuilder() :
    STATE_ORDER_TO_PLOT = ['NOUN', 'DET', 'PRON', 'ADJ', 'ADP', 'VERB', 'ADV', 'NUM', 'X', 'PRT', 'CONJ']

    def __init__(self,tags_path = "data/tags-universal.txt",data_path = "data/brown-universal.txt"):
        self.row_data = Dataset(tags_path, data_path, train_test_split=0.8)
        self._set_of_pos = [pos for pos in self.row_data.tagset if pos != '.']

        self.test_set_words = None
        self.test_set_tags = None

        self.transitions_probs = None
        self.transitions_probs_df = None

        self.markovien_sentence_words = None
        self.markovien_sentence_tags = None

        self.markovien_few_obs_test_set_words = None
        self.markovien_few_obs_test_set_tags = None

        self.few_obs_test_set_words, self.few_obs_test_set_tags = None,None

        self.emms_probs = None
        self.start_probs = None

        self.train_set_words = None
        self.train_set_tags = None

    def get_experiment_sets_from_real_data(self, pc, partial_trajs=True) :
        test_set_words, test_set_tags,test_set_ws = self.get_experiment_test_sets_from_real_data(pc,partial_trajs)
        train_set_words, train_set_tags= self.get_experiment_train_sets_from_real_data()
        return test_set_words, test_set_tags,train_set_words, train_set_tags
        print("here")

    def get_experiment_train_sets_from_real_data(self):
        if self.train_set_words is None :
            train_set_words, train_set_tags = self.build_clean_train_sets()
            self.train_set_words = train_set_words
            self.train_set_tags = train_set_tags
        else :
            train_set_words = self.train_set_words
            train_set_tags = self.train_set_tags

        return train_set_words, train_set_tags

    def get_experiment_test_sets_from_real_data(self,pc,partial_trajs = True):
        if self.test_set_words is None :
            test_set_words, test_set_tags = self.build_clean_test_sets()
        else :
            test_set_words = self.test_set_words
            test_set_tags = self.test_set_tags

        if not partial_trajs :
            return test_set_words,test_set_tags,None

        if self.few_obs_test_set_words is None :
            few_observations = Simulator_for_Gibbs.sample_traj_for_few_obs(pc, test_set_words)
            few_obs_test_set_words,few_obs_test_set_ws = few_observations[0], few_observations[1]
            few_obs_test_set_tags = [[sentence_tags[w] for w in ws] for (ws,sentence_tags) in zip(few_obs_test_set_ws,test_set_tags)]
            self.few_obs_test_set_words, self.few_obs_test_set_tags,self.few_obs_test_set_ws = few_obs_test_set_words,\
                                                                                               few_obs_test_set_tags,\
                                                                                               few_obs_test_set_ws
        else :
            few_obs_test_set_words, few_obs_test_set_tags,few_obs_test_set_ws = self.few_obs_test_set_words,\
                                                                                self.few_obs_test_set_tags,\
                                                                                self.few_obs_test_set_ws

        return few_obs_test_set_words,few_obs_test_set_tags,few_obs_test_set_ws

    def get_experiment_sets_from_markovien_sets(self,pc,length,partial_trajs = True):
        if self.markovien_sentence_words is None :
            transitions_probs,_ = self.get_known_transitions()

            path = "simulated_for_markov_pos.pkl"
            if not os.path.exists(path) :
                start_probs, emms_probs = self._build_starting_probabilites(), self._build_emissions_probabilites()
                known_model_as_pome = PosDataBuilder.build_pome_for_pos_exp(transitions_probs,start_probs, emms_probs)
                sampled_trajs_with_state = known_model_as_pome.sample(n=6000, length=length, path=True)

                with open(path,'wb') as f :
                    pickle.dump(sampled_trajs_with_state,f)
            else :
                with open(path,'rb') as f :
                    sampled_trajs_with_state = pickle.load(f)

            markovien_sentence_words = [[obs for obs in traj[0]] for traj in sampled_trajs_with_state]
            markovien_sentence_tags = [[obs.name for obs in traj[1] if 'start' not in obs.name] for traj in
                                       sampled_trajs_with_state]

            self.markovien_sentence_words = markovien_sentence_words
            self.markovien_sentence_tags = markovien_sentence_tags

        if not partial_trajs :
            return self.markovien_sentence_words,self.markovien_sentence_tags

        few_observations = Simulator_for_Gibbs.sample_traj_for_few_obs(pc, self.markovien_sentence_words)
        markovien_few_obs_test_set_words, markovien_few_obs_test_set_tags = few_observations[0], few_observations[1]
        self.markovien_few_obs_test_set_words, self.markovien_few_obs_test_set_tags = markovien_few_obs_test_set_words, markovien_few_obs_test_set_tags

        return markovien_few_obs_test_set_words, markovien_few_obs_test_set_tags

    def _build_emissions_probabilites(self):
        if self.emms_probs is not None:
            return self.emms_probs

        emms_counts = {}  # word => POS mapping

        train_set_words, train_set_tags = self.get_experiment_train_sets_from_real_data()
        for _word, _pos in zip(itertools.chain(*train_set_words),itertools.chain(*train_set_tags)):
            if _pos == '.': continue
            if _pos not in emms_counts.keys():
                emms_counts[_pos] = {_word: 1}
                continue
            if _word not in emms_counts[_pos]:
                emms_counts[_pos][_word] = 1
                continue
            emms_counts[_pos][_word] += 1

        emms_probs = {word: {pos: (count / sum(poss.values())) for pos, count in poss.items()} for word, poss in
                  emms_counts.items()}

        self.emms_probs = emms_probs
        return emms_probs

    def _build_starting_probabilites(self):
        if self.start_probs is not None :
            return self.start_probs

        start_count = {pos: 0 for pos in self._set_of_pos}
        _, train_set_tags = self.get_experiment_train_sets_from_real_data()
        for sentences_tags in train_set_tags:
            start_count[sentences_tags[0]] += 1

        start_probs = {pos: count / sum(start_count.values()) for pos, count in start_count.items()}

        self.start_probs = start_probs
        return start_probs

    def build_clean_test_sets(self):
        stemmer = snowballstemmer.stemmer('english')
        num_to_symbol = lambda x: x if not x.isnumeric() else "#"
        non_relevent_words = string.punctuation + '``' + '.' + '--' + "''" + ','

        test_set_words = []
        test_set_tags = []

        for idx, sentence in self.row_data.testing_set:
            clean_tuples = [(num_to_symbol(stemmer.stemWord(word)), tag) for word, tag in zip(sentence.words, sentence.tags) if
                            word not in non_relevent_words]
            if len(clean_tuples) < 2: continue
            test_set_words.append([word for word, tag in clean_tuples])
            test_set_tags.append([tag for word, tag in clean_tuples])

        self.test_set_words = test_set_words
        self.test_set_tags = test_set_tags

        return test_set_words,test_set_tags

    def build_clean_train_sets(self):
        stemmer = snowballstemmer.stemmer('english')
        num_to_symbol = lambda x: x if not x.isnumeric() else "#"
        non_relevent_words = string.punctuation + '``' + '.' + '--' + "''" + ','

        training_set_words = []
        training_set_tags = []

        for idx, sentence in self.row_data.training_set:
            clean_tuples = [(num_to_symbol(stemmer.stemWord(word)), tag) for word, tag in zip(sentence.words, sentence.tags) if
                            word not in non_relevent_words]
            if len(clean_tuples) < 2: continue
            training_set_words.append([word for word, tag in clean_tuples])
            training_set_tags.append([tag for word, tag in clean_tuples])

        self.training_set_words = training_set_words
        self.training_set_tags = training_set_tags

        return training_set_words,training_set_tags

    def get_known_transitions(self):
        if self.test_set_tags is None :
            _,test_set_tags = self.build_clean_test_sets()
        else :
            test_set_tags = self.test_set_tags

        transitions_count_tuples = Counter(chain(*[list(zip(sen, sen[1:])) for sen in test_set_tags]))
        transitions_count_dict = {}
        count_sum = {}
        for (_from, _to), count in transitions_count_tuples.items():
            if _from not in transitions_count_dict.keys():
                transitions_count_dict[_from] = {_to: count}
                count_sum[_from] = count if _from not in count_sum.keys() else count_sum[_from] + count
                continue
            if _to not in transitions_count_dict[_from].keys():
                transitions_count_dict[_from][_to] = count
                count_sum[_from] = count if _from not in count_sum.keys() else count_sum[_from] + count
                continue
        transitions_probs = {k: {kk: (vv / count_sum[k]) for kk, vv in v.items()} for k, v in
                             transitions_count_dict.items()}
        transitions_probs_df = pd.DataFrame(transitions_probs)

        self.transitions_probs = transitions_probs
        self.transitions_probs_df = transitions_probs_df

        return transitions_probs,transitions_probs_df

    @staticmethod
    def build_pome_for_pos_exp(trnasitions,start_probs, emms_probs):
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


if __name__ == '__main__':
    _builder = PosDataBuilder()
    _builder.get_experiment_sets_from_real_data(0.5,True)

    _builder._build_starting_probabilites()
    _builder._build_emissions_probabilites()