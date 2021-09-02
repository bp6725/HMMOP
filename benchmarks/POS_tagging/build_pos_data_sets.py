from helpers import show_model, Dataset



class BuildPos_Data_Sets() :
    def __init__(self,tags_path = "data/tags-universal.txt",data_path = "data/brown-universal.txt"):
        self.row_data = Dataset(tags_path, data_path, train_test_split=0.8)

    def Plot_states(self):
        length_of_sentences,count_of_pos_in_sentence = self._plot_hist_length_of_sentences(self.row_data)
        #TODO: more if nedded
