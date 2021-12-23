import pandas as pd
import numpy as np
import pickle
import copy

class KnownTransition_matrices() :

    @staticmethod
    def load_data_set_by_name(name) :
        if (name == "SCHIZX1_drug") :
            return KnownTransition_matrices.get_Schizophrenia_matrix_drug()
        if (name == "SCHIZX1_plcebo") :
            return KnownTransition_matrices.get_Schizophrenia_matrix_plcebo()
        if (name == "POS") :
            return KnownTransition_matrices.get_POS_data()

        return None

    @staticmethod
    def get_Schizophrenia_matrix_drug() :
        transition_dict = pd.DataFrame(index=['A','B','C','D'], columns=['A','B','C','D'], data=[[0.982,0.06 ,0.06 ,0.06],
                                                                                [0.149,0.66 ,0.170,0.021],
                                                                                [0.082,0.447,0.365,0.106],
                                                                                [0.046,0.227,0.304,0.423]]).T.to_dict()
        emmisions = {f:{t:int(t==f) for t in ['A','B','C','D']} for f in ['A','B','C','D']}
        start_probs = {t:0.25 for t in ['A','B','C','D']}

        transition_dict["start"] = start_probs

        return emmisions,transition_dict,start_probs

    @staticmethod
    def get_Schizophrenia_matrix_plcebo():
        transition_dict = pd.DataFrame(index=['A', 'B', 'C', 'D'], columns=['A', 'B', 'C', 'D'],
                                       data=[[0.25, 0.25, 0.25, 0.25],
                                             [0.091, 0.636, 0.182, 0.091],
                                             [0.001, 0.361, 0.472, 0.166],
                                             [0.016, 0.078, 0.125, 0.781]]).T.to_dict()
        emmisions = {f: {t: int(t == f) for t in ['A', 'B', 'C', 'D']} for f in ['A', 'B', 'C', 'D']}
        start_probs = {t: 0.25 for t in ['A', 'B', 'C', 'D']}

        transition_dict["start"] = start_probs

        return emmisions,transition_dict,start_probs

    @staticmethod
    def get_POS_data():
        with open("../../who_cell/simulation/pos_transitions_probs",'rb') as f :
            transition_dict = pickle.load(f)
        with open("../../who_cell/simulation/start_probs",'rb') as f :
            start_probs = pickle.load(f)
        with open("../../who_cell/simulation/emissions",'rb') as f :
            emmisions = pickle.load(f)

        transition_dict["start"] = start_probs

        return emmisions, transition_dict, start_probs