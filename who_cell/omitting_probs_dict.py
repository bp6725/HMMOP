import os
import copy
import numpy as np



class omitting_probs_dict(dict) :
    def __init__(self):
        self.unknown_omittings = []
        self.known_omittings = []
        self.latent_omitiing_probs = None

    def __repr__(self):
        return super().__repr__() + f" ; contain unknown: {self.unknown_omittings}"

    @staticmethod
    def build_omitting_probs_dict(full_latent_dict, unknown_omittings_keys):
        omittings_with_unknowns = omitting_probs_dict()

        for k,v in full_latent_dict.items() :
            is_known = k not in unknown_omittings_keys
            omittings_with_unknowns[k] = v if is_known else None

            if is_known :
                omittings_with_unknowns.known_omittings.append(k)
            else :
                omittings_with_unknowns.unknown_omittings.append(k)

        omittings_with_unknowns.latent_omitiing_probs = full_latent_dict

        return omittings_with_unknowns

    # def fill_unknowns_with_none(self):
    #     for uko in self.unknown_omittings :
    #         super()[uko] = np.random.rand()



if __name__ == '__main__':
    pass
    # old_dict = {1:2,3:4,5:None}
    # tr_dict = omitting_probs_dict.create_from_another_dict(old_dict)
    # print(tr_dict)

