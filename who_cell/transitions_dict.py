import os
import copy



class transitions_dict(dict) :
    def __init__(self):
        self.null_transitions_dict = {}
        self.observed_transitions_dict  = {}
        self.extended_observed_transitions_dict  = {}

    def __repr__(self):
        return "all :" + super().__repr__() +os.linesep +"null :"+ self.null_transitions_dict.__repr__()+\
               os.linesep +"not null :"+ self.observed_transitions_dict.__repr__() +\
               os.linesep + "extended : " +self.extended_observed_transitions_dict.__repr__()


    def update_with_none(self,_from,_to,value = 1,is_null=False):
        if _to in self[_from].keys():
            self[_from][_to] += value
        else:
            self[_from][_to] = value

        if is_null:
            if _to in self.null_transitions_dict[_from].keys():
                self.null_transitions_dict[_from][_to] += value
            else:
                self.null_transitions_dict[_from][_to] = value
        else:
            if _to in self.observed_transitions_dict[_from].keys():
                self.observed_transitions_dict[_from][_to] += value
            else:
                self.observed_transitions_dict[_from][_to] = value

    def update_extended_seen_transitions(self,list_of_seen_transitions):
        for seen_trans in list_of_seen_transitions :
            _f = seen_trans[0]
            _t = seen_trans[1]

            if _f not in self.extended_observed_transitions_dict.keys() :
                self.extended_observed_transitions_dict[_f] = {_t:1}
                continue
            if _t not in self.extended_observed_transitions_dict[_f].keys() :
                self.extended_observed_transitions_dict[_f][_t] = 1
                continue
            self.extended_observed_transitions_dict[_f][_t] += 1


    @staticmethod
    def create_from_another_dict(other_dict):
        new_trans_dict = transitions_dict()
        for k,v in other_dict.items() :
            new_trans_dict[copy.deepcopy(k)] = copy.deepcopy(v)
            new_trans_dict.null_transitions_dict[copy.deepcopy(k)] = copy.deepcopy(v)
            new_trans_dict.observed_transitions_dict[copy.deepcopy(k)] = copy.deepcopy(v)
            new_trans_dict.extended_observed_transitions_dict[copy.deepcopy(k)] = copy.deepcopy(v)
        return new_trans_dict


if __name__ == '__main__':
    tr_dict = transitions_dict()

    tr_dict[1] = 2
    tr_dict.null_transitions_dict[1] = 2
    tr_dict.update_with_none({11:22},False)
    tr_dict.update_with_none({111: 222}, True)
    print(tr_dict)