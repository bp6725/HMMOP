from functools import reduce
from functools import wraps
import numpy as np
import pomegranate
import math


class Utils() :
    @staticmethod
    def update_based_on_alpha(func):
        @wraps(func)
        def _update_based_on_alpha(*args, **kwargs):
            if "stage_name" not in kwargs.keys() :
                return func(*args, **kwargs),None

            if kwargs["stage_name"] == "no_mh" :
                func_inputs = {k: v for k, v in kwargs.items() if k not in ["curr_params", "stage_name", "observations"]}
                return func(*args, **func_inputs),0

            stage_name = kwargs["stage_name"]
            observations = kwargs["observations"]
            func_inputs = {k: v for k, v in kwargs.items() if k not in ["curr_params", "stage_name", "observations"]}
            new_vel = func(*args, **func_inputs)

            curr_trans, curr_w, curr_walk, curr_mue, known_emissions = kwargs["curr_params"]
            new_trans = new_vel if stage_name == "transitions" else curr_trans
            new_w = new_vel if stage_name == "w" else curr_w
            new_walk = new_vel if stage_name == "walk" else curr_walk
            new_mue = new_vel if stage_name == "mue" else curr_mue

            is_known_emissions = type(known_emissions[curr_walk[0][0]]) is dict

            old_prob = Utils._calc_probability_function_for_alpha(observations, curr_trans, curr_w, curr_walk,
                                                                         curr_mu=curr_mue,
                                                                         known_emissions=known_emissions,
                                                                  is_known_emissions = is_known_emissions)
            new_prob = Utils._calc_probability_function_for_alpha(observations, new_trans, new_w, new_walk,
                                                                         curr_mu=new_mue,
                                                                         known_emissions=known_emissions,
                                                                  is_known_emissions = is_known_emissions)

            alpha = new_prob / old_prob

            if np.random.rand() < alpha:
                return new_vel,alpha
            else:
                return {"transitions": curr_trans,
                        "w": curr_w,
                        "walk": curr_walk,
                        "mue": curr_mue}[stage_name],alpha

        return _update_based_on_alpha

    @staticmethod
    def _calc_probability_function_for_alpha(all_relvent_observations, curr_trans, curr_w, curr_walk,
                                             curr_mu=None, known_emissions=None,is_known_emissions = True):
        probs_per_traj = []
        for i, traj in enumerate(all_relvent_observations):
            _w = curr_w[i]
            _walk = curr_walk[i]

            _prob = Utils.__calc_probability_function_per_traj(traj, curr_trans, _w, _walk, curr_mu, known_emissions,is_known_emissions)
            probs_per_traj.append(_prob)
        return sum(probs_per_traj)

    @staticmethod
    def normpdf(x, mean, sd):
        var = float(sd) ** 2
        denom = (2 * math.pi * var) ** .5
        num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
        return num / denom

    @staticmethod
    def multiply(vec):
        x = 1
        for val in vec:
            x = x * val
        return x

    @staticmethod
    def __calc_probability_function_per_traj(traj, _trans, _w, _walk,
                                             curr_mu=None, known_emissions=None,is_known_emissions = True):
        if known_emissions is not None:
            if is_known_emissions :
                emissions_prob = [known_emissions[_walk[__w]][traj[k]] for k,__w in enumerate(_w)]
            else :
                emissions_prob = [Utils.normpdf(traj[k],known_emissions[_walk[__w]][0],known_emissions[_walk[__w]][1])
                                  for k, __w in enumerate(_w)]
        elif curr_mu is not None:
            raise NotImplementedError("lazy you")
        transitions_prob = [(_trans[_f][_t] if _t in _trans[_f].keys() else 0) for _f, _t in zip(_walk, _walk[1:])]

        return reduce(lambda x, y: x * y, emissions_prob) * reduce(lambda x, y: x * y, transitions_prob)

