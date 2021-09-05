import os
import pickle as pkl
from pathlib import Path

class Infras() :

    def __init__(self):
        pass

    @staticmethod
    def _return_cache_relevent_params(func,*args, **kwargs) :
        func_name = func.__name__
        if "bipartite" in func_name :
            N, d, mues, sigmas ,ratio = args
            return ['build_model_parameters',f'{N}', str(d), str(mues), str(sigmas),"bipartite",str(ratio)]
        if func_name == "simulate_observations" :
            return ['simulate_observations',args[2],f"p_{args[1]['p_prob_of_observation']}"]
        if ('build' in func_name) and ('model_parameters' in func_name) :
            N, d, mues, sigmas = args
            is_acyclic = "acyclic" if ('acyclic' in func_name) else 'not acyclic'
            return ['build_model_parameters',f'{N}', str(d), str(mues), str(sigmas),is_acyclic]

    @staticmethod
    def storage_cache(func):
        def _decorator(self, *args, **kwargs):

            params = Infras._return_cache_relevent_params(func,*args, **kwargs)

            params_signature = '_'.join(params)
            params_signature = params_signature.replace(' ', '').replace('.', '')
            cache_path = os.path.join(r"../../cache",
                                      params_signature + ".pkl")

            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    res = pkl.load(f)
            else:
                res = func(self, *args, **kwargs)
                with open(cache_path, "wb") as f:
                    pkl.dump(res, f)

            return res,'_'.join(params[1:])

        return _decorator