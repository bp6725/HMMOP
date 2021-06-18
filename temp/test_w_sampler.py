import numpy as np
import math
import matplotlib.pyplot as plt
from functools import partial
import itertools
import random
import copy
import pandas as pd
from numba import jit
from scipy.stats import binom
import pprint

import pomegranate
from tqdm import tqdm

def simulate_data():
    N = 50
    p_prob_of_observation = 0.5

    mues = list(range(N))*10
    sigmas = [1 for i in range(N)]

    distrbutions = [pomegranate.distributions.NormalDistribution(mues[i], sigmas[i]) for i in range(N)]
    full_sample = list(range(N)) #[dist.sample() for dist in distrbutions]

    binom_dist = binom(N, p_prob_of_observation)
    k = 30#binom_dist.rvs()

    partial_sample = [full_sample[i] for i in sorted(np.random.choice(range(N), k, replace=False))]

    return partial_sample,distrbutions

# def prob_maping_from_seqs(obs, dists, seq_to_step,all_steps,k):
#     dists = copy.copy(dists)
#     seq_to_step = copy.copy(seq_to_step)
#
#     if ((len(obs) > 0) and len(dists) == 0):
#         return 0
#
#     if len(obs) == 0:
#         return 1
#
#     curr_obs = obs[0]
#     obs_tail = obs[1:]
#
#     probs_per_map = {}
#     for dist_ind, dist in enumerate(dists):
#         dists_tail = dists[(dist_ind+1):]
#         pro_of_dist = dist.probability(curr_obs)
#
#         new_seq = seq_to_step + [(curr_obs,dist.parameters[0])]
#         pro_tail = prob_maping_from_seqs(obs_tail, dists_tail, new_seq ,all_steps,k )
#         pro_full =  pro_of_dist * pro_tail
#
#         probs_per_map[frozenset(new_seq)] = pro_full
#
#     for key,value in probs_per_map.items() :
#         if len(key) == k :
#             all_steps.append({key:value})
#
#     probs_sum = sum(probs_per_map.values())
#     if probs_sum == 0 : return 0
#
#     return probs_sum

def memoize(function):
  memo = {}
  def wrapper(*args):
    _f_arg = args[0][0] if len(args[0]) > 0 else "obs_end"
    _s_arg = args[1][0].parameters[0] if len(args[1]) > 0 else "time_end"
    short_args = f"{(_f_arg,_s_arg)}"
    if short_args in memo:
      return memo[short_args]
    else:
      rv = function(*args)
      memo[short_args] = rv
      return rv
  return wrapper

@memoize
def prob_maping_from_seqs(obs, dists):
    dists = copy.copy(dists)

    if ((len(obs) > 0) and len(dists) == 0):
        return {frozenset(["end"]):0}

    if len(obs) == 0:
        return {frozenset(["end"]):1}

    curr_obs = obs[0]
    obs_tail = obs[1:]

    new_tail = {}
    for dist_ind, dist in enumerate(dists):
        dists_tail = dists[(dist_ind+1):]
        pro_of_dist = dist.probability(curr_obs)

        if pro_of_dist < 0.001 : continue

        pro_tail = prob_maping_from_seqs(obs_tail, dists_tail )

        for key,value in pro_tail.items() :
            new_val = value * pro_of_dist
            if new_val ==0 : continue
            new_key = key.union( frozenset([(curr_obs,dist.parameters[0])]))
            new_tail[new_key] = new_val


    return new_tail


sampled_seq = []
all_steps = []
partial_sample,distrbutions = simulate_data()
print(partial_sample)
tails = prob_maping_from_seqs(partial_sample,distrbutions)
print("done")
# pprint.pprint(tails)
