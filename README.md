# HMMOP
HMMOP model : Learning Hidden Markov Models When the Locations of Missing Observations are Unknown.

HMMs have the ability to deal with missing data. However, standard HMM learning algorithms rely crucially on the assumption that the positions of the missing observations within the observation sequence are known. In some situations where such assumptions are not feasible, a number of special algorithms have been developed. Currently, these algorithms rely strongly on specific structural assumptions of the underlying chain, such as acyclicity, and are not applicable in the general case. Here we consider a general model for learning HMMs from data with unknown missing observation locations (i.e., only the order of the non-missing observations are known). We introduce a generative model of the location omissions and propose two learning methods based on a Gibbs sampler.

This project contains all one need to use the HMMOP model and all the code for the validations.
https://arxiv.org/abs/2203.06527
