#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Time        : 2017/11/16 下午10:57
# @Author      : Zoe
# @File        : hmm_test.py
# @Description :

import numpy as np
from hmmlearn import hmm

states = ["box 1", "box 2", "box3"]
n_states = len(states)

observations = ["red", "white"]
n_observations = len(observations)

start_probability = np.array([0.2, 0.4, 0.4])

transition_probability = np.array([
  [0.5, 0.2, 0.3],
  [0.3, 0.5, 0.2],
  [0.2, 0.3, 0.5]
])

emission_probability = np.array([
  [0.5, 0.5],
  [0.4, 0.6],
  [0.7, 0.3]
])

model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_=start_probability
model.transmat_=transition_probability
model.emissionprob_=emission_probability

seen = np.array([[0,1,0]]).T
logprob, box = model.decode(seen, algorithm="viterbi")
seen = [[0],[1],[0]]
print(map(lambda x: observations[x], seen))
# print("The ball picked:", ", ".join(map(lambda x: observations[x], seen)))
# print("The hidden box", ", ".join(list(map(lambda x: states[x], box))))

print(model.score(seen))