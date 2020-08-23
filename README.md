# Simple Implementation for Stochastic Beam Search for Synthesis

a simple implementation without any neural networks on a simple domain.

paper : https://arxiv.org/pdf/1903.06059.pdf

this repo is a working implementation of Algorithm 1. in this paper, without any NN fluffs.

requirements : numpy, sklearn, matplotlib

## env.py

contains a simple tetris environment. given a final silouett of tetris shapes and a list of tetris to place(no rotation) synthesize a sequence of x coordinates to drop the list of tetris so that it makes the silowett (sp? ah forget it english amirite)

task :

input : a silowett, a list of 5 tetris i.e. [block1, block2, ... block5]
output : a sequence of drops specified as x coordinates, i.e. [0,3,2,7,1]

run python env.py to generate some drawings that can clear things up

## policy.py

contains a simple decisionforest based bigram generator that takes in the silowett spec, and a list of tetris, create a bi-gram (9x8, include start symbol) that can be used to sample the sequence of x coordinates

run python policy.py to generate some drawings, notably, spec is the spec you're trying to hit, and you'll see some candidates being sampled, some will hit the spec, some wont. keep running python policy.py until something good happens (i.e. spec is satisfied)

## inference.py

the meat of the paper. given a spec and a list of tetris, we first use the bigram generator to give a bigram

then, conditioned on the same bigram, the same number of budgets, 3 search algorithms : random, beam, stokastik_beam try to find correct program within budget

run python inference.py to see numbers popping off
