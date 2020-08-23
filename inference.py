from env import *
from policy import *
import numpy as np
from numpy.random import gumbel
# different ways to sample programs

np.seterr(divide='ignore', invalid='ignore')

def random_rollout(bg_p):
    ro = [-1]
    for i in range(5):
        nxt = bg_p.act(ro[-1])
        ro.append(nxt)
    return ro[1:]

def random_rollout_k(spec, blocks, bigram_gen, k):
    bg = bigram_gen.gen_bigram(spec, blocks)
    bg_p = BigramPolicy(bg)
    for i in range(k):
        ro = get_rollout(bg_p)
        rec_spec = get_spec(drop_blocks(blocks, ro))
        gots = np.all(rec_spec == spec)
        if gots:
            return True
    return False

def beam_k(spec, blocks, bigram_gen, k):
    bg = bigram_gen.gen_bigram(spec, blocks)
    bg_p = BigramPolicy(bg)
    fringe = [(0, (-1,))]
    for i in range(5):
        new_fringe = []
        for neglogp, prefix in fringe:
            nxt_probs = bg_p.bigram[prefix[-1]]
            for next_action, next_prob in enumerate(nxt_probs):
                new_item = (neglogp - np.log(next_prob), prefix + (next_action,))
                new_fringe.append(new_item)
        fringe = sorted(new_fringe)[:k]
    
    for x in fringe:
        seq = x[1][1:]
        rec_spec = get_spec(drop_blocks(blocks, seq))
        gots = np.all(rec_spec == spec)
        if gots:
            return True
    return False        

def stochastic_beam_k(spec, blocks, bigram_gen, k):
    bg = bigram_gen.gen_bigram(spec, blocks)
    bg_p = BigramPolicy(bg)   
    # each element in the fringe is a triple
    # g_phi_s = the gumbled logpr of partial sequence
    # phi_s = the logpr of partial seq
    # s = the partial seq 
    fringe = [(0, 0, (-1,))]
    for i in range(5):
        new_fringe = []
        for g_phi_s, phi_s, s in fringe:
            z = float('-inf')
            nxt_probs = bg_p.bigram[s[-1]]
            # the next sequence
            ss = []
            # the next log_prob
            phi_ss = []
            # the gumbeled next log_prob
            g_phi_ss = []
            # the hat gumbeled next log_prob
            ghat_phi_ss = []
            for next_action, next_prob in enumerate(nxt_probs):
                ss.append(s + (next_action,))
                item_phi_ss = phi_s + np.log(next_prob)
                phi_ss.append(item_phi_ss)
                item_g_phi_ss = gumbel(item_phi_ss)
                z = max(z, item_g_phi_ss)
                g_phi_ss.append(item_g_phi_ss)

            for i in range(len(nxt_probs)):
                item_ghat_phi_ss = - np.log(np.exp(-g_phi_s) - np.exp(-z) + np.exp(-g_phi_ss[i]))
                ghat_phi_ss.append(item_ghat_phi_ss)

            for xx in zip(ghat_phi_ss, phi_ss, ss):
                new_fringe.append(xx)

            fringe = list(reversed(sorted(new_fringe)))[:k]

    for x in fringe:
        seq = x[2][1:]
        rec_spec = get_spec(drop_blocks(blocks, seq))
        gots = np.all(rec_spec == spec)
        if gots:
            return True
    return False   

if __name__ == '__main__':
    # training
    bigram_gen = BigramGenerator()
    train_data = gen_training(10000)
    bigram_gen.train(train_data)

    M = 100
    rand, beam, gumb = 0,0,0
    for i in range(1, 10000):
        blocks, x_poss, spec = sample_program(5)
        ro_solv = 1 if random_rollout_k(spec, blocks, bigram_gen, M) else 0
        bm_solv = 1 if beam_k(spec, blocks, bigram_gen, M) else 0
        sb_solv = 1 if stochastic_beam_k(spec, blocks, bigram_gen, M) else 0

        rand += ro_solv
        beam += bm_solv
        gumb += sb_solv 
        print (rand, beam, gumb, "of ", i)