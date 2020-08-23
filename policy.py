from env import L, BLOCKS, sample_program
import numpy as np
#from sklearn import linear_model
#from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

def encode_input(spec, blocks):
    block_enc = np.zeros((len(blocks), len(BLOCKS)))
    for i, bl in enumerate(blocks):
        block_enc[i][BLOCKS.index(bl)] = 1
    return np.concatenate((np.reshape(spec, (L*L,)),np.reshape(block_enc,(len(blocks)*len(BLOCKS),))))

def get_bigram(prog):
    return [(-1,prog[0])] + [(prog[i],prog[i+1]) for i in range(len(prog) - 1)]

def gen_data(spec, blocks, prog):
    enc_input = encode_input(spec, blocks)
    bigram = get_bigram(prog)
    
    ret = dict()
    for bg in bigram:
        u,v = bg
        if u not in ret:
            ret[u] = []
        ret[u].append((enc_input, v))        
    return ret

def gen_training(n):
    ret = dict()
    for i in range(n):
        blocks, x_poss, spec = sample_program(5)
        data = gen_data(spec, blocks, x_poss)
        for key in data:
            if key not in ret:
                ret[key] = []
            ret[key] += data[key]
    return ret

class BigramGenerator:
    def __init__(self):
        self.input_size = None
        # self.bigram_gens = dict([(i, linear_model.SGDClassifier(loss='log',max_iter=1000, tol=1e-3)) for i in range(-1,L)])
        # self.bigram_gens = dict([(i, tree.DecisionTreeClassifier()) for i in range(-1,L)])
        self.bigram_gens = dict([(i, RandomForestClassifier(n_estimators=100)) for i in range(-1,L)])

    def train(self, train_data):
        for key in train_data:
            bg_gen = self.bigram_gens[key]
            X = [x[0] for x in train_data[key]]
            Y = [x[1] for x in train_data[key]]
            bg_gen.fit(X,Y)

    def gen_bigram(self, spec, blocks):
        x = encode_input(spec, blocks)
        ps = dict([(i, self.bigram_gens[i].predict_proba([x])[0]) for i in self.bigram_gens])
        return ps

class BigramPolicy:
    def __init__(self, bigram):
        self.bigram = bigram
    def act(self, state):
        p = self.bigram[state]
        idxs = [_ for _ in range(L)]
        return np.random.choice(idxs, 1, p=p)[0]

def get_rollout(bg_p):
    ro = [-1]
    for i in range(5):
        nxt = bg_p.act(ro[-1])
        ro.append(nxt)
    return ro[1:]

if __name__ == '__main__':
    from env import sample_program, get_spec, drop_blocks, render_spec

    # training
    bigram_gen = BigramGenerator()
    train_data = gen_training(4000)
    bigram_gen.train(train_data)

    # testing
    blocks, x_poss, spec = sample_program(5)
    render_spec(spec, name='spec')

    bg = bigram_gen.gen_bigram(spec, blocks)
    print (bg)
    print ("ground truth :")
    print (x_poss)
    bg_p = BigramPolicy(bg)
    for i in range(1000):
        ro = get_rollout(bg_p)

        rec_spec = get_spec(drop_blocks(blocks, ro))
        gots = np.all(rec_spec == spec)
        if i < 10:
            render_spec(rec_spec, name=f'rec_{i}')
        if gots:
            print (i)
            render_spec(rec_spec, name=f'rec_work')
            assert 0, "we done"
