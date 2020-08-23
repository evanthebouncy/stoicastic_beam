import numpy as np
import random
import collections
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

Block = collections.namedtuple('Block', 'w h')
PlacedBlock = collections.namedtuple('PlacedBlock', 'w h x y')

L = 8
BLOCKS = [(1,3),(3,1),(2,2),(2,3),(3,2),(3,3)]

def gen_block():
    return Block(*random.choice(BLOCKS))

def render_placed_blocks(placed_blocks, name='placed_blocks'):

    plt.figure()
    currentAxis = plt.gca()
    currentAxis.set_aspect('equal')

    for pb in placed_blocks:
        currentAxis.add_patch(Rectangle((pb.x/L, pb.y/L), pb.w/L, pb.h/L,facecolor="red",edgecolor='black'))
    
    plt.savefig(f'drawings/{name}.png')
    plt.close()

def render_spec(spec, name='spec'):
    plt.imshow(spec)
    plt.savefig(f'drawings/{name}.png')
    plt.close()

# take a group of blocks, and randomly drop them in a tetris fashoin
def drop_blocks(bs, x_poss):
    # get the countour for y axis
    def get_y_countour(placed_blocks):
        ret = dict([(i,0) for i in range(L)])
        for pb in placed_blocks:
            for x in range(pb.x, pb.x+pb.w):
                ret[x] = max(ret[x], pb.y+pb.h)
        return ret

    def drop(b, bx, dropped):
        bx_min, bx_max = bx, bx + b.w

        countour = get_y_countour(dropped)
        hh = max(countour[x] for x in range(bx_min, bx_max))
        return PlacedBlock(b.w, b.h, bx, hh)

    ret = []
    for b,x_pos in zip(bs, x_poss):
        try:
            placed_block = drop(b, x_pos, ret)
            ret.append(placed_block)
        except:
            return False
    return ret

def sample_program(n_blocks, return_block = False):
    blocks = [gen_block() for _ in range(n_blocks)]
    x_poss = [random.choice([x for x in range(L)]) for _ in range(n_blocks)]
    placed_blocks = drop_blocks(blocks, x_poss)
    if not placed_blocks:
        return sample_program(n_blocks, return_block)
    if return_block:
        return blocks, x_poss, placed_blocks
    return blocks, x_poss, get_spec(placed_blocks)

def get_spec(placed_blocks):
    ret = np.zeros((L,L))
    if placed_blocks == False:
        return ret
    for w,h,x,y in placed_blocks:
        ret[x:x+w,y:y+h] = 1
    return ret

if __name__ == '__main__':

    blocks, x_poss, placed_blocks = sample_program(5, return_block = True)
    print (placed_blocks)
    render_placed_blocks(placed_blocks, "pieces")
    spec = get_spec(placed_blocks)
    print (x_poss)
    render_spec(spec)
