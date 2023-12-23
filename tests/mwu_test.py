import numpy as np
import torch
from agtlib.competitive.mwu import MultiplicativeWeights 

if __name__ == "__main__":
    game = np.stack([
        np.array([0, 1, -1]), 
        np.array([1, 0, -1]), 
        np.array([-1, 1, 0]), 
    ])
    mw1 = MultiplicativeWeights(game, 0.5)
    mw2 = MultiplicativeWeights(-game, 0.5)

    for i in range(10000):
        move1 = mw1.get_action().item()
        move2 = mw2.get_action().item()

        mw1.step(move2)
        mw2.step(move1)

        print("1: ", mw1.get_strategy())
        print("2: ", mw2.get_strategy())
        
# pdoc --docformat numpy agtlib
