import numpy as np
import torch
from agtlib.competitive.gd import VanillaGradientDescent 

if __name__ == "__main__":
    game = np.stack([
        np.array([0, 1, -1]), 
        np.array([1, 0, -1]), 
        np.array([-1, 1, 0]), 
    ])
    gd1 = VanillaGradientDescent(3, torch.tensor([1,0,0], requires_grad=True, dtype=torch.float64))
    gd2 = VanillaGradientDescent(3, torch.tensor([0,0,1], requires_grad=True, dtype=torch.float64))
    for i in range(100):
        util = game[gd1.get_action(), gd2.get_action()]

        gd1.step(util)
        gd2.step(-util)

        print("1: ", gd1.current)
        print("2: ", gd2.current)
        
# pdoc --docformat numpy agtlib
