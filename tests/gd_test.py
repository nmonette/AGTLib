import numpy as np
import torch
from agtlib.competitive.gd import VanillaGradientDescent 

def test_gd():
    game = np.stack([
        np.array([0, -1, 1]), 
        np.array([1, 0, -1]), 
        np.array([-1, 1, 0]), 
    ])
    game = torch.tensor(game).double()
    gd1 = VanillaGradientDescent(3, torch.tensor([1,0,0], requires_grad=True, dtype=torch.float64))
    gd2 = VanillaGradientDescent(3, torch.tensor([0,0,1], requires_grad=True, dtype=torch.float64))
    for i in range(10000):
        util = game[gd1.get_action(), gd2.get_action()] # gd1.current.T @ game @ gd2.current # can also try gd1.current.T @ game and y.T @ (-game).T
        
        gd1.step(util)
        gd2.step(-util)

        print("1: ", gd1.current)
        print("2: ", gd2.current)
        
# pdoc --docformat numpy agtlib
