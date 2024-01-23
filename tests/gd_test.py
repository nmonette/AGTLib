import numpy as np
import torch
from agtlib.competitive.gd import VanillaGradientDescent 

def test_gd():
    torch.autograd.set_detect_anomaly(True)    
    game = np.stack([
        np.array([0, -1, 1]), 
        np.array([1, 0, -1]), 
        np.array([-1, 1, 0]), 
    ])
    game = torch.tensor(game.astype(np.double))
    gd1 = VanillaGradientDescent(3, torch.tensor([1,0,0], requires_grad=True, dtype=torch.float64))
    gd2 = VanillaGradientDescent(3, torch.tensor([0,0,1], requires_grad=True, dtype=torch.float64))
    for i in range(1000):
        util1 = gd1.current.T @ game.detach() @ gd2.current.detach() 
        util2 = gd2.current.T @ (-game).T.detach() @ gd1.current.detach()

        # grad1 = game.detach() @ gd2.current.detach()
        # grad2 = (-game.detach()).T @ gd1.current.detach()
        # print("grad1: ", grad1)
        # print("grad2: ", grad2)

        gd1.step(util1) # util1
        gd2.step(util2) # util2 

        print("1: ", gd1.current)
        print("2: ", gd2.current)
    
    print("1: ", torch.mean(torch.stack(gd1.hist), dim=0, keepdim=True))
    print("2: ", torch.mean(torch.stack(gd2.hist), dim=0, keepdim=True))
        
# pdoc --docformat numpy agtlib
