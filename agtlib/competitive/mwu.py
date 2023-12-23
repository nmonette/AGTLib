import numpy as np
import torch 

from ..utils.exceptions import GradDisabledException, StrategyNotInSimplexException

class MultiplicativeWeights:
    """
    Implementation of Multiplicative Weights Update. For more details on its theory, view 
    `Theory.mwu`
    """

    def __init__(self, num_actions: int, stepsize = 0.01) -> None:
        """
        Parameters
        ----------
        num_actions: int
            The number of actions in the action space for the player performing Gradient Descent. 
        stepsize: int, float or other scalar representation
            The stepsize for gradient descent. Defaults to 0.01.

        """
        if not isinstance(num_actions, int) and num_actions > 0:
            raise ValueError("Parameter 'num_actions' is not an integer greater than 0")
        else:
            self.num_actions = num_actions
    
        self.current = torch.tensor([1/num_actions for _ in range(num_actions)], requires_grad=True)

        if not (np.isscalar(stepsize) and stepsize > 0):
            raise ValueError("Parameter 'stepsize' is not a scalar greater than 0")
        else:
            self.stepsize = stepsize

    def step(self, o_action: int | float) -> bool:
        """
        Updates `self.current` with the next gradient step. 

        Parameters
        ----------
        utility: int, float or other scalar representation.
            The player's utility at the current time-step.
        
        Returns
        -------
        bool
            True if the algorithm has converged.
        """

        if not np.isscalar(utility):
            raise ValueError("Parameter 'utility' is not a scalar")
        
        reg_term = sum()
        for i in range(len(self.current)):
            self.current[i] 
    
