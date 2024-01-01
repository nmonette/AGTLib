from typing import Union

import numpy as np
import torch 
from torch.nn import Softmax

from ..utils.exceptions import GradDisabledException, StrategyNotInSimplexException

class VanillaGradientDescent:
    """
    Implementation of Vanilla Gradient Descent. For more details on its theory, view 
    `Theory.gd`. Algorithm pseudocode sourced from 
    https://panageas.github.io/agt23slides/L10%20Other%20equilibrium%20notions.pdf.
    """
    
    def __init__(self, num_actions: int, initial: torch.Tensor = None, stepsize = 0.01) -> None:
        """
        Parameters
        ----------
        num_actions: int
            The number of actions in the action space for the player performing Gradient Descent. 
        initial: torch.Tensor, optional
            The initial strategy for the player.
                Must be in the simplex, i.e. sum to 1 and be nonnegative. Defaults to a uniform distribution over the number of actions.
                Must have `requires_grad = True`, or else a `GradDisabledException` will be raised. 
        stepsize: int, float or other scalar representation, optional
            The stepsize for gradient descent. Defaults to `0.01`.

        """
        self.num_actions = num_actions
        
        if initial is None:
            self.current = torch.tensor([1/num_actions for _ in range(num_actions)], requires_grad=True)
        elif not initial.requires_grad:
            raise GradDisabledException("initial")
        elif torch.sum(initial).item() != 1 or torch.any(torch.lt(initial, 0)).item():
            raise StrategyNotInSimplexException("initial")
        else:
            self.current = initial

        if not (stepsize > 0):
            raise ValueError("Parameter 'stepsize' is not nonnegative")
        else:
            self.stepsize = stepsize

    def step(self, utility: Union[int, float]) -> bool:
        """
        Updates `self.current` with the next gradient step. 

        Parameters
        ----------
        utility: int, float or other scalar representation
            The player's utility at the current time-step.

        Returns
        -------
        bool
            True if the algorithm has converged (up to a limited point of precision).
        """

        loss = torch.tensor(-utility, requires_grad = True, dtype=torch.float64) # -utility #
        loss.backward()
        next = (self.current.data - self.stepsize * loss.grad.data).softmax(-1)
        loss.grad.data.zero_()
        if torch.equal(next,self.current):
            return True
        else:
            self.current = next
            return False
        

    def get_action(self) -> int:
        """
        Samples an action from the current strategy and returns it as an integer index.

        Returns
        -------
        int
            The integer index of the action samples from the strategy.
        """
        dist = torch.distributions.Categorical(self.current)
        return dist.sample()
    
    def get_strategy(self) -> torch.Tensor:
        """
        Returns the current strategy as a vector.

        Returns
        -------
        torch.Tensor
            Probability vector containing the action-probabilities.
        """
        return self.current.data



