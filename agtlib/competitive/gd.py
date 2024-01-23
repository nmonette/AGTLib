from typing import Union

import numpy as np
import torch 

from ..utils.exceptions import GradDisabledException, StrategyNotInSimplexException
from ..utils.projection import project_simplex, projection_simplex_sort, nate_projection, project_simplex_2

class VanillaGradientDescent:
    """
    Implementation of Projected Gradient Descent onto the Simplex. 
    For more details on its theory, view 
    `Theory.gd`. Algorithm pseudocode sourced from 
    https://panageas.github.io/agt23slides/L10%20Other%20equilibrium%20notions.pdf.
    """
    
    def __init__(self, n_actions: int, initial: torch.Tensor = None, stepsize = 0.01) -> None:
        """
        Parameters
        ----------
        n_actions: int
            The number of actions in the action space for the player performing Gradient Descent. 
        initial: torch.Tensor, optional
            The initial strategy for the player.
                Must be in the simplex, i.e. sum to 1 and be nonnegative. Defaults to a uniform distribution over the number of actions.
                Must have `requires_grad = True`, or else a `GradDisabledException` will be raised. 
        stepsize: int, float or other scalar representation, optional
            The stepsize for gradient descent. Defaults to `0.01`.

        """
        self.n_actions = n_actions
        
        if initial is None:
            self.current = torch.tensor([1/n_actions for _ in range(n_actions)], requires_grad=True)
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

        self.optimizer = torch.optim.Adam([self.current], stepsize)

        self.hist = []

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
        """
        loss = -utility 
        self.optimizer.zero_grad()
        loss.backward() # commented out because we are passing in gradient
        self.optimizer.step()
        with torch.no_grad():
            self.current = projection_simplex_sort(self.current)
        """
        
        loss = -utility
        loss.backward()

        x = self.current - self.stepsize * self.current.grad
        self.current.grad.zero_()

        with torch.no_grad():
            self.current = projection_simplex_sort(x)
            self.hist.append(x)
        
       #  self.current = torch.tensor(x, requires_grad=True)

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



