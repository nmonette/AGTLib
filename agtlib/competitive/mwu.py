import numpy as np
import torch 

from ..utils.exceptions import GradDisabledException, StrategyNotInSimplexException

class MultiplicativeWeights:
    """
    Implementation of Multiplicative Weights Update. For more details on its theory, view 
    `Theory.mwu`. Algorithm pseudocode sourced from 
    https://www.cs.princeton.edu/~arora/pubs/MWsurvey.pdf.
    """

    def __init__(self, game: np.ndarray, stepsize = 0.01) -> None:
        """
        Parameters
        ----------
        game: numpy.ndarray
            Payoff matrix for the player performing Multiplicative Weights Update. 
            Assumes active player is the row player. 
        stepsize: int, float or other scalar representation, optional
            The stepsize for gradient descent. Defaults to `0.01`. Must be in the range 
            `(0, 0.5]`.
        """
        self.game = game
    
        self.current = torch.ones(self.game.shape[0])

        if not (0 < stepsize <= 0.5):
            raise ValueError("Parameter 'stepsize' is not in the range (0,0.5]")
        else:
            self.stepsize = stepsize

    def step(self, c_action: int) -> bool:
        """
        Updates `self.current` with the next set of weights. 

        Parameters
        ----------
        c_action: int
            The opponent's action in the form of integer index. Referred to as the choice of the adversary in the literature. 
        
        Returns
        -------
        bool
            True if the algorithm has converged (up to a limited point of precision).
        """

        if not 0 <= c_action <= self.game.shape[0]:
            raise ValueError("Parameter 'c_action' is not in the range [0, number of actions)")
        
        prev = self.current.clone()

        for i in range(len(self.current)):
            if self.game[i, c_action] >= 0:
                self.current[i] *= (1-self.stepsize)**(self.game[i, c_action] / self.game.shape[0])
            else:
                self.current[i] *= (1+self.stepsize)**(-self.game[i, c_action] / self.game.shape[0])

        return torch.equal(prev, self.current)

    def get_action(self) -> int:
        """
        Samples an action from the current strategy and returns it as an integer index.

        Returns
        -------
        int
            The integer index of the action samples from the strategy.
        """
        dist = torch.distributions.Categorical(self.get_strategy())
        return dist.sample()
    
    def get_strategy(self) -> torch.Tensor:
        """
        Returns the current strategy as a vector.

        Returns
        -------
        torch.Tensor
            Probability vector containing the action-probabilities.
        """
        reg_term = torch.sum(self.current).item()
        return torch.tensor([self.current[i] / reg_term for i in range(len(self.game))])
    
