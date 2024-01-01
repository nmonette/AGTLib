"""
A set of exceptions used throughout the library. 
"""

class GradDisabledException(ValueError):
    """
    Raised when a strategy does not have gradient descent enabled.
    """
    def __init__(self, var_name):
        super().__init__(self, f"The parameter '{var_name}' does not have 'requires_grad' set to True")

class StrategyNotInSimplexException(ValueError):
    """
    Raised when a strategy is not in the simplex, namely, 
    it has negative values and/or does not sum to 1.
    """
    def __init__(self, var_name):
        super().__init__(f"The parameter '{var_name}' is not in the simplex. Please input a vector that is nonnegative and sums to one")
