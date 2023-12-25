# Some Notes on Style:
### Documentation
Please note that our documentation is being processed by pdoc (https://pdoc.dev/docs/pdoc.html). The way this works is that it turns docstrings written in certain places and turns it into documentation, along with the code itself. To check the documentation, run this command from the root of the directory: 
```cmd 
pdoc --math --docformat numpy agtlib
```
### Docstrings
When writing these, try to write them in a similar style to the ones in the examples / the ones already written. 
1. Beginning of files: if the file has an implementation of more than one thing, write a brief summary of the things being implemented in the file
    - place this before anything else in the file, including imports

example from `agtlib.cooperative.base`:
```py
"""
## Base Classes for Cooperative RL Algorithms
Some utility objects that will be used, as well as 
base classes to inherit from.
"""
```

2. Classes: Before the `__init__` function, write a docstring saying what the class is impplementing. If it is a novel concept (i.e. a concept that isn't super widely known), give the link where we got the algorithm pseudocode/concept from, and link 
to the associated `Theory` file if there is one

example from `agtlib.cooperative.base`
```py
class VanillaGradientDescent:
    """
    Implementation of Vanilla Gradient Descent. For more details on its theory, view 
    `Theory.gd`. Algorithm pseudocode sourced from 
    https://panageas.github.io/agt23slides/L10%20Other%20equilibrium%20notions.pdf.
    """
    
    def __init__(...):
        ...
```


3. Functions: write a description of what the function does in the numpy docstring style. 
- Do this for every function, even private functions and `__init__`.
- Include details about parameters and return values in the format from the example
- Specify default values in the description of each parameter, as well as if it is keyword only.
example from `agtlib.cooperative.base`
```
def forward(self, x: torch.Tensor) -> torch.Tensor:
"""
Performs the forward pass of the neural network.
Parameters
----------
x: torch.Tensor
    The flattened observation of the agent(s).

Returns
-------
torch.Tensor
    Probability vector containing the action-probabilities.
"""
```

## Other Notes
1. Try to make each function have the least amount of required parameters as possible, and make the rest keyword-only. 
2. Try to utilize object-oriented programming a decent amount, so that new implementations could be built on old ones. This might mean adding new files to a `base.py` file, or in the `agtlib.utils` directory.
3. As far as the implementation of RL/online learning models (e.g. ppo) goes, try to include the following functions for ease of use
- step: performings a training update
- get_action: samples an action from the policy given the current state
- get strategy: (only works if there is a finite action space) Take the 




