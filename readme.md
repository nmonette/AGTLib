# Installation Instructions
1. I suggest you create a venv in the directory where you clone this:
```py
python -m venv "VENV NAME"
```
2. Install multigrid environment
```py
cd multigrid
pip install -e .
```
3. Install AGTLib Dependencies (from root directory)
```py
cd ..
pip install -r requirements.txt
```
4. Run experiments:
```py
python -m main
```

# TO DO !

## PPO (agt.cooperative.ppo)
2. Add child classes to PPO (MAPPO, IPPO)
3. Add optional orthogonal initialization for weights
## Trainer Classes
1. Add classes that will train two teams against each other, including running the rollout manager and then passing the buffers
2. Team-PSRO, FXP, Self-PLay
## Other Algorithms
1. MA-POCA with (REINFORCE?)
    - we can also try doing it with a PPO update
2. HAPPO
3. Multi-Agent A3C (?)
    - implementing asynchronous gradient descent for multiple agents may prove to be difficult
Future Stuff (from the future folder)
