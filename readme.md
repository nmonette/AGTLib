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
4. Install stable_baselines3 modifications
```py
cd stable-baselines3
pip install -e .
```
5. Run experiments:
```py
python -m main
```
# Using the CLI for Adversarial GDmax
- Algorithm Selection
    - Deep REINFORCE for the team:
        1. `-a NREINORCE` uses deep REINFORCE for the adversary
        2. `-a QREINFORCE` uses SARSA for the adversary
        3. `-a PREINFORCE` uses PPO for the adversary
    - Softmax REINFORCE for the team:
        1. `-a TQREINFORCE` uses SARSA for the adversary
- Hyperparameters
    1. `-i NUM_ITERATIONS`
    2. `-l NUM_EPISODES_PER_ITER`
    3. `-br NUM_UPDATES_PER_BEST_RESPONSE`
    4. `-lr LEARNING_RATE`
    5. `-g DISCOUNT FACTOR`
    6. `-na w1,w2,...` gives hidden network architecture
- Eval Mode:
    1. `-e` enables eval mode
    2. `-de` disables post-training eval mode
    3. `-adv PATH` supplies path for adversarial policy
    4. `-team PATH` supplies path for team policy 
- Metrics/Checkpoints:
    1. `-ds` disables checkpoint saving and eval video saving
    2. `-si NUM` sets interval for checkpoint saving
    3. `-ng` enables collection of Nash-Gap metric
    4. `-mi NUM` sets interval for collecting metrics
- Environment
    1. `-dim` sets grid dimension
    2. `-f` fixes grid to single configuration 

## An example experiment:
``` 
python -m main -a QREINFORCE -l 25000 -i 400 -ng
```

# TO DO !
# Flight goals:
1. Write docstrings
2. Fully implement PPO variants
    - Write GDmax Experiment Using PPO
3. Add PPO experiment to argparser
4. Add customizability so that it works with multiple environments (i.e. action spaces etc. )
5. Fix multigrid so that it doesn't require us to disable the env checker
6. Write multigrid in pytorch
7. Add Direct Parameterization for the team versus Q as the adversary.

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
