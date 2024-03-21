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
4. Unzip "3x3-3-agents-table.npy.zip" into the root directory 
5. Run experiments:
```py
python -m main
```
# Using the CLI for Adversarial GDmax
There are many arguments, but here are the highlights:
- Adversarial Algorithm Selection
    1. `-a NREINORCE` uses deep REINFORCE
    2. `-a QREINFORCE` uses SARSA
- Hyperparameters
    1. `-i NUM_ITERATIONS`
    2. `-l NUM_EPISODES_PER_ITER`
    3. `-br NUM_UPDATES_PER_BEST_RESPONSE`
    4. `-lr LEARNING_RATE`
    5. `-g DISCOUNT FACTOR`
    6. `-na w1,w2,...` gives hidden network architecture
- Eval Mode:
    1. `-e` enables eval mode
    2. `-adv PATH` supplies path for adversarial policy
    3. `-team PATH` supplies path for team policy 
- Metrics:
    1. `-ds` disables checkpoint saving
    2. `-si NUM` sets interval for checkpoint saving
    3. `-ng` enables collection of Nash-Gap metric
    4. `-mi NUM` sets interval for collecting metrics
## An example experiment:
``` 
python -m main -a QREINFORCE -l 25000 -i 400 -ng
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
