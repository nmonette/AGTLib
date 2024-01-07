# TO DO !
## GAE (agtlib.utils.rollout)
1. Transition From Dictionaries to a buffer for each agent
    - this requires us to duplicate the advantage list for each agent
## PPO (agt.cooperative.ppo)
0. Add optimizer so it works
1. Finish single agent PPO first 
2. Add child classes to PPO (MAPPO, IPPO)
3. Add optional orthogonal initialization for weights
## Trainer Classes
1. Add classes that will train two teams against each other, including running the rollout manager and then passing the buffers
2. Team-PSRO, FXP, Self-PLay
## General
1. Add multithreading capabilities for rollouts
## Other Algorithms
1. MA-POCA with (REINFORCE?)
    - we can also try doing it with a PPO update
2. HAPPO
3. Multi-Agent A3C (?)
    - implementing asynchronous gradient descent for multiple agents may prove to be difficult
Future Stuff (from the future folder)

Baselines Imports
1. monitor
2. utils
3. running_mean_std
4. atari_wrappers
