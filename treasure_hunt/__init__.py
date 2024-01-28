from .env import TeamEmptyEnv

CONFIGURATIONS = {
    'TreasureHunt-8x8-Team': (TeamEmptyEnv, {'dim': 8, "num_agents_t1":2, "num_agents_t2": 1, "max_episode_steps":60}),
    'TreasureHunt-6x6-Team': (TeamEmptyEnv, {'dim': 6, "num_agents_t1":2, "num_agents_t2": 1, "max_episode_steps":60}),
    'TreasureHunt-5x5-Team': (TeamEmptyEnv, {'dim': 5, "num_agents_t1":2, "num_agents_t2": 1, "max_episode_steps":50}),
    'TreasureHunt-4x4-Team': (TeamEmptyEnv, {'dim': 4, "num_agents_t1":2, "num_agents_t2": 1, "max_episode_steps":40}),
    'TreasureHunt-3x3-Team': (TeamEmptyEnv, {'dim': 3, "num_agents_t1":2, "num_agents_t2": 1, "max_episode_steps":12})
}

# Register environments with gymnasium
from gymnasium.envs.registration import register

for name, (env_cls, config) in CONFIGURATIONS.items():
    register(id=name, entry_point=env_cls, kwargs=config)
