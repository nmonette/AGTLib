import gymnasium as gym
import multigrid.envs
from multigrid.envs.team_empty import TeamEmptyEnv
from gymnasium.envs.registration import register

CONFIGURATIONS = {
            'MultiGrid-Empty-6x6-Team': (TeamEmptyEnv, {'size': 8, "agents": 3, "allow_agent_overlap":True, "max_steps":3000}),
            'MultiGrid-Empty-4x4-Team': (TeamEmptyEnv, {'size': 6, "agents": 3, "allow_agent_overlap":True, "max_steps":3000}),
            'MultiGrid-Empty-3x3-Team': (TeamEmptyEnv, {'size': 5, "agents": 3, "allow_agent_overlap":True, "max_steps":3000})
        }
    
for name, (env_cls, config) in CONFIGURATIONS.items():
    register(id=name, entry_point=env_cls, kwargs=config)