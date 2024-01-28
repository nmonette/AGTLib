from typing import Any

import gymnasium as gym
import numpy as np

from .grid import Grid


class TeamEmptyEnv(gym.Env):
    render: bool = False
    def __init__(self, dim, render_mode=None, max_episode_steps=12, num_agents_t1=2, num_agents_t2=1):
        self.dim = dim
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.num_agents_t1 = num_agents_t1
        self.num_agents_t2 = num_agents_t2

        self.observation_space = gym.spaces.MultiDiscrete(dim,dim, 2, dim,dim, 2, dim,dim, 2, dim, dim, 2, dim ,dim, 2)
        self.action_space = gym.spaces.Discrete(4)

    def step(self, actions) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray], dict]:
        """
        Returns obs, reward, done, trunc, info
        """
        rewards, trunc, done = self.grid.handle_actions(actions)
        obs = self.grid.get_state()
        return {i:obs for i in range(self.num_agents_t1+self.num_agents_t2)}, rewards, done, trunc, {}

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self.grid = Grid(self.dim, self.num_agents_t1, self.num_agents_t2)
        obs = self.grid.get_state()
        return {i:obs for i in range(self.num_agents_t1+self.num_agents_t2)}, {}

    def render(self):
        pass