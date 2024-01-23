import gymnasium as gym
import numpy as np

class SingleAgentEnvWrapper(gym.Wrapper):
    """
    Wrapper in order to use Single-Agent environments in PPO.
    Mainly exists for test purposes. The rest of the functions
    are there to match the functions of a single agent gym 
    environment.
    """
    def __init__(self, env: gym.Env) -> None:
        """
        Parameters
        ----------
        env: gym.Env
            Simulation environment.
        """
        self.env = env

    def reset(self):
        obs, info = self.env.reset()
        return {0: obs}, info
    
    def step(self, action: dict):
        obs, reward, done, trunc, _ = self.env.step(action[0])
        return {0: obs}, {0: reward}, done, trunc, _
    
    def render(self):
        self.env.render()

class MultiGridWrapper(gym.Wrapper):
    """
    Wrapper in order to use Multigrid environments in PPO.
    Mainly there because of the odd observation format. 
    The rest of the functions are there to match the 
    functions of a standard multi-agent gym 
    environment.
    """

    def __init__(self, env: gym.Env) -> None:
        """
        Parameters
        ----------
        env: gym.Env
            Simulation environment.
        """
        self.env = env

    def reset(self, *args, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        for i in obs:
            obs[i] = np.concatenate((obs[i]["image"].flatten(), np.array(obs[i]["index"]).flatten() ))

        return obs, _

    def step(self, action: dict):
        obs, reward, done, trunc, _ = self.env.step(action)
        for i in obs:
            obs[i] = np.concatenate((obs[i]["image"].flatten(), np.array(obs[i]["index"]).flatten()))

        return obs, reward, done, trunc, _

    def render(self):
        self.env.render()