import gymnasium as gym

class SingleAgentEnvWrapper:
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