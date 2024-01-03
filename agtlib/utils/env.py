import gymnasium as gym

class SingleAgentEnvWrapper:
    def __init__(self, env: gym.Env):
        self.env = env

    def reset(self):
        obs, info = self.env.reset()
        return {0: obs}, info
    
    def step(self, action: dict):
        obs, reward, done, trunc, _ = self.env.step(action[0])
        return {0: obs}, {0: reward}, done, trunc, _
    
    def render(self):
        self.env.render()