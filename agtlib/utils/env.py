import gymnasium as gym
import numpy as np
# import ray


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

        return obs, _

    def step(self, action: dict):
        obs, reward, done, trunc, _ = self.env.step(action)
        # for i in obs:
        #     obs[i] = np.concatenate([np.array(j).flatten() for j in obs[i].values()])
        if isinstance(trunc, bool):
            trunc = {i: trunc for i in range(len(obs))}
        return obs, reward, done, trunc, _
    
    def render(self):
        return self.env.render()

class DecentralizedMGWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        """
        Parameters
        ----------
        env: gym.Env
            Simulation environment.
        """
        self.env = env
        dim = env.unwrapped.grid_size -2 

        # Set for adversary in stable_baselines GDmax
        self.observation_space = gym.spaces.Dict({0:gym.spaces.MultiDiscrete([dim, dim, 2, dim, dim, 2, dim, dim, 2, dim, dim, 2]), 1: gym.spaces.MultiDiscrete([dim, dim, dim, dim, 2, dim, dim, 2])})
        self.action_space = gym.spaces.Discrete(4)
    
    def reset(self, *args, **kwargs):
        obs, _ = self.env.reset()
        obs = obs[0]
        return {0: obs[[i for i in range(len(obs) - 9)] + [i for i in range(len(obs) - 6, len(obs))]], 1: obs[[i for i in range(len(obs) - 9, len(obs)) if i != len(obs) - 7]]}, _
    
    def step(self, action: dict):
        obs, reward, done, trunc, _ = self.env.step(action)
        obs = obs[0]
        obs = {0: obs[[i for i in range(len(obs) - 9)] + [i for i in range(len(obs) - 6, len(obs))]], 1: obs[[i for i in range(len(obs) - 9, len(obs)) if i != len(obs) - 7]]}
    
        if isinstance(trunc, bool):
            trunc = {i: trunc for i in range(len(obs))}
        
        return obs, reward, done, trunc, _
    
    def render(self):
        return self.env.render()
    
class IndepdendentTeamWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        """
        Parameters
        ----------
        env: gym.Env
            Simulation environment.
        """
        self.env = env
        dim = env.unwrapped.grid_size -2 

        # Set for adversary in stable_baselines GDmax
        self.observation_space = gym.spaces.Dict({0:gym.spaces.MultiDiscrete([dim, dim, 2, dim, dim, 2, dim, dim, 2]), 1: gym.spaces.MultiDiscrete([dim, dim, dim, dim, 2, dim, dim, 2]), 2:gym.spaces.MultiDiscrete([dim, dim, 2, dim, dim, 2, dim, dim, 2])})
        self.action_space = gym.spaces.Discrete(4)
    
    def reset(self, *args, **kwargs):
        obs, _ = self.env.reset()
        obs = obs[0]
        new_obs =  {i: np.concatenate((obs[i * 3: i*3 + 3], obs[len(obs) - 6: len(obs)])) for i in range(len(self.observation_space) - 1)}
        new_obs[len(self.observation_space) - 1] =  obs[[i for i in range(len(obs) - 9, len(obs)) if i != len(obs) - 7]]
        return new_obs, _
    
    def step(self, action: dict):
        obs, reward, done, trunc, _ = self.env.step(action)
        obs = obs[0]
        new_obs =  {i: np.concatenate((obs[i * 3: i*3 + 3], obs[len(obs) - 6: len(obs)])) for i in range(len(self.observation_space) - 1)}
        new_obs[len(self.observation_space) - 1] =  obs[[i for i in range(len(obs) - 9, len(obs)) if i != len(obs) - 7]]
    
        if isinstance(trunc, bool):
            trunc = {i: trunc for i in range(len(obs))}
        
        return new_obs, reward, done, trunc, _
    
    def render(self):
        return self.env.render()
    
def action_to_index(action, n_agents):
    """
    Action is a numpy vector (a1, a2,..., a_n)
    where a_i is in 1,2,3,4

    assumes number of actions is 4 here.
    """
    weights = np.concatenate(np.ones((1,)) + 4 * np.ones((n_agents-1, )))
    return np.sum(np.dot(action, np.cumprod(weights)))

def state_action_to_index(self, state, action, dim, n_agents):
    idx = np.cumsum(np.ones((dim, dim))).reshape((dim, dim))

    values = np.concatenate((np.array([
        idx[state["goal1_x"] - 1, state["goal1_y"] - 1],
        state["goal1_terminated"],
        idx[state["goal2_x"] - 1, state["goal2_y"] - 1],
        state["goal2_terminated"],
    ]), np.concatenate([[idx[state[f"{i}_x"] - 1, state[f"{i}_y"] - 1], state[f"{i}_terminated"]]  for i in range(n_agents)]),  np.array(action)))

    weights = np.concatenate((np.array([
        1, 
        dim * dim, 
        2,
        dim * dim - 1,
        2
    ]), np.concatenate([[dim * dim, 2] for i in range(n_agents)]), 4 * np.ones(n_agents - 1)))

    return np.sum(np.dot(values, np.cumprod(weights)))

def calculate_reward(state, action, n_agents, dim):
    """
    finds reward with reference to the team, not the adversary

    Guide for action numbering:
    ```
    class BallAction(enum.IntEnum):
        left = 0
        right = enum.auto()
        up = enum.auto()
        down = enum.auto()
    ```
    """
    action_map = {
        0: lambda x,y: (x-1,y),
        1: lambda x,y: (x+1,y),
        2: lambda x,y: (x,y+1),
        3: lambda x,y: (x,y-1)
    }
    total_reward = 0
    for agent in range(n_agents):
        if state[f"{agent}_terminated"]:
            continue

        if not state["goal1_terminated"] and action_map[action[agent]](state[f"{agent}_x"], state[f"{agent}_y"]) == (state["goal1_x"], state["goal1_y"]):
            total_reward += 1 if agent < n_agents - (n_agents // 2)  else -1

        if not state["goal2_terminated"] and action_map[action[agent]](state[f"{agent}_x"], state[f"{agent}_y"]) == (state["goal2_x"], state["goal2_y"]):
            total_reward += 1 if agent < n_agents - (n_agents // 2)  else -1

    return total_reward

def generate_reward(dim, n_agents):
    """
    This is a table with the state-action reward
    for every state and action in the game (model-based)
    n_agents is amount of agents in total, not the team

    TODO: need to reflect the state-action-reward table
    as a sparse matrix, otherwise computation will take 
    WAY too long.
    - maybe we should just do a dictionary with 
        state-action_id: 1
        and then the "in" operation is fast, so we can just assume
        that it is 0 otherwise
    """
    def where(table, value):
        return [int(i[0] + 1) for i in np.where(table == value)]
    
    idx = [(i,j) for i in range(dim) for j in range(dim)]
    table = np.zeros([dim, dim, 2] * (n_agents + 2) + [4] * n_agents) 
    state = {}
    for n1 in range(dim * dim):
        state[f"{0}_x"], state[f"{0}_y"] = idx[n1]
        for o_n1 in range(2): # hard coding for 3 agents
            state[f"{0}_terminated"] = o_n1
            for n2 in range(dim * dim):
                state[f"{1}_x"], state[f"{1}_y"] = idx[n2]
                for o_n2 in range(2):
                    state[f"{1}_terminated"] = o_n2
                    for n3 in range(dim * dim): 
                        state[f"{2}_x"], state[f"{2}_y"] = idx[n3]
                        for o_n3 in range(2):
                            state[f"{2}_terminated"] = o_n3
                            for i in range(dim * dim):
                                state["goal1_x"], state["goal1_y"] = idx[i]
                                for k in range(2):
                                    state["goal1_terminated"] = k
                                    for j in range(dim * dim):
                                        if j == i:
                                            continue
                                        state["goal2_x"], state["goal2_y"] = idx[j]
                                        for l in range(2):
                                            state["goal2_terminated"] = l
                                            for x1 in range(4):
                                                for x2 in range(4):
                                                    for x3 in range(4):
                                                        if (x:= calculate_reward(state, (x1,x2,x3), 3, dim)) != 0:
                                                            # table[self.state_action_to_index(state, (x1,x2,x3), dim, n_agents)] = 1
                                                            table[
                                                                    idx[n1][0],idx[n1][1],
                                                                    o_n1,
                                                                    idx[n2][0],idx[n2][1],
                                                                    o_n2,
                                                                    idx[n3][0],idx[n3][1],
                                                                    o_n3,
                                                                    idx[i][0],idx[i][1],
                                                                    k,
                                                                    idx[j][0],idx[j][1],
                                                                    l,
                                                                    x1, x2, x3
                                                            ] = x
                                                            
    with open(f"{dim}x{dim}-{n_agents}-agents-table.npy", 'wb') as f:
        np.save(f, table)
    return table

class PettingZooWrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env

    def step(self, action):
        new = {}
        for i in action:
            if i == len(action) - 1:
                new["adversary_0"] = action[i]
            else:
                new[f"agent_{i}"] = action[i]

        observations, rewards, terminations, truncations, infos =  self.env.step(new)
        obs = {}
        rew = {}
        done = {}
        trunc = {}
        
        count = 0
        for i in observations:
            if i == "adversary_0":
                obs[len(observations) - 1] = observations["adversary_0"]
                rew[len(observations) - 1] = rewards["adversary_0"]
                done[len(observations) - 1] = terminations["adversary_0"]
                trunc[len(observations) - 1] = truncations["adversary_0"]
            else:
                obs[count] = observations[i]
                rew[count] = rewards[i]
                done[count] = terminations[i]
                trunc[count] = truncations[i]
                count += 1

        return obs, rew, done, trunc, infos

    
    def reset(self, *args, **kwargs):
        obs, infos =  self.env.reset(*args, **kwargs)
        
        new = {}
        count = 0
        for i in obs:
            if i == "adversary_0":
                new[len(obs) - 1] = obs["adversary_0"]
            else:
                new[count] = obs[i]
                count += 1

        return new, infos 
    
    def render(self):
        return self.env.render()