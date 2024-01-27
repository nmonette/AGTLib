import gymnasium as gym
import numpy as np
import ray

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
            obs[i] = np.concatenate([np.array(j).flatten() for j in obs[i].values()])

        return obs, _

    def step(self, action: dict):
        obs, reward, done, trunc, _ = self.env.step(action)
        for i in obs:
            obs[i] = np.concatenate([np.array(j).flatten() for j in obs[i].values()])
        if isinstance(trunc, bool):
            trunc = {i: trunc for i in range(len(obs))}
        return obs, reward, done, trunc, _
    
    def render(self):
        self.env.render()

@ray.remote
class RayMultiGridWrapper(MultiGridWrapper):
    pass
    
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


