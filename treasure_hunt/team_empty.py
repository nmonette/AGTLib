from __future__ import annotations
from random import randint, shuffle
import random

# from multigrid.base import TeamMultiGridEnv
# from multigrid.core import Grid, Agent, BallAgent
# from multigrid.core.constants import Direction
# from multigrid.core.world_object import Goal
from ..core.grid import Grid
from multigrid.utils.obs import gen_obs_grid_encoding


class TeamEmptyEnv():
    """
    .. image:: https://i.imgur.com/wY0tT7R.gif
        :width: 200

    ***********
    Description
    ***********

    This environment is an empty room, and the goal for each agent is to reach the
    green goal square, which provides a sparse reward. A small penalty is subtracted
    for the number of steps to reach the goal.

    The standard setting is competitive, where agents race to the goal, and
    only the winner receives a reward.

    This environment is useful with small rooms, to validate that your RL algorithm
    works correctly, and with large rooms to experiment with sparse rewards and
    exploration. The random variants of the environment have the agents starting
    at a random position for each episode, while the regular variants have the
    agent always starting in the corner opposite to the goal.

    *************
    Mission Space
    *************

    "get to the green goal square"

    *****************
    Observation Space
    *****************

    The multi-agent observation space is a Dict mapping from agent index to
    corresponding agent observation space.

    Each agent observation is a dictionary with the following entries:

    * image : ndarray[int] of shape (view_size, view_size, :attr:`.WorldObj.dim`)
        Encoding of the agent's partially observable view of the environment,
        where the object at each grid cell is encoded as a vector:
        (:class:`.Type`, :class:`.Color`, :class:`.State`)
    * direction : int
        Agent's direction (0: right, 1: down, 2: left, 3: up)
    * mission : Mission
        Task string corresponding to the current environment configuration

    ************
    Action Space
    ************

    The multi-agent action space is a Dict mapping from agent index to
    corresponding agent action space.

    Agent actions are discrete integer values, given by:

    +-----+--------------+-----------------------------+
    | Num | Name         | Action                      |
    +=====+==============+=============================+
    | 0   | left         | Turn left                   |
    +-----+--------------+-----------------------------+
    | 1   | right        | Turn right                  |
    +-----+--------------+-----------------------------+
    | 2   | forward      | Move forward                |
    +-----+--------------+-----------------------------+
    | 3   | pickup       | Pick up an object           |
    +-----+--------------+-----------------------------+
    | 4   | drop         | Drop an object              |
    +-----+--------------+-----------------------------+
    | 5   | toggle       | Toggle / activate an object |
    +-----+--------------+-----------------------------+
    | 6   | done         | Done completing task        |
    +-----+--------------+-----------------------------+

    *******
    Rewards
    *******

    A reward of ``1 - 0.9 * (step_count / max_steps)`` is given for success,
    and ``0`` for failure.

    ***********
    Termination
    ***********

    The episode ends if any one of the following conditions is met:

    * Any agent reaches the goal
    * Timeout (see ``max_steps``)

    *************************
    Registered Configurations
    *************************

    * ``MultiGrid-Empty-5x5-v0``
    * ``MultiGrid-Empty-Random-5x5-v0``
    * ``MultiGrid-Empty-6x6-v0``
    * ``MultiGrid-Empty-Random-6x6-v0``
    * ``MultiGrid-Empty-8x8-v0``
    * ``MultiGrid-Empty-16x16-v0``
    """

    def __init__(
        self,
        size: int = 8,
        num_team1: int = 2,
        num_team2: int = 1,
    ):
        # max_steps: int | None = None,
        # joint_reward: bool = False,
        # success_termination_mode: str = 'all',
        # **kwargs):
        """
        Parameters
        ----------
        size : int, default=8
            Width and height of the grid
        agent_start_pos : tuple[int, int], default=(1, 1)
            Starting position of the agents (random if None)
        agent_start_dir : Direction, default=Direction.right
            Starting direction of the agents (random if None)
        max_steps : int, optional
            Maximum number of steps per episode
        joint_reward : bool, default=True
            Whether all agents receive the reward when the task is completed
        success_termination_mode : 'any' or 'all', default='any'
            Whether to terminate the environment when any agent reaches the goal
            or after all agents reach the goal
        **kwargs
            See :attr:`multigrid.base.MultiGridEnv.__init__`
        """
        # agents = [BallAgent(index=i, view_size=7, see_through_walls=True) for i in range(agents)]
        # self.agent_start_pos = agent_start_pos
        # self.agent_start_dir = agent_start_dir
        # Generating all coordinates of the grid
        self.COORD_PAIRS = [(i,j) for i in range(1,size-1) for j in range(1,size-1)]
        self.num_team1 = num_team1
        self.num_team2 = num_team2
        self.size = size

        # super().__init__(
        #     mission_space="get to the closest green square without an agent on it",
        #     grid_size=size,
        #     max_steps=max_steps or (4 * size**2),
        #     joint_reward=joint_reward,
        #     success_termination_mode=success_termination_mode,
        #     agents = agents,
        #     **kwargs,
        # )

    def on_success(
        self,
        agent,
        rewards,
        terminations):
        super().on_success(agent, rewards, terminations)

        if self.goal1 == agent.state.pos:
            self.goal1_terminated = True
        
        if self.goal2 == agent.state.pos:
            self.goal2_terminated = True

    def gen_obs(self, step=False):
        if step:
            ## Amount of possibilities in NxN is (N^2 - 2)^num_agents * 2^(num_agents + 2)
            obs = {}
            for i in range(self.num_agents):
                obs[f"{i}_x"] = self.agents[i].pos[0] - 1
                obs[f"{i}_y"] = self.agents[i].pos[1] - 1
                obs[f"{i}_terminated"] = int(self.agents[i].terminated)
                    
            obs["goal1_x"] = self.goal1[0] - 1
            obs["goal1_y"] = self.goal1[1] - 1
            obs["goal1_terminated"] = int(self.goal1_terminated)

            obs["goal2_x"] = self.goal2[0] - 1
            obs["goal2_y"] = self.goal2[1] - 1
            obs["goal2_terminated"] = int(self.goal2_terminated)


            # final = {}
            # for i in range(self.num_agents):
            #     temp = obs.copy()
            #     for j in range(self.num_agents):
            #         temp[f"is_{j}"] = int(i == j)
            #     final[i] = temp

            return {i:obs for i in range(self.num_agents)} # final

        else:
            direction = self.agent_states.dir
        image = gen_obs_grid_encoding(
            self.grid.state,
            self.agent_states,
            self.agents[0].view_size,
            self.agents[0].see_through_walls,
        )

        observations = {}
        for i in range(self.num_agents):
            observations[i] = {
                'image': image[i],
                'direction': direction[i],
                'mission': self.agents[i].mission,
            }

        return observations

    def _gen_grid(self):
        """
        :meta private:
        """
        
        # Create an empty grid
        self.grid = Grid(self.size, self.size)
        idxs = [i for i in range(len(self.COORD_PAIRS))]

        # Generate the surrounding walls
        # self.grid.wall_rect(0, 0, width, height)

        goal_1_idx = random.choice(idxs)
        idxs.pop(goal_1_idx)
        goal_2_idx = random.choice(idxs)
        idxs.pop(goal_2_idx)
        
        # Place a goal square in the bottom-right corner
        self.place_goals(self.COORD_PAIRS[goal_1_idx], self.COORD_PAIRS[goal_2_idx])

        self.goal1_terminated = False
        self.goal2_terminated = False

        # Place the agent
        # for agent in self.agents:
        #     # Setting team colors
        #     if agent.index < self.num_agents - (self.num_agents // 2):
        #         agent.state.color = "blue"
        #     else:       
        #         agent.state.color = "red"
            
        #     if self.agent_start_pos is not None and self.agent_start_dir is not None:
        #         agent.state.pos =  self.agent_start_pos
        #         agent.state.dir = self.agent_start_dir
        #     else:
        #         self.place_agent(agent)

        team_1_agents = [self.COORD_PAIRS[random.choice(idxs)] for i in range(self.num_team1)]
        team_2_agents = [self.COORD_PAIRS[random.choice(idxs)] for i in range(self.num_team2)]
        self.grid.place_agents(team_1_agents, team_2_agents)

            
