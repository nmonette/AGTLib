import random

import gymnasium as gym
import numpy as np

# Agent
# Goal
# 

TEAM1 = 0
TEAM2 = 1
GOAL = 2

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

class Grid:
    agent_locations_t1: list[tuple[int, int]] = []
    agent_locations_t2: list[tuple[int, int]] = []
    goal_locations: list[tuple[int, int]] = []
    goal1_terminated: bool = False
    goal2_terminated:bool = False
    terminations: list[bool, ]
    num_steps: int = 0
 
    def __init__(self, dim:int, num_agents_t1:int, num_agents_t2:int, max_episode_steps:int = 12) -> None:
        available_coord_pairs = [(i,j) for i in range(0,dim) for j in range(0,dim)]
        self.dim = dim
        self.num_agents_t1 = num_agents_t1
        self.num_agents_t2 = num_agents_t2
        self.terminations = {i:False for i in range(num_agents_t1 + num_agents_t2)}

        self.max_episode_steps = max_episode_steps

        # idxs = [i for i in range()]
        goal_1 = random.choice(available_coord_pairs)
        # print(available_coord_pairs)
        available_coord_pairs.remove(goal_1)
        goal_2 = random.choice(available_coord_pairs)
        # print(available_coord_pairs)
        available_coord_pairs.remove(goal_2)
        # print(available_coord_pairs)
        
        self.place_goals(goal_1, goal_2)

        self.goal1_terminated = False
        self.goal2_terminated = False

        team_1_agents = [random.choice(available_coord_pairs) for i in range(self.num_agents_t1)]
        team_2_agents = [random.choice(available_coord_pairs) for i in range(self.num_agents_t2)]
        self.place_agents(team_1_agents, team_2_agents)

        self.done = {i:False for i in range(self.num_agents_t1 + self.num_agents_t2)}

        
    def handle_actions(self, actions) -> np.ndarray:
        self.num_steps += 1
        rewards = {i:0 for i in range(self.num_agents_t1 + self.num_agents_t2 )}
        for i in range(len(actions)):
            if self.terminations[i]:
                continue

            # Hacky-ish, should improve
            locations = self.agent_locations_t1 if i < self.num_agents_t1 else self.agent_locations_t2
            if i >= self.num_agents_t1:
                i -= self.num_agents_t1

            team = i < self.num_agents_t1 # True if team 1 else False
            x,y = locations[i]
            if actions[i] == UP and y < self.dim - 1:locations[i] = (x, y+1)
            elif actions[i] == DOWN and y > 0:
                locations[i] = (x,y-1)
            elif actions[i] == LEFT and x > 0:
                locations[i] = (x-1, y)
            elif actions[i] == RIGHT and x < self.dim - 1:
                locations[i] = (x+1,y)
            else:
                # invalid action
                continue

            reward = 1 # can be changed if reward should be changed (e.g. discounted based on number of steps)
            if locations[i] == self.goal_locations[0]:
                self.terminations[i] = True
                self.goal1_terminated = True
                for i in range(self.num_agents_t1):
                    rewards[i] += reward * (1 if team else 1)
                for i in range(self.num_agents_t1, self.num_agents_t2):
                    rewards[i] += reward * (-1 if team else 1)
            elif locations[i] == self.goal_locations[1]:
                self.terminations[i] = True
                self.goal2_terminated = True
                for i in range(self.num_agents_t1):
                    rewards[i] += reward * (1 if team else 1)
                for i in range(self.num_agents_t1, self.num_agents_t2):
                    rewards[i] += reward * (-1 if team else 1)
            else:
                continue

            if (self.goal1_terminated and self.goal2_terminated) or self.num_steps >= self.max_episode_steps:
                self.done = {i:True for i in range(self.num_agents_t1+self.num_agents_t2)}

        return rewards, self.terminations, self.done
        
    def place_goals(self, *goals: list[tuple[int, int]]):
         self.goal_locations = goals

    def place_agents(self, team_one: list[tuple[int, int]], team_two: list[tuple[int, int]]):
        self.agent_locations_t1 = team_one
        self.agent_locations_t2 = team_two

    def get_state(self) -> np.ndarray:
        """
        returns the following vector:
        [x1, y1, 1_terminated, x2, y2, 2_terminated,..., x_goal1, y_goal1, goal1_terminated, x_goal2 y_goal2, goal2_terminated]
        terminations are bools but they can be written as 0 and 1
        
        """
        i = 0
        agents = []
        for agent_arr in [self.agent_locations_t1, self.agent_locations_t2]:
            for (x, y) in agent_arr:
                terminated = self.terminations[i]
                agents.extend([x, y, 1 if terminated else 0])
                i += 1
        
        goals = []
        for i, (x, y) in enumerate(self.goal_locations):
            terminated = self.goal1_terminated if i == 0 else self.goal2_terminated
            goals.extend([x, y, 1 if terminated else 0])

        return np.array([
            *agents,
            *goals
        ])


