import torch

def calculate_reward(state, action, n_agents, dim):
    """
    finds reward with reference to the adversary, not the team

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
        if state[f"{agent}_term"]:
            continue

        if not state["goal1_term"] and action_map[action[agent]](state[f"{agent}_x"], state[f"{agent}_y"]) == (state["goal1_x"], state["goal1_y"]):
            total_reward += -1 if agent < n_agents - (n_agents // 2)  else 1

        if not state["goal2_term"] and action_map[action[agent]](state[f"{agent}_x"], state[f"{agent}_y"]) == (state["goal2_x"], state["goal2_y"]):
            total_reward += -1 if agent < n_agents - (n_agents // 2)  else 1

    return total_reward



def generate_reward_3x3():
    dims = [4, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 3, 3, 2, 4, 4]
    table = torch.zeros(dims)
    for a1 in range(4):
        for x1 in range(3):
            for y1 in range(3):
                for term_1 in range(2):
                    for x2 in range(3):
                        for y2 in range(3):
                            for term_2 in range(2):
                                for x3 in range(3):
                                    for y3 in range(3):
                                        for term_3 in range(2):
                                            for goal1_x in range(3):
                                                for goal1_y in range(3):
                                                    for goal1_term in range(2):
                                                        for goal2_x in range(3):
                                                            for goal2_y in range(3):
                                                                for goal2_term in range(2):
                                                                    for a2 in range(4):
                                                                        for a3 in range(4):
                                                                            state = {
                                                                                "1_x": x1,
                                                                                "1_y": y1,
                                                                                "1_term":term_1,
                                                                                "2_x": x2,
                                                                                "2_y": y2,
                                                                                "2_term":term_2,
                                                                                "3_x": x3,
                                                                                "3_y": y3,
                                                                                "3_term":term_3,
                                                                                "goal1_x": goal1_x,
                                                                                "goal1_y": goal1_y,
                                                                                "goal1_term": goal1_term,
                                                                                "goal2_x": goal2_x,
                                                                                "goal2_y": goal2_y,
                                                                                "goal2_term": goal2_term
                                                                            }
                                                                            table[a1, x1, y1, term_1, x2, y2, term_2, x3, y3, term_3,
                                                                                  goal1_x, goal1_y, goal1_term, goal2_x, goal2_y, 
                                                                                  goal2_term, a2, a3] = calculate_reward(state, (a1, a2, a3), 3)
    torch.save(table, "3x3-3-agents-table.pt")
