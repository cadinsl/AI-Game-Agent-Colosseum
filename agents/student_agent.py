# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
from copy import deepcopy
from time import time
import numpy as np

moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
opposites = {0: 2, 1: 3, 2: 0, 3: 1}
start_time = None
round_time = 1.5
first_round_time = 25
round = 0

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.autoplay = True
        global round
        round = 0

    def step(self, chess_board, my_pos, adv_pos, max_step):
        global start_time
        global round
        start_time = time()
        round +=1
        state = State(my_pos, adv_pos, chess_board)
        mcnode = MonteCarloNode(state, max_step)
        best_node = mcnode.best_action()
        best_node_action = best_node.parent_action
        return best_node_action.pos, best_node_action.dir


#Action is a step that a player can take
class Action:
    def __init__(self, pos, dir):
        self.pos = pos
        self.dir = dir
    def __str__(self):
        x, y = self.pos
        return "Action: position: X=" + str(x.item()) + " Y=" + str(y.item()) + " direction="+ str(self.dir)
    def __eq__(self, other):
        x, y = self.pos
        o_x, o_y = other.pos
        return x == o_x and y == o_y and self.dir == other.dir
    def __hash__(self):
        x, y = self.pos
        return hash((x.item(), y.item(), self.dir))
#A state is used my the monte carlo node to save the current game
class State:
    def __init__(self, my_pos, adv_pos, chess_board):
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.chess_board = chess_board
#Monte carlo node used by tree to store information
class MonteCarloNode():
    def __init__(self, state, max_step, parent=None, parent_action=None):
        self.state = state
        self.max_step = max_step
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self.visits= 0
        self.wins = 0
        self.draws = 0
        self.loses = 0
        self.untried_actions = self.untried_actions()
    #gets the possible action the ai can take from this node state
    def untried_actions(self):
        possibleActions = BFS(self.state.my_pos, self.max_step, self.state.chess_board, self.state.adv_pos)
        return possibleActions
    #q used for calculations of upper confidence
    def upper_confidence_q(self):
        return self.wins - self.loses

    #expands the node on the monte carlo tree
    def expand(self):
       
        action = get_best_untried_action(self.untried_actions, self.max_step, self.state)
        self.untried_actions.remove(action)
        next_state = step_tree(action.pos, self.state.adv_pos, self.max_step, deepcopy(self.state.chess_board), action.dir)
        child_node = MonteCarloNode(next_state, self.max_step, self, action)

        self.children.append(child_node)
        return child_node 

    #simulates a game for this specific node
    def rollout(self):
        current_rollout_state = self.state
        
        ended, result = check_endgame_result(current_rollout_state)

        while not ended:
            if check_pastTime():
                return 0
            #possibleActions=BFS(current_rollout_state.my_pos, self.max_step, current_rollout_state.chess_board, current_rollout_state.adv_pos)
            
            #action = self.rollout_policy(current_rollout_state.adv_pos, current_rollout_state.chess_board, possibleActions)
            current_rollout_state = step_rollout(current_rollout_state.my_pos, current_rollout_state.adv_pos, self.max_step, current_rollout_state.chess_board)
            ended, result = check_endgame_result(current_rollout_state)
        return result
    #policy to use which action to take during simulation (uses heuristic)
    def rollout_policy(self, adv_pos, chess_board, possible_actions):
        action = next(iter(possible_actions))
        best_h = get_heuristic_for_action(action, adv_pos, self.max_step, chess_board)
        best_p = action
        for p in possible_actions:
            if check_pastTime():
                return best_p
            p_heuristic = get_heuristic_for_action(p, adv_pos, self.max_step, chess_board)
            if p_heuristic > best_h:
                best_h = p_heuristic
                best_p = p
        return best_p
    #policy to choose which node from the tree to take (uses upper confidence)
    def _tree_policy(self):

        current_node = self

        while not check_endgame(current_node.state):
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_upper_confidence()
        return current_node


    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    #upper confidence calculation for tree policy
    def best_upper_confidence(self, c_param=0.1):
        max = -1
        best_node = None
        for c in self.children: 
            upper_confidence = (c.upper_confidence_q() / c.visits) + 1 * np.sqrt((2 * np.log(self.visits) / c.visits))
            if best_node is None:
                best_node = c
                max = upper_confidence
            elif upper_confidence > max:
                max = upper_confidence
                best_node = c

        return best_node

    def best_action(self):
        while not(check_pastTime()):
            
            node = self._tree_policy()
            result = node.rollout()
            node.backpropagate(result)
        
        best_action = self.best_upper_confidence()
        return best_action
    #sends results of simulation back to parent node
    def backpropagate(self, result):
        self.visits += 1.
        if result == -1:
            self.loses += 1
        elif result == 0:
            self.draws += 1
        elif result == 1:
            self.wins += 1
        if self.parent:
            self.parent.backpropagate(result)
        

def is_position_in_bounds(pos,chess_board):
    x, y = pos
    return 0 <= x < chess_board.shape[0] and 0 <= y < chess_board.shape[1]

#this heurisitc checks to see how many spaces would be available to the ai vs the spaces avaiable to the player
#it prioritizes the player having more actions available compared to the other player
def get_heuristic_for_action(action, adv_pos, max_step, chess_board):

    action_board = deepcopy(chess_board)
    set_barrier(action.pos, action_board, action.dir)

    possibleActions = BFS(action.pos, max_step, action_board, adv_pos)
    number_of_player_actions = len(possibleActions)

    possibleActions = BFS(adv_pos, max_step, action_board, action.pos)
    number_of_adv_actions = len(possibleActions)

    board_size = action_board.shape[0]
    
    return (number_of_player_actions * 1.5) - (number_of_adv_actions)

def get_best_untried_action(actions, max_step, state):
    best_p = next(iter(actions))
    best_h = get_heuristic_for_action(best_p, state.adv_pos, max_step, state.chess_board)
    for p in actions:
        if check_pastTime():
            return best_p
        p_heuristic = get_heuristic_for_action(p, state.adv_pos, max_step, state.chess_board)
        if p_heuristic > best_h:
            best_h = p_heuristic
            best_p = p
    return best_p

#Copied from world script
def check_endgame(state):

    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    # Union-Find
    father = dict()
    for r in range(state.chess_board.shape[0]):
        for c in range(state.chess_board.shape[1]):
            father[(r, c)] = (r, c)

    def find(pos):
        if father[pos] != pos:
            father[pos] = find(father[pos])
        return father[pos]

    def union(pos1, pos2):
        father[pos1] = pos2

    for r in range(state.chess_board.shape[0]):
        for c in range(state.chess_board.shape[1]):
            for dir, move in enumerate(
                moves[1:3]
            ):  # Only check down and right
                if state.chess_board[r, c, dir + 1]:
                    continue
                pos_a = find((r, c))
                pos_b = find((r + move[0], c + move[1]))
                if pos_a != pos_b:
                    union(pos_a, pos_b)

    for r in range(state.chess_board.shape[0]):
        for c in range(state.chess_board.shape[1]):
            find((r, c))
    p0_r = find(tuple(state.my_pos))
    p1_r = find(tuple(state.adv_pos))
    p0_score = list(father.values()).count(p0_r)
    p1_score = list(father.values()).count(p1_r)
    if p0_r == p1_r:
        return False
    else:
        return True
#Copied from world script
def check_endgame_result(state):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        # Union-Find
        father = dict()
        for r in range(state.chess_board.shape[0]):
            for c in range(state.chess_board.shape[1]):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(state.chess_board.shape[0]):
            for c in range(state.chess_board.shape[1]):
                for dir, move in enumerate(
                    moves[1:3]
                ):  # Only check down and right
                    if state.chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(state.chess_board.shape[0]):
            for c in range(state.chess_board.shape[1]):
                find((r, c))
        p0_r = find(tuple(state.my_pos))
        p1_r = find(tuple(state.adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, 0
        player_win = None
        win_blocks = -1
        if p0_score > p1_score:
            player_win = 1
            win_blocks = p0_score
        elif p0_score < p1_score:
            player_win = -1
            win_blocks = p1_score
        else:
            player_win = 0  # Tie
        return True, player_win

def step_rollout(my_pos, adv_pos, max_step, chess_board):
    step_my_pos, dir = random_walk(my_pos, adv_pos, max_step, chess_board)
    set_barrier(step_my_pos, chess_board, dir)
    if not check_endgame(State(step_my_pos, adv_pos, chess_board)):
        step_adv_pos, dir = random_walk(adv_pos, step_my_pos, max_step, chess_board)
        set_barrier(step_adv_pos, chess_board, dir)
        return State(step_my_pos, step_adv_pos, chess_board)
    else: 
        return State(step_my_pos, adv_pos, chess_board)

def step_tree(next_pos, adv_pos, max_step, chess_board, dir):
        step_chessboard = chess_board
        r, c = next_pos
        set_barrier(next_pos, step_chessboard,dir)
        new_state = State(next_pos, adv_pos, step_chessboard)
        gameEnded, result = check_endgame_result(new_state)
        if gameEnded:
            return new_state

        possible_actions = BFS(adv_pos, max_step, step_chessboard, next_pos)
        action = next(iter(possible_actions))
        best_h = get_heuristic_for_action(action, new_state.my_pos, max_step, step_chessboard)
        best_p = action
        for p in possible_actions:
            if check_pastTime():
                return State(next_pos, best_p.pos, step_chessboard)
            p_heuristic = get_heuristic_for_action(action, new_state.my_pos, max_step, step_chessboard)
            if p_heuristic > best_h:
                best_h = p_heuristic
                best_p = p
        adv_action = best_p


        set_barrier(adv_action.pos, step_chessboard, adv_action.dir)

        return State(next_pos, adv_pos, step_chessboard)
#Copied from world script
def set_barrier(pos, chess_board, dir):
        r, c = pos
        # Set the barrier to True
        chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = moves[dir]
        chess_board[r + move[0], c + move[1], opposites[dir]] = True
        return chess_board
#Copied from world script
def BFS(start_pos, max_step, chess_board, adv_pos):
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    state_queue = [(start_pos, 0)]
    visited = set()
    is_reached = False
    while state_queue:
        cur_pos, cur_step = state_queue.pop(0)
        r, c = cur_pos
        if cur_step == max_step:
            break
        for dir, move in enumerate(moves):
            if chess_board[r, c, dir]:
                continue
            visited.add(Action(cur_pos, dir))
            m_r, m_c = move
            next_pos = (r + m_r, c + m_c)
            if np.array_equal(next_pos, adv_pos) or Action(next_pos, dir) in visited:
                    continue

            if cur_step + 1 != max_step + 1:
                state_queue.append((next_pos, cur_step + 1))

    return visited

def random_walk(my_pos, adv_pos, max_step, chess_board):
        """
        Randomly walk to the next position in the board.

        Parameters
        ----------
        my_pos : tuple
            The position of the agent.
        adv_pos : tuple
            The position of the adversary.
        """
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        ori_pos = deepcopy(my_pos)
        steps = np.random.randint(0, max_step + 1)
        # Random Walk
        for _ in range(steps):
            r, c = my_pos
            dir = np.random.randint(0, 4)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

            # Special Case enclosed by Adversary
            k = 0
            while chess_board[r, c, dir] or my_pos == adv_pos:
                k += 1
                if k > 300:
                    break
                dir = np.random.randint(0, 4)
                m_r, m_c = moves[dir]
                my_pos = (r + m_r, c + m_c)

            if k > 300:
                my_pos = ori_pos
                break

        # Put Barrier
        dir = np.random.randint(0, 4)
        r, c = my_pos
        while chess_board[r, c, dir]:
            dir = np.random.randint(0, 4)

        return my_pos, dir

#checks to keep ai agent within time constraints
def check_pastTime():
    global start_time
    global round
    global first_round_time
    global round_time
    passedTime = time() - start_time
    if round <= 1:
        if passedTime >= first_round_time:
            
            return True
            
        else:
            return False
    else:
        if passedTime >= round_time:
            
            return True
        else:
            return False