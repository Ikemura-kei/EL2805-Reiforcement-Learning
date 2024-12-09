from dataclasses import dataclass
import itertools
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED = '#FFC4CC'
RED = '#FF0000'
LIGHT_GREEN = '#95FD99'
BLACK = '#000000'
WHITE = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'


@dataclass
class State:
    player: tuple
    minotaur: tuple
    key: bool

    def canonical(self):
        return (self.player, self.minotaur, self.key)

    def __hash__(self) -> int:
        return hash(self.canonical())

    def __eq__(self, other) -> bool:
        return self.canonical() == other.canonical()

    def __str__(self) -> str:
        return str(self.canonical())


class MazeRef:

    # Actions
    STAY = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Standard reward values
    STEP_REWARD = 0
    GOAL_REWARD = 1
    IMPOSSIBLE_REWARD = 0
    EATEN_REWARD = 0

    def __init__(self, maze, simultaneous=True, minotaur_can_stay=False, poison_prob=1/50, minotaur_direct_move_prob=0.35):
        """ Constructor of the environment Maze.
        """
        # Reward values

        self.maze = maze
        self.simultaneous = simultaneous
        self.minotaur_can_stay = minotaur_can_stay
        self.poison_prob = poison_prob
        self.minotaur_direct_move_prob = minotaur_direct_move_prob

        self.minotaur_start = tuple(np.array(np.where(self.maze == 2))[:, 0])
        self.player_start = (0, 0)

        self.actions = self.__actions()
        self.states, self.map = self.__states()
        self.n_actions = len(self.actions)
        self.n_states = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.reset_rewards()

    def reset_rewards(self):
        self.set_rewards(MazeRef.STEP_REWARD, MazeRef.GOAL_REWARD,
                         MazeRef.IMPOSSIBLE_REWARD, MazeRef.EATEN_REWARD)

    def set_rewards(self, step, goal, impossible, eaten):
        self.STEP_REWARD = step
        self.GOAL_REWARD = goal
        self.IMPOSSIBLE_REWARD = impossible
        self.EATEN_REWARD = eaten
        self.rewards = self.__rewards()

    def __actions(self):
        actions = dict()
        actions[self.STAY] = (0, 0)
        actions[self.MOVE_LEFT] = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP] = (-1, 0)
        actions[self.MOVE_DOWN] = (1, 0)
        return actions

    def __states(self):
        states = dict()
        map = dict()
        s = 0
        n, m = self.maze.shape
        for i, j, k, l, b in itertools.product(range(n), range(m), range(n), range(m), range(2)):
            if self.maze[i, j] != 1:
                state_representation = State((i, j), (k, l), b)
                states[s] = state_representation
                map[state_representation] = s
                s += 1
        return states, map

    def __player_move(self, s, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        if self.__done(s):
            return s
        # Compute the future position given current (state, action)
        state = self.states[s]
        ((i, j), (k, l)) = state.player, state.minotaur
        row = i + self.actions[action][0]
        col = j + self.actions[action][1]
        # Is the future position an impossible one ?
        hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
            (col == -1) or (col == self.maze.shape[1]) or \
            (self.maze[row, col] == 1)  # in obstacle
        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return s
        else:
            key = 1 if state.key or self.maze[(row, col)] == 3 else 0
            return self.map[State((row, col), (k, l), key)]

    def __minotaur_move(self, s_after_player, s_before_player, minotaur_action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        if self.__done(s_before_player if self.simultaneous else s_after_player):
            return s_after_player
        # Compute the future position given current (state, action)
        state = self.states[s_after_player]
        ((i, j), (k, l)) = state.player, state.minotaur
        row = k + self.actions[minotaur_action][0]
        col = l + self.actions[minotaur_action][1]
        # Is the future position an impossible one ?
        hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
            (col == -1) or (col == self.maze.shape[1])
        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return s_after_player
        else:
            return self.map[State((i, j), (row, col), state.key)]

    def __minotaur_moves(self, s_after_player, s_before_player):
        if self.__done(s_before_player if self.simultaneous else s_after_player):
            return [s_after_player]
        minotaur_moves = set()
        for a in range(self.n_actions):
            new_state = self.__minotaur_move(
                s_after_player, s_before_player, a)
            if new_state != s_after_player or self.minotaur_can_stay:
                minotaur_moves.add(new_state)
        return minotaur_moves

    def __next_states(self, s, action):
        if self.__done(s):
            return [s]
        else:
            s_after_player = self.__player_move(s, action)
            next_states = self.__minotaur_moves(s_after_player, s)
            return next_states

    def __player(self, s):
        return self.states[s].player

    def __minotaur(self, s):
        return self.states[s].minotaur

    def eaten(self, s):
        return self.__minotaur(s) == self.__player(s)

    def __at_exit(self, state):
        return self.maze[self.__player(state)] == 2 and self.states[state].key

    def win(self, state):
        return self.__at_exit(state) and not self.eaten(state)

    def lose(self, state):
        return not self.win(state)

    def __done(self, state):
        return self.eaten(state) or self.__at_exit(state)

    def dist(self, s):
        state = self.states[s]
        (i, j), (k, l) = state.player, state.minotaur
        return self.manhattan_distance((i, j), (k, l))

    def manhattan_distance(self, p1, p2):
        (i, j), (k, l) = p1, p2
        return abs(i-k) + abs(j-l)

    def dist_2(self, s1, s2):
        p1, p2 = self.__player(s1), self.__minotaur(s2)
        return self.manhattan_distance(p1, p2)

    def best_minotaur_move(self, s, next_states):
        if self.simultaneous:
            return min(next_states, key=lambda x: self.dist_2(s, x))
        else:
            return min(next_states, key=lambda x: self.dist(x))

    def start_state(self, player=None, minotaur=None, key=0):
        player = self.player_start if player is None else player
        minotaur = self.minotaur_start if minotaur is None else minotaur
        start_state = self.map[State(player, minotaur, key)]
        return start_state

    def reset(self):
        return self.start_state()

    def step(self, s, a):
        next_s = np.random.choice(
            self.n_states, p=self.transition_probabilities[:, s, a])
        reward = self.rewards[s, a]
        done = self.__done(s)
        return next_s, reward, done, {}, {}

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            if self.__done(s):  # logic is already in __next_states
                transition_probabilities[s, s, :] = 1
            else:
                for a in range(self.n_actions):
                    next_states = self.__next_states(s, a)
                    for next_state in next_states:
                        transition_probabilities[next_state,
                                                 s, a] += 1/len(next_states)

                    if self.minotaur_direct_move_prob != 0:
                        transition_probabilities[:, s, a] *= (1 -
                                                              self.minotaur_direct_move_prob)
                        best_minotaur_move = self.best_minotaur_move(
                            s, next_states)
                        transition_probabilities[best_minotaur_move,
                                                 s, a] += self.minotaur_direct_move_prob

            if self.poison_prob != 0 and not self.__done(s):
                (i, j) = self.__player(s)
                transition_probabilities[:, s, :] *= (1-self.poison_prob)
                transition_probabilities[self.map[State(
                    (i, j), (i, j), self.states[s].key)], s, :] += self.poison_prob

        for s in range(self.n_states):
            for a in range(self.n_actions):
                pass
                if abs(transition_probabilities[:, s, a].sum()-1) > 10**-5:
                    print(s, a)
                    print(np.where(transition_probabilities[:, s, a]))
                    print(transition_probabilities[:, s, a][np.where(
                        transition_probabilities[:, s, a])])

        return transition_probabilities

    def __rewards(self):

        rewards = np.zeros((self.n_states, self.n_actions))

        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s = self.__player_move(s, a)
                # Reward for being eaten
                if self.eaten(s):
                    rewards[s, a] = self.EATEN_REWARD
                # Reward for reaching the goal
                elif self.__at_exit(s):
                    rewards[s, a] = self.GOAL_REWARD
                # Reward for hitting a wall
                elif self.__player(s) == self.__player(next_s) and a != self.STAY:
                    rewards[s, a] = self.IMPOSSIBLE_REWARD
                # Reward for taking a step to an empty cell that is not the exit
                else:
                    rewards[s, a] = self.STEP_REWARD

        for s in range(self.n_states):
            if self.eaten(s):
                rewards[s, :] = self.EATEN_REWARD

        return rewards

    def simulate(self, player_start, policy, method, minotaur_start=None, key=0):
        """ Simulates the agent in the maze."""
        if minotaur_start is None:
            minotaur_start = tuple(np.array(np.where(self.maze == 2))[:, 0])
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        start = State(player_start, minotaur_start, key)

        path = list()
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1]
            # Initialize current state and time
            t = 0
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = np.random.choice(
                    self.n_states, p=self.transition_probabilities[:, s, int(policy[s, t])])
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
                s = next_s
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            # Move to next state given the policy and the current state
            next_s = np.random.choice(
                self.n_states, p=self.transition_probabilities[:, s, int(policy[s])])
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # Loop while state is not the goal state
            while s != next_s:
                # Update state
                s = next_s
                # Move to next state given the policy and the current state
                next_s = np.random.choice(
                    self.n_states, p=self.transition_probabilities[:, s, int(policy[s])])
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
        return path

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)


def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions
    T = horizon

    # The variables involved in the dynamic programming backwards recursions
    V = np.zeros((n_states, T+1))
    policy = np.zeros((n_states, T+1))
    Q = np.zeros((n_states, n_actions))

    # Initialization
    Q = np.copy(r)
    V[:, T] = np.max(Q, 1)
    policy[:, T] = np.argmax(Q, 1)

    # The dynamic programming bakwards recursion
    for t in range(T-1, -1, -1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s, a] = r[s, a] + np.dot(p[:, s, a], V[:, t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:, t] = np.max(Q, 1)
        # The optimal action is the one that maximizes the Q function
        policy[:, t] = np.argmax(Q, 1)
    return V, policy


def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    BV = np.zeros(n_states)
    # Iteration counter
    n = 0
    # Tolerance error
    tol = (1 - gamma) * epsilon/gamma

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:, s, a], V)
    BV = np.max(Q, 1)

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1
        # Update the value function
        V = np.copy(BV)
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:, s, a], V)
        BV = np.max(Q, 1)
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q, 1)
    # Return the obtained policy
    return V, policy


def draw_maze(maze):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK,
               2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    rows, cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)


def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK,
               2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Size of the maze
    rows, cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    # Update the color at each frame
    for t in range(len(path)):
        (i, j), (k, l) = path[t]
        grid.get_celld()[(i, j)].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(i, j)].get_text().set_text('Player')
        grid.get_celld()[(k, l)].set_facecolor(RED)
        grid.get_celld()[(k, l)].get_text().set_text('Minotaur')
        if t > 0:
            if (i, j) == (k, l):
                grid.get_celld()[(i, j)].set_facecolor(RED)
                grid.get_celld()[(i, j)].get_text().set_text(
                    'Player was eaten')
            elif maze[i, j] == 2:
                grid.get_celld()[(i, j)].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(i, j)].get_text().set_text(
                    'Player is out')

            if path[t][0] != path[t-1][0] != path[t][1]:
                grid.get_celld()[(path[t-1][0])
                                 ].set_facecolor(col_map[maze[path[t-1][0]]])
                grid.get_celld()[(path[t-1][0])].get_text().set_text('')
            if path[t][1] != path[t-1][1] != path[t][0]:
                grid.get_celld()[(path[t-1][1])
                                 ].set_facecolor(col_map[maze[path[t-1][1]]])
                grid.get_celld()[(path[t-1][1])].get_text().set_text('')
        display.display(fig)
        fig.show()
        display.clear_output(wait=True)
        time.sleep(0.5)
