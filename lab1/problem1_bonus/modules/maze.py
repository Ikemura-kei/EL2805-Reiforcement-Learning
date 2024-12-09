import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import cv2

LIGHT_RED = '#FFC4CC'
LIGHT_GREEN = '#95FD99'
BLACK = '#000000'
WHITE = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
BLUE = '#ffbb11'
KEY_GOT = '#11bb11'


class MazeEnv():
    AGENT_ACTION_STAY = 0
    AGENT_ACTION_LEFT = 1
    AGENT_ACTION_RIGHT = 2
    AGENT_ACTION_UP = 3
    AGENT_ACTION_DOWN = 4

    AGENT_ACTION_NAMES = {3: 'AGENT_ACTION_UP', 1: 'AGENT_ACTION_LEFT',
                          4: 'AGENT_ACTION_DOWN', 2: 'AGENT_ACTION_RIGHT', 0: 'AGENT_ACTION_STAY'}

    AGENT_ACTION_MOVES = {3: [-1, 0], 1: [0, -1],
                          4: [1, 0], 2: [0, 1], 0: [0, 0]}

    MINOTAUR_ACTION_MOVES = {0: [-1, 0], 1: [0, -1],
                             2: [1, 0], 3: [0, 1]}

    STEP_REWARD = -1
    IMPOSSIBLE_REWARD = -10
    WIN_REWARD = 1e5
    LOSE_REWARD = -1e3
    KEY_REWARD = 1e3

    def __init__(self, maze, episode_len=1000, save_frames=False, log_dir=None, beta=0.65, minotaur_fix=False):
        self.maze = maze
        H, W = maze.shape
        self.state_space_dim = 5
        self.beta = beta
        self.minotaur_fix = minotaur_fix

        self.episode_len = episode_len
        self.save_frames = save_frames
        self.log_dir = log_dir

        # -- a mapping between state index (dim0) and the state value (dim1) --
        self.idx_to_states = []
        self.states_to_idx = {}
        cnt = 0

        for ia in range(H):
            for ja in range(W):
                if maze[ia, ja] == 1:  # is obstacle, agent can't be in this state anyways, skip
                    continue

                for im in range(H):
                    for jm in range(W):
                        for key_state in [0, 1]:
                            self.idx_to_states.append(
                                (ia, ja, key_state, im, jm))
                            self.states_to_idx[(
                                ia, ja, key_state, im, jm)] = cnt
                            cnt += 1

        self.idx_to_states.append('win')
        self.states_to_idx['win'] = cnt
        self.idx_to_states.append('lose')
        self.states_to_idx['lose'] = cnt + 1

        self.n_states = len(self.idx_to_states)
        print("==> Number of possible states:", self.n_states)

        # -- actions --
        self.minotaur_num_actions = 4
        self.n_actions = 5
        self.agent_actions = np.array(range(self.n_actions))

        # -- prepare animation --
        self.animation_canvas = self._prep_animation()

        # -- pre-compute transition probabilties --
        self.trans_probs = self._make_transition_probs()

        # -- pre-compute rewards --
        self.rewards = self._make_rewards()

    def reset(self):
        return self.states_to_idx[(0, 0, 0, 6, 4)]

    def step(self, s_idx, a_idx, time=0, do_render=False, show=False):
        # -- sample a next state --
        s_prime_idx = np.random.choice(
            self.n_states, p=self.trans_probs[:, s_idx, a_idx])

        # -- get reward --
        reward = self.rewards[s_idx, a_idx]

        info = {}

        # -- compute terminated --
        terminated = self.idx_to_states[s_idx] == 'win' or self.idx_to_states[s_idx] == 'lose'

        # -- compute truncated --
        truncated = time >= self.episode_len

        if do_render:
            frame = self._render(
                self.idx_to_states[s_prime_idx], self.idx_to_states[s_idx], show)

        if (terminated or truncated) and do_render:
            self._reset_render(self.idx_to_states[s_prime_idx])

        return s_prime_idx, reward, terminated, truncated, info

    def _make_transition_probs(self):
        trans_probs = np.zeros((self.n_states, self.n_states, self.n_actions))

        for s_idx in range(self.n_states):
            state = self.idx_to_states[s_idx]
            if state == 'win' or state == 'lose':
                trans_probs[s_idx, s_idx, :] = 1  # in absorbing states
            else:
                for a_idx in range(self.n_actions):
                    s_prime_candidate_idx = self._possible_next_states(
                        s_idx, a_idx)
                    m = len(s_prime_candidate_idx) - 1
                    if m == 0:  # either win or lose
                        trans_probs[s_prime_candidate_idx[0], s_idx, a_idx] = 1
                    else:  # not win nor lose, proceed with movements
                        for k, s_prime_idx in enumerate(s_prime_candidate_idx):
                            if k != (len(s_prime_candidate_idx) - 1):
                                trans_probs[s_prime_idx, s_idx,
                                            a_idx] += self.beta / float(m)
                            else:
                                trans_probs[s_prime_idx, s_idx,
                                            a_idx] += (1-self.beta)

        return trans_probs

    def _make_rewards(self):
        rewards = np.zeros((self.n_states, self.n_actions))

        for s_idx in range(self.n_states):
            for a_idx in range(self.n_actions):

                if self.idx_to_states[s_idx] == 'lose':
                    rewards[s_idx, a_idx] += self.LOSE_REWARD
                elif self.idx_to_states[s_idx] == 'win':
                    rewards[s_idx, a_idx] += self.WIN_REWARD
                else:
                    s_prime_agent, agent_move_impossible, on_key = self._agent_possible_next_states(
                        s_idx, a_idx)
                    if agent_move_impossible:
                        rewards[s_idx, a_idx] += self.IMPOSSIBLE_REWARD
                    else:
                        rewards[s_idx, a_idx] += self.STEP_REWARD
                        if on_key:
                            rewards[s_idx, a_idx] += self.KEY_REWARD

        return rewards

    def _possible_next_states(self, s_idx, a_idx):
        s_prime_candidate_idx = []

        state = self.idx_to_states[s_idx]
        if state[0] == state[3] and state[1] == state[4]:
            s_prime_candidate_idx.append(self.states_to_idx['lose'])
        elif self.maze[state[0], state[1]] == 2 and state[2] == 1:
            s_prime_candidate_idx.append(self.states_to_idx['win'])
        else:
            # -- check agent next states --
            next_agent_state, agent_move_impossible, on_key = self._agent_possible_next_states(
                s_idx, a_idx)

            # -- check minotaur next states --
            if not self.minotaur_fix:
                # NOTE: output includes the best move (i.e. the best move will be duplicated, one accounting for uniform motion, and one accounting for the best movement)
                next_minotaur_states = self._minotaur_possible_next_states(
                    s_idx)
            else:
                next_minotaur_states = [(self.idx_to_states[s_idx][3], self.idx_to_states[s_idx][4]), (
                    self.idx_to_states[s_idx][3], self.idx_to_states[s_idx][4])]

            for next_minotaur_state in next_minotaur_states:
                s_prime_candidate_idx.append(
                    self.states_to_idx[(*next_agent_state, *next_minotaur_state)])

        return s_prime_candidate_idx

    def _reset_render(self, ended_s_idx):
        if ended_s_idx != 'lose' and ended_s_idx != 'win':
            self.grid.get_celld()[(ended_s_idx[0], ended_s_idx[1])].set_facecolor(
                self.col_map[self.maze[ended_s_idx[0], ended_s_idx[1]]])  # Position of the player
            self.grid.get_celld()[(ended_s_idx[3], ended_s_idx[4])].set_facecolor(
                self.col_map[self.maze[ended_s_idx[3], ended_s_idx[4]]])  # Position of the minotaur
        self.animation_canvas.canvas.draw()
        img_plot = np.array(
            self.animation_canvas.canvas.renderer.buffer_rgba())
        frame = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)

    def _win_condition(self, i, j, key):
        return self.maze[i, j] == 2 and key == 1

    def _is_inbound(self, i, j):
        H = self.maze.shape[0]  # max row
        W = self.maze.shape[1]  # max col
        left_out_of_bound = j < 0
        right_out_of_bound = j >= W
        top_out_of_bound = i < 0
        bot_out_of_bound = i >= H
        is_inbound = not (
            left_out_of_bound or right_out_of_bound or top_out_of_bound or bot_out_of_bound)

        return is_inbound

    def _agent_possible_next_states(self, s_idx, a_idx):
        cur_agent_state = list(self.idx_to_states[s_idx][:3])
        next_agent_state = cur_agent_state
        ia = cur_agent_state[0] + self.AGENT_ACTION_MOVES[a_idx][0]
        ja = cur_agent_state[1] + self.AGENT_ACTION_MOVES[a_idx][1]

        is_inbound = self._is_inbound(ia, ja)

        agent_move_impossible = False
        if not is_inbound or self.maze[ia, ja] == 1:
            agent_move_impossible = True
            next_agent_state = cur_agent_state
        else:  # movement possible
            next_agent_state[0] = ia
            next_agent_state[1] = ja

        # -- check if we are on key --
        on_key = (self.maze[next_agent_state[0], next_agent_state[1]] == 3) and (
            cur_agent_state[2] != 1)
        if on_key:
            next_agent_state[2] = 1

        return next_agent_state, agent_move_impossible, on_key

    def _minotaur_possible_next_states(self, s_idx):
        cur_agent_state = self.idx_to_states[s_idx][:3]
        cur_minotaur_state = self.idx_to_states[s_idx][3:]
        next_minotaur_states = []

        for a_minotaur_idx in range(self.minotaur_num_actions):
            im = cur_minotaur_state[0] + \
                self.MINOTAUR_ACTION_MOVES[a_minotaur_idx][0]
            jm = cur_minotaur_state[1] + \
                self.MINOTAUR_ACTION_MOVES[a_minotaur_idx][1]
            is_inbound = self._is_inbound(im, jm)

            if is_inbound:
                next_minotaur_states.append([im, jm])

        min_dist_2_agent = 1e10
        best_move = None
        for next_minotaur_state in next_minotaur_states:
            dist_2_agent = np.linalg.norm(
                np.array(next_minotaur_state) - np.array(cur_agent_state[:2]))
            if dist_2_agent < min_dist_2_agent:
                min_dist_2_agent = dist_2_agent
                best_move = [next_minotaur_state[0], next_minotaur_state[1]]

        next_minotaur_states.append(best_move)
        return next_minotaur_states

    def _prep_animation(self):
        # Map a color to each cell in the maze
        self.col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -
                        1: LIGHT_RED, -2: LIGHT_PURPLE, 3: BLUE}

        rows, cols = self.maze.shape  # Size of the maze
        # Create figure of the size of the maze
        fig = plt.figure(1, figsize=(cols, rows))

        # Remove the axis ticks and add title
        ax = plt.gca()
        ax.set_title('Policy simulation')
        ax.set_xticks([])
        ax.set_yticks([])

        # Give a color to each cell
        colored_maze = [[self.col_map[self.maze[j, i]]
                         for i in range(cols)] for j in range(rows)]

        # Create a table to color
        self.grid = plt.table(
            cellText=None,
            cellColours=colored_maze,
            cellLoc='center',
            loc=(0, 0),
            edges='closed'
        )

        # Modify the height and width of the cells in the table
        tc = self.grid.properties()['children']
        for cell in tc:
            cell.set_height(1.0/rows)
            cell.set_width(1.0/cols)

        return fig

    def _render(self, state, prev_state, show):
        if prev_state != 'lose' and prev_state != 'win':
            self.grid.get_celld()[(prev_state[0], prev_state[1])].set_facecolor(
                self.col_map[self.maze[prev_state[0], prev_state[1]]])  # Position of the player
            self.grid.get_celld()[(prev_state[3], prev_state[4])].set_facecolor(
                self.col_map[self.maze[prev_state[3], prev_state[4]]])  # Position of the minotaur

        if state != 'lose' and state != 'win':
            if state[2] == 1:
                self.grid.get_celld()[((0, 7))].set_facecolor(
                    KEY_GOT)  # Position of the player
            self.grid.get_celld()[(state[0], state[1])].set_facecolor(
                self.col_map[-2])  # Position of the player
            self.grid.get_celld()[(state[3], state[4])].set_facecolor(
                self.col_map[-1])  # Position of the minotaur

        if state == 'lose':
            # print("You are eaten!")
            pass
        elif state == 'win':
            # print("You escaped!")
            pass

        self.animation_canvas.canvas.draw()
        img_plot = np.array(
            self.animation_canvas.canvas.renderer.buffer_rgba())
        frame = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
        if show:
            cv2.imshow('Image', frame)
            cv2.waitKey(50)

        if self.save_frames:
            return frame
        else:
            return None


if __name__ == "__main__":
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 3],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]])

    horizon = 1000
    maze_test = MazeEnv(maze, episode_len=horizon, beta=1.0, minotaur_fix=True)

    start = (6, 6, 1, 4, 5)
    s_idx = maze_test.states_to_idx[start]

    for i in range(horizon):
        # -- random policy --
        a_idx = np.random.randint(0, maze_test.n_actions)

        s_prime_idx, reward, terminated, truncated, info = maze_test.step(
            s_idx, a_idx, i, do_render=True, show=True)

        s_idx = s_prime_idx
