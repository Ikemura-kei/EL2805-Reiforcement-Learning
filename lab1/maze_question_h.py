# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train')
args = parser.parse_args()

# Implemented methods
methods = ['DynProg', 'ValIter']
FIX_MINOTAUR = False
# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
BLUE = '#ffbb11'
KEY_GOT = '#11bb11'

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values 
    STEP_REWARD = 0
    GOAL_REWARD = 1
    IMPOSSIBLE_REWARD = 0
    MINOTAUR_REWARD = 0
    KEY_REWARD = 0
    
    state_of_key = 0
    is_key_retreived = False

    def __init__(self, maze):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze
        self.actions                  = self.__actions()
        self.states, self.map         = self.__states()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        # self.transition_probabilities = self.__transitions()
        self.rewards                  = self.__rewards()

    def __actions(self):
        actions = dict()
        actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1,0)
        return actions

    def __states(self):
        
        states = dict()
        map = dict()
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[0]):
                    for l in range(self.maze.shape[1]):
                        if self.maze[i,j] != 1:
                            states[s] = ((i,j), (k,l))
                            map[((i,j), (k,l))] = s
                            s += 1
        
        states[s] = 'Eaten'
        map['Eaten'] = s
        s += 1
        
        states[s] = 'Win'
        map['Win'] = s
        
        return states, map

    def __move(self, state, action):               
        """ Makes a step in the maze, given a current position and an action. 
            If the action STAY or an inadmissible action is used, the player stays in place.
        
            :return list of tuples next_state: Possible states ((x,y), (x',y')) on the maze that the system can transition to.
        """
        
        if self.states[state] == 'Eaten' or self.states[state] == 'Win': # In these states, the game is over
            return [self.states[state]]
        
        else: # Compute the future possible positions given current (state, action)
            row_player = self.states[state][0][0] + self.actions[action][0] # Row of the player's next position 
            col_player = self.states[state][0][1] + self.actions[action][1] # Column of the player's next position 
            
            # Is the player getting out of the limits of the maze or hitting a wall?
            
            # -- condition 1: player inside maze --
            H = self.maze.shape[0] # max row
            W = self.maze.shape[1] # max col 
            left_out_of_bound = col_player < 0
            right_out_of_bound = col_player >= W
            top_out_of_bound = row_player < 0
            bot_out_of_bound = row_player >= H
            is_inbound = not (left_out_of_bound or right_out_of_bound or top_out_of_bound or bot_out_of_bound)
            
            # -- condition 2: player not running into wall --
            if is_inbound:
                is_not_hitting_obstacle = self.maze[row_player, col_player] != 1
            else:
                is_not_hitting_obstacle = False
            
            impossible_action_player = not (is_inbound and is_not_hitting_obstacle)
        
            actions_minotaur = [[0, -1], [0, 1], [-1, 0], [1, 0]] # Possible moves for the Minotaur
            rows_minotaur, cols_minotaur = [], []
            min_distance = 1e10
            min_distance_pos = None
            for i in range(len(actions_minotaur)):
                # Is the minotaur getting out of the limits of the maze?
                impossible_action_minotaur = ((self.states[state][1][0] + actions_minotaur[i][0]) == -1) or \
                                             ((self.states[state][1][0] + actions_minotaur[i][0]) == self.maze.shape[0]) or \
                                             ((self.states[state][1][1] + actions_minotaur[i][1]) == -1) or \
                                             ((self.states[state][1][1] + actions_minotaur[i][1]) == self.maze.shape[1])
            
                if not impossible_action_minotaur:
                    rows_minotaur.append(self.states[state][1][0] + actions_minotaur[i][0])
                    cols_minotaur.append(self.states[state][1][1] + actions_minotaur[i][1])  
                    
                    j_dist = self.states[state][0][0] - (self.states[state][1][0] + actions_minotaur[i][0])
                    i_dist = self.states[state][0][1] - (self.states[state][1][1] + actions_minotaur[i][1])
                    distance = np.sqrt(j_dist**2 + i_dist**2)
                    if distance < min_distance:
                        min_distance = distance
                        min_distance_pos = (self.states[state][1][0] + actions_minotaur[i][0], self.states[state][1][1] + actions_minotaur[i][1])
                        
            # -- in addition, the minotaur could move towards the player, which is the action that reduces the distance between minotaur and player most --
            rows_minotaur.append(min_distance_pos[0])
            cols_minotaur.append(min_distance_pos[1])
          

            # Based on the impossiblity check return the next possible states.
            if impossible_action_player: # The action is not possible, so the player remains in place
                states = []
                for i in range(len(rows_minotaur)):
                    
                    if (rows_minotaur[i] == self.states[state][0][0] and cols_minotaur[i] == self.states[state][0][1]): # We met the minotaur
                        states.append('Eaten')
                    
                    elif (self.maze[self.states[state][0][0], self.states[state][0][1]] == 2) and self.is_key_retreived: # We are at the exit state, without meeting the minotaur
                        states.append('Win')
                
                    else:  # The player remains in place, the minotaur moves randomly
                        states.append(((self.states[state][0][0], self.states[state][0][1]), (rows_minotaur[i], cols_minotaur[i])))
                
                return states # -- NOTE: the last one is always that will aproach the player --
          
            else: # The action is possible, the player and the minotaur both move
                states = []
                for i in range(len(rows_minotaur)):
                
                    if (rows_minotaur[i] == row_player and cols_minotaur[i] == col_player): # We met the minotaur
                        states.append('Eaten')
                    
                    elif (self.maze[row_player, col_player] == 2) and self.is_key_retreived: # We are at the exit state, without meeting the minotaur
                        states.append('Win')
                    
                    else: # The player moves, the minotaur moves randomly
                        states.append(((row_player, col_player), (rows_minotaur[i], cols_minotaur[i])))
              
                return states # -- NOTE: the last one is always that will aproach the player --
        
        
        

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions)
        transition_probabilities = np.zeros(dimensions)
        print(transition_probabilities.shape)
        # TODO: Compute the transition probabilities.
        for i in range(self.n_states):
            s = self.states[i]
            
            if isinstance(s, str): # -- either win or eaten, keep being in that state --
                transition_probabilities[i, i, :] = 1
                continue
            
            # -- agent position at source state --
            i_a, j_a = s[0]
            # -- minotaur position at source state --
            i_m, j_m = s[1]
            
            # # -- check if win --
            # if self.maze[i_a, j_a] == 2:
            #     transition_probabilities[i, self.map['Win'], :] = 1 # -- the only possible next state is win --
                
            # # -- check eaten --
            # if i_a == i_m and j_a == j_m:
            #     transition_probabilities[i, self.map['Eaten'], :] = 1 # -- the only possible next state is eaten --
                
            for k in range(self.n_actions):
                a = self.actions[k]
                
                # -- the probability of state transition is uniform over the possible next states --
                possible_next_states = self.__move(i, k)
                
                    
                
                is_win_included = False
                for s_prime in possible_next_states:
                    if s_prime == 'Win':
                        is_win_included = True
                        break
                
                if is_win_included and len(possible_next_states) == 1:
                    uniform_prob = 1.0
                else:
                    uniform_prob = 1.0/(len(possible_next_states)-1) if is_win_included else 1.0/len(possible_next_states)
                for s_prime in possible_next_states:
                    j = self.map[s_prime]
                    if s_prime == 'Win':
                        transition_probabilities[i, j, k] = 1.0
                    else:
                        transition_probabilities[i, j, k] = uniform_prob
                        
                # if self.states[i][0][0] == 6 and self.states[i][0][1] == 6 and self.states[i][1][0] == 3 and self.states[i][1][1] == 3:
                #     # print(uniform_prob)
                #     print(self.actions_names[k], possible_next_states)
                #     # print(transition_probabilities.shape)
                #     print('transition_probabilities', transition_probabilities[i, j, k])
        # exit()
        return transition_probabilities



    def __rewards(self):
        
        """ Computes the rewards for every state action pair """

        rewards = np.zeros((self.n_states, self.n_actions))
        
        for s in range(self.n_states):
            for a in range(self.n_actions):
                
                if self.states[s] == 'Eaten': # The player has been eaten
                    rewards[s, a] = self.MINOTAUR_REWARD
                
                elif self.states[s] == 'Win': # The player has won
                    rewards[s, a] = self.GOAL_REWARD
                
                else:                
                    next_states = self.__move(s,a)
                    next_s = next_states[0] # The reward does not depend on the next position of the minotaur, we just consider the first one
                    
                    if self.states[s][0] == next_s[0] and a != self.STAY: # The player hits a wall
                        rewards[s, a] = self.IMPOSSIBLE_REWARD
                    
                    else: # Regular move
                        rewards[s, a] = self.STEP_REWARD

        return rewards




    def simulate(self, start, policy):
        self.reset()
        path = list()
        
        t = 1 # Initialize current state, next state and time
        s = self.map[start]
        path.append(start) # Add the starting position in the maze to the path
        next_states = self.__move(s, policy[s, 1 if self.is_key_retreived else 0]) # Move to next state given the policy and the current state
        next_s = next_states[np.random.randint(0, len(next_states))] # TODO
        path.append(next_s) # Add the next state to the path
        
        horizon =  1e2                             # Question e
        # Loop while state is not the goal state
        while s != next_s and t <= horizon:
            s = self.map[next_s] # Update state
            if np.random.rand() < 0.0:
                act = np.random.randint(0, self.n_actions)
            else:
                act =  policy[s, 1 if self.is_key_retreived else 0]
            next_states = self.__move(s,act) # Move to next state given the policy and the current state
            # -- 0.35 chance of moving towards the player --
            if not len(next_states) == 1:
                if np.random.rand() <= 0.35:
                    next_s = next_states[-1]
                else:
                    next_s = next_states[np.random.randint(0, len(next_states)-1)]
            else:
                next_s = next_states[0]
            path.append(next_s) # Add the next state to the path
            
            if self.states[s] != 'Win' and self.states[s] != 'Eaten' and self.is_key_retreived and (not self.is_key_retreived) and self.maze[self.states[s][0]] == 3:
                # reward += self.KEY_REWARD
                self.is_key_retreived = True
            t += 1 # Update time for next iteration
        
        return [path, horizon] # Return the horizon as well, to plot the histograms for the VI


    def reset(self):
        self.state_of_key = 0
        self.is_key_retreived = False

    def online_learning(self, start, init_q):
        # -- prepare log directory --
        import os
        import datetime
        subdir = datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        log_dir = './logs/{}'.format(subdir)
        frame_dir = os.path.join(log_dir, 'frames')
        os.makedirs(frame_dir, exist_ok=True)

        # -- start online learning --
        MAX_NUM_EPOSIDES = 50000
        # MAX_NUM_EPOSIDES = 101
        EPSILON = 0.2
        INIT_EPSILON = 0.5
        LAMBDA = 0.9
        SHOW_PERIOD = MAX_NUM_EPOSIDES // 15 # we show 15 times
        horizon = 2e2
        # AVAILABLE_SPAWN_POINTS = [[0, 0], [0, 4], [0, 5], [0, 6], [4, 4], [4, 2], [1, 4], [1, 3], [2, 3]]
        AVAILABLE_SPAWN_POINTS = [[0, 0]]
        MINOTAUR_AVAILABLE_SPAWN_POINTS = [[6, 5], [4, 5], [2, 6], [5, 6], [3, 5]]
        # AVAILABLE_SPAWN_POINTS = [[0, 6]]
        
        # -- initialize q-function --
        q_func = np.zeros((self.n_states, 2, self.n_actions))
        q_func[...,0] = - (1 - LAMBDA)
        q_func[...,1:] = (1 - LAMBDA)
        # q_func += np.random.rand(self.n_states, 2, self.n_actions) * 0.001
        state_visits = np.zeros((self.n_states, 2, self.n_actions))
        success_rates = np.zeros(MAX_NUM_EPOSIDES)
        best_success_rate_info = {'value': -1e10, 'episode': 0}
        best_key_rate_info = {'value': -1e10, 'episode': 0}
        success_count = 0
        key_retrieve_count = 0
        total_count = 1
        
        for episode in range(MAX_NUM_EPOSIDES):
            self.reset()
            # eps_e = np.max([INIT_EPSILON - episode*0.00001, EPSILON])
            eps_e = EPSILON
            if (episode+1) % (MAX_NUM_EPOSIDES // 100) == 0:
                success_rates[episode] = success_count / float(total_count)
                if success_rates[episode] >= best_success_rate_info['value']:
                    best_success_rate_info['value'] = success_rates[episode]
                    best_success_rate_info['episode'] = episode
                    np.save(os.path.join(log_dir, 'q_function_best.npy'), q_func)
                
                key_rate = key_retrieve_count / float(total_count)
                if key_rate >= best_key_rate_info['value']:
                    best_key_rate_info['value'] = key_rate
                    best_key_rate_info['episode'] = episode
                    
                success_count = 0
                key_retrieve_count = 0
                total_count = 0
                
                print("{}/{}".format(episode, MAX_NUM_EPOSIDES))
                print("     Current success rate {}, key rate {}".format(success_rates[episode], key_rate))
                print("     Best success rate {} at {}".format(best_success_rate_info['value'], best_success_rate_info['episode']))
                print("     Best key rate {} at {}".format(best_key_rate_info['value'], best_key_rate_info['episode']))
            
            if (episode+1) % SHOW_PERIOD == 0:
                num_non_visited_states = np.sum(np.where(state_visits < 1, 1, 0))
                print("Percentage of non-visited state-action pair: {:.5f}%".format(num_non_visited_states / float(self.n_actions * self.n_states * 2) * 100))
                num_non_visited_states_key_retreived = np.sum(np.where(state_visits[:,1,:] < 1, 1, 0))
                print("Percentage of non-visited state-action pair, given key retrieved: {:.5f}%".format(num_non_visited_states_key_retreived / float(self.n_actions * self.n_states) * 100))
                num_non_visited_states_key_not_retreived = np.sum(np.where(state_visits[:,0,:] < 1, 1, 0))
                print("Percentage of non-visited state-action pair, given key not retrieved: {:.5f}%".format(num_non_visited_states_key_not_retreived / float(self.n_actions * self.n_states) * 100))
                
                np.save(os.path.join(log_dir, 'q_function_{:06d}.npy'.format(episode)), q_func)
                os.makedirs(os.path.join(frame_dir, "episode_{:06d}".format(episode)), exist_ok=True)
                print("Episode {}/{}".format(episode+1, MAX_NUM_EPOSIDES))
                
                # -- prepare animation --
                # Map a color to each cell in the maze
                col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -1: LIGHT_RED, -2: LIGHT_PURPLE, 3: BLUE}
                
                rows, cols = maze.shape # Size of the maze
                fig = plt.figure(1, figsize=(cols, rows)) # Create figure of the size of the maze

                # Remove the axis ticks and add title
                ax = plt.gca()
                ax.set_title('Policy simulation')
                ax.set_xticks([])
                ax.set_yticks([])

                # Give a color to each cell
                colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

                # Create a table to color
                grid = plt.table(
                    cellText = None, 
                    cellColours = colored_maze, 
                    cellLoc = 'center', 
                    loc = (0,0), 
                    edges = 'closed'
                )
                
                # Modify the height and width of the cells in the table
                tc = grid.properties()['children']
                for cell in tc:
                    cell.set_height(1.0/rows)
                    cell.set_width(1.0/cols)
            
            path = list()
            data = list()
            
            t = 1 # Initialize current state, next state and time
            
            # -- TODO: How about random starting point? --
            start = (tuple(AVAILABLE_SPAWN_POINTS[np.random.randint(0, len(AVAILABLE_SPAWN_POINTS))]), start[1])
            s = self.map[start]
            path.append(start) # Add the starting position in the maze to the path
            next_states = self.__move(s, 0) # Move to next state given the policy and the current state
            next_s = next_states[np.random.randint(0, len(next_states))] # TODO
            path.append(next_s) # Add the next state to the path

            # Loop while state is not the goal state
            while s != next_s and t <= horizon:
                # -- update policy and get the next movement --
                

                if np.random.rand() <= eps_e:
                    # print("Explore")
                    act = np.random.randint(0, self.n_actions)
                else:
                    # -- greedy action --
                    # print("Greedy")
                    act = np.argmax(q_func[s, 1 if self.is_key_retreived else 0])
                # print("Best act at {} is {}, q value is {}".format(s, self.actions_names[act], q_func[s, 1 if self.is_key_retreived else 0, act]))
                
                # -- step the environment and get observations --
                s = self.map[next_s] # Update state
                next_states = self.__move(s, act) # Move to next state given the policy and the current state
                # -- 0.35 chance of moving towards the player --
                if not len(next_states) == 1:
                    if np.random.rand() <= 0.35:
                        next_s = next_states[-1]
                    else:
                        next_s = next_states[np.random.randint(0, len(next_states)-1)]
                else:
                    next_s = next_states[0]

                # -- for simplicity, first fix minotaur --
                # if next_s != 'Eaten' and next_s != 'Win':
                #     next_s = (next_s[0], (6, 1))
                
                path.append(next_s) # Add the next state to the path
                reward = self.rewards[s, act]
                
                should_update_retreive_state = False
                if next_s != 'Win' and next_s != 'Eaten' and (not self.is_key_retreived) and self.maze[next_s[0]] == 3:
                    reward += self.KEY_REWARD
                    key_retrieve_count += 1
                    should_update_retreive_state = True
                data.append([s, act, reward, self.map[next_s]])
                
                # -- update q-function --
                # print("updated state {}, reward was {}, action was {}".format(self.states[s], reward, self.actions_names[act]))
                state_visits[s, 1 if self.is_key_retreived else 0, act] += 1
                alpha_k = 1.0 / (state_visits[s, 1 if self.is_key_retreived else 0, act] ** 2/3.0)
                # print(q_func[s, 1, :])
                q_func[s, 1 if self.is_key_retreived else 0, act] = q_func[s, 1 if self.is_key_retreived else 0, act] + alpha_k * (reward + LAMBDA * np.max(q_func[self.map[next_s], 1 if self.is_key_retreived else 0]) - q_func[s, 1 if self.is_key_retreived else 0, act])
                # print(q_func[s, 1, :])
                # print(q_func[self.map[((0,6),(3,1))], 0, :])
                # print(q_func[self.map[((0,7),(3,1))], 0, :])
                
                if should_update_retreive_state:
                    self.is_key_retreived = True
                    
                # -- perform animation --
                if (episode+1) % SHOW_PERIOD == 0:
                    if len(path) >= 2 and path[t-2] != 'Eaten' and path[t-2] != 'Win':
                        grid.get_celld()[(path[t-2][0])].set_facecolor(col_map[maze[path[t-2][0]]])
                        grid.get_celld()[(path[t-2][1])].set_facecolor(col_map[maze[path[t-2][1]]])
                        
                    if path[t-1] != 'Eaten' and path[t-1] != 'Win':
                        grid.get_celld()[(path[t-1][0])].set_facecolor(col_map[-2]) # Position of the player
                        grid.get_celld()[(path[t-1][1])].set_facecolor(col_map[-1]) # Position of the minotaur
                        
                        if self.is_key_retreived:
                            grid.get_celld()[((0, 7))].set_facecolor(KEY_GOT) # Position of the player
                        
                    if path[t-1] == 'Eaten':
                        print("You are eaten!")
                        break
                    elif path[t-1] == 'Win':
                        print("You escaped!")
                        break

                    fig.canvas.draw()
                    img_plot = np.array(fig.canvas.renderer.buffer_rgba())
                    frame = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
                    # cv2.imwrite(os.path.join(frame_dir, 'episode_{:06d}/frame_{:06d}.png'.format(episode, t-1)), frame)
                    cv2.imshow('Image', frame)
                    cv2.waitKey(50)
            
                t += 1 # Update time for next iteration
            total_count += 1
            if self.states[s] == 'Win' or next_s == 'Win':
                    success_count += 1
                    
            # print(q_func[self.map[((0,6),(3,1))], 0, :])
            # print(q_func[self.map[((0,7),(3,1))], 0, :])
            # print()
            # exit()
        return q_func
    
    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)



def dynamic_programming(env: Maze, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    #TODO
    u = np.zeros([env.n_states, horizon])
    policy = np.zeros([env.n_states, horizon])
    
    for s in range(env.n_states):
        max_r = -1e20
        best_a = 0
        
        for a in range(env.n_actions):
            r = env.rewards[s, a]
            # print(env.actions_names[a], env.states[s], r)
            if max_r < r:
                max_r = r
                best_a = a
                
        u[s, -1] = max_r
        policy[s, -1] = best_a
        
    # for i in range(env.n_states):
    #     print(policy[i,-1])
    # exit()
    
    LOG = False
        
    for t in range(horizon-2, -1, -1):
        for s in range(env.n_states):
            if LOG and env.states[s][0][0] == 6 and env.states[s][0][1] == 6 and env.states[s][1][0] == 3 and env.states[s][1][1] == 3:
                print("BEFORE, ", u[s, t], env.states[s], env.actions_names[best_a])
            max_u = -1e20
            best_a = 0
            
            for a in range(env.n_actions):
                if LOG and env.states[s][0][0] == 6 and env.states[s][0][1] == 6 and env.states[s][1][0] == 3 and env.states[s][1][1] == 3:
                    print("action ", env.actions_names[a], env.rewards[s, a], env.transition_probabilities[s, :, a] @ u[:, t+1])
                    print(env.transition_probabilities[s, :, a])
                candidate_u = env.rewards[s, a] + env.transition_probabilities[s, :, a] @ u[:, t+1]
                
                if max_u < candidate_u:
                    max_u = candidate_u
                    best_a = a
                    
            u[s, t] = max_u
            policy[s, t] = best_a
            if LOG and env.states[s][0][0] == 6 and env.states[s][0][1] == 6 and env.states[s][1][0] == 3 and env.states[s][1][1] == 3:
                print("After, ", u[s, t], env.states[s], env.actions_names[best_a])
        # exit()
    V = u[:,0]

    return V, policy

def value_iteration(env: Maze, gamma, epsilon):
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
    #TODO

    V = np.zeros(env.n_states)
    policy = np.zeros(env.n_states)
    threshold = epsilon * (1 - gamma) / gamma
    
    should_not_stop = True
    while should_not_stop:
        V_prime = np.zeros_like(V)
        for s in range(env.n_states):
            best_new_v = -1e10
            best_a = 0
            for a in range(env.n_actions):
                v_here = env.rewards[s, a] + gamma * env.transition_probabilities[s, :, a] @ V
                if v_here > best_new_v:
                    best_new_v = v_here
                    best_a = a
            V_prime[s] = best_new_v
            policy[s] = best_a
            
        delta = np.linalg.norm((V_prime-V))
        print("Delta", delta)
        if delta < threshold:
            should_not_stop = False
            
        V = V_prime

    return V, policy

def animate_solution(maze, path):
    import os
    import datetime
    subdir = datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    frame_dir = './frames/{}'.format(subdir)
    os.makedirs(frame_dir, exist_ok=True)

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -1: LIGHT_RED, -2: LIGHT_PURPLE, 3: BLUE}
    
    rows, cols = maze.shape # Size of the maze
    fig = plt.figure(1, figsize=(cols, rows)) # Create figure of the size of the maze

    # Remove the axis ticks and add title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create a table to color
    grid = plt.table(
        cellText = None, 
        cellColours = colored_maze, 
        cellLoc = 'center', 
        loc = (0,0), 
        edges = 'closed'
    )
    
    # Modify the height and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    for i in range(0, len(path)):
        if path[i-1] != 'Eaten' and path[i-1] != 'Win':
            grid.get_celld()[(path[i-1][0])].set_facecolor(col_map[maze[path[i-1][0]]])
            grid.get_celld()[(path[i-1][1])].set_facecolor(col_map[maze[path[i-1][1]]])
        if path[i] != 'Eaten' and path[i] != 'Win':
            grid.get_celld()[(path[i][0])].set_facecolor(col_map[-2]) # Position of the player
            grid.get_celld()[(path[i][1])].set_facecolor(col_map[-1]) # Position of the minotaur
            
        if path[i] == 'Eaten':
            print("You are eaten!")
            break
        elif path[i] == 'Win':
            print("You escaped!")
            break

        fig.canvas.draw()
        img_plot = np.array(fig.canvas.renderer.buffer_rgba())
        frame = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
        cv2.imwrite(os.path.join(frame_dir, 'frame_{:3d}.png'.format(i)), frame)
        cv2.imshow('Image', frame)
        cv2.waitKey(100)


mode = args.mode
if __name__ == "__main__":
    # Description of the maze as a numpy array
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 3],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]])
    # With the convention 0 = empty cell, 1 = obstacle, 2 = exit of the Maze
    # np.random.seed(20010427)
    env = Maze(maze) # Create an environment maze
    horizon = 50       # TODO: Finite horizon

    # -- dummy random policy, for debugging --
    policy = np.zeros((env.n_states, 2), dtype=int)
    policy += np.random.randint(0, 4, policy.shape)

    start  = ((0,0), (6,5))
    if mode == 'train':
        # Simulate the shortest path starting from position A
        q_func = env.online_learning(start, None)
        np.save('q_function.npy', q_func)
    else:
        ckpt = '/home/kei/code/el2805/lab1-1/logs/2024-12-02 15:07:47/q_function_best.npy'
        q_func = np.load(ckpt)
        for n in range(env.n_states):
            for i in range(2):
                policy[n, i] = np.argmax(q_func[n, i])
        
        num_play = 10
        for k in range(num_play):
            path, h = env.simulate(start, policy)
            animate_solution(maze, path)
    
    