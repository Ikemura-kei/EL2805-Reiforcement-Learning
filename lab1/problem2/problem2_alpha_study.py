# Copyright [2024] [KTH Royal Institute of Technology] 
# Licensed under the Educational Community License, Version 2.0 (ECL-2.0)
# This file is part of the Computer Lab 1 for EL2805 - Reinforcement Learning.

# Load packages
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import cv2

import json
import datetime
import os
import pickle    
import copy
import shutil

out_dir = './outputs/{}'.format(datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
os.makedirs(out_dir, exist_ok=True)
# -- save a copy of this code --
shutil.copyfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'problem2.py'), os.path.join(out_dir, 'problem2.py'))

# -- set random seed --
seed=42
np.random.seed(seed)

# -- create environment --
# render_mode = None
# render_mode='human'
render_mode = 'rgb_array'
env = gym.make('MountainCar-v0', render_mode=render_mode)
env.reset(seed=seed)

# -- helper functions --
def running_average(x, N):
    ''' Function used to compute the running mean
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    ''' Rescaling of s to the box [0,1]^2 '''
    x = (s - low) / (high - low)
    return x

def get_phi(state, eta):
    # state: (STATE_DIM, )
    # eta: (N_BASIS, STATE_DIM)
    # return: (N_BASIS, )
    
    eta_s = eta @ state
    return np.cos(eta_s * np.pi)

def epsilon_greedy_policy(epsilon, q_s):
    rand_num = np.random.rand()
    if rand_num <= epsilon:
        return np.random.randint(0, N_ACTIONS)
    else:
        return np.argmax(q_s)

def evaluate(env, eval_episodes, eta, weights):
    low, high = env.observation_space.low, env.observation_space.high
    
    def scale_state_varibles(s, eta, low=env.observation_space.low, high=env.observation_space.high):
        ''' Rescaling of s to the box [0,1]^2
            and features transformation
        '''
        x = (s-low) / (high-low)
        return np.cos(np.pi * np.dot(eta, x))
    
    def Qvalues(s, w):
        ''' Q Value computation '''
        return np.dot(w, s)
    
    episode_reward_list = []
    for i in range(eval_episodes):
        # Reset enviroment data
        done = False
        truncated = False
        state = scale_state_varibles(env.reset()[0], eta, low, high)
        total_episode_reward = 0.

        qvalues = Qvalues(state, weights)
        action = np.argmax(qvalues)

        while not (done or truncated):
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = scale_state_varibles(next_state, eta, low, high)
            qvalues_next = Qvalues(next_state, weights)
            next_action = np.argmax(qvalues_next)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            qvalues = qvalues_next
            action = next_action

        # Append episode reward
        episode_reward_list.append(total_episode_reward)

        # Close environment
        env.close()
        
    avg_reward = np.mean(episode_reward_list)
    confidence = np.std(episode_reward_list)
    return avg_reward, confidence
    
def train(env, episode_reward_list, eta, weights, v, alpha_scaling=None, alpha=2e-3, show=False):
    low, high = env.observation_space.low, env.observation_space.high
    
    # -- reset enviroment data --
    done = False
    truncated = False
    state = scale_state_variables(env.reset()[0])
    total_episode_reward = 0.
    
    # -- update epsilon for this episode, if needed --
    epsilon = HYPER_PARAM_EPSILON
    
    # -- initialize the eligibility trace --
    z = np.zeros_like(weights)
    
    t = 0
    while not (done or truncated):
        # -- update the learning rate, if needed --
        alpha_ = alpha
        if HYPER_PARAM_REDUCE_LR:
            alpha_ *= alpha_scaling
            
        # -- compute the state-action value function for s --
        q_s = np.zeros((N_ACTIONS))
        phi_s = get_phi(state, eta) # (N_BASIS)
        q_s = weights @ phi_s # (N_ACTIONS)
        action = epsilon_greedy_policy(epsilon, q_s)
        q_s_a = q_s[action]
            
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise. Truncated is true if you reach 
        # the maximal number of time steps, False else.
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = scale_state_variables(next_state)
        
        # -- compute the state-action value function for s' --
        q_s_prime = np.zeros((N_ACTIONS))
        q_s_prime = weights @ get_phi(next_state, eta) # (N_ACTIONS)
        next_action = epsilon_greedy_policy(epsilon, q_s_prime)
        q_s_prime_a_prime = q_s_prime[next_action]
        
        # -- compute delta, the TD error --
        delta_t = reward + HYPER_PARAM_GAMMA * q_s_prime_a_prime - q_s_a

        # -- update the eligibility trace --
        z = HYPER_PARAM_GAMMA * HYPER_PARAM_LAMBDA * z
        z[action] += phi_s # gradient of Q w.r.t w is phi_s
        z = np.clip(z, HYPER_PARAM_Z_CLIP_THRESHOLDS[0], HYPER_PARAM_Z_CLIP_THRESHOLDS[1])

        # -- update the weight parameter --
        if HYPER_PARAM_MOMENTUM <= 0:
            # -- Vanilla SGD --
            weights = weights + alpha_ * delta_t * z
        else:
            v = HYPER_PARAM_MOMENTUM * v + alpha_ * delta_t * z
            # -- SGD with Nesterov Acceleration / SGD with Momentum --
            weights = weights + ((HYPER_PARAM_MOMENTUM * v + alpha_ * delta_t * z) if HYPER_PARAM_NESTROV else v)
            
        # Update episode reward
        total_episode_reward += reward
            
        # Update state for next iteration
        state = next_state
        t += 1
        
        # -- render --
        if (i % VISUALIZATION_PERIOD) == 0 and show:
            if t == 1:
                cv2.destroyAllWindows()
            frame = env.render()
            cv2.imshow('Episode_{}'.format(i), frame)
            cv2.waitKey(1)

    # Append episode reward
    episode_reward_list.append(total_episode_reward)

    # Close environment
    env.close()
    cv2.destroyAllWindows()
    
    return episode_reward_list, weights

# -- hyper-params --
ETA = np.array([[1, 1],
                [1, 0],
                [0, 1], ])
N_ACTIONS = env.action_space.n
N_BASIS = ETA.shape[0]
STATE_DIM = 2
HYPER_PARAM_TRAIN_EPISODE = 1000        # Number of episodes to run for training
VISUALIZATION_PERIOD = HYPER_PARAM_TRAIN_EPISODE // 5
NUM_EVAL_EPISODES = 50
EVAL_PERIOD = HYPER_PARAM_TRAIN_EPISODE // 20
HYPER_PARAM_ALPHA = np.linspace(5e-4, 1e-2, 100)
HYPER_PARAM_EPSILON = 0.2
HYPER_PARAM_LAMBDA = 0.99
HYPER_PARAM_GAMMA = 1
HYPER_PARAM_Z_CLIP_THRESHOLDS = [-2, 2]
HYPER_PARAM_MOMENTUM = 0.9
HYPER_PARAM_NESTROV = True
HYPER_PARAM_EXPECTED_GOOD_PERFORMANCE = -135
HYPER_PARAM_REDUCE_LR = False
HYPER_PARAM_LR_SCALER = 0.7

# -- parameters --
weights = np.zeros((N_ACTIONS, N_BASIS)) + np.random.normal(0, 0.1, (N_ACTIONS, N_BASIS))
if HYPER_PARAM_MOMENTUM:
    v = np.zeros_like(weights)
else:
    v = None
alpha_scaling = 1.0

best_perfs = []
for alpha in HYPER_PARAM_ALPHA:
    print("alpha: ", alpha)
    # -- the main training loop --
    best_avg_reward = -1e10
    best_conf = 0
    best_weights = None
    best_episode = 0
    episode_reward_list = []  # Used to save episodes reward
    eval_results = []
    
    for i in range(HYPER_PARAM_TRAIN_EPISODE):
        if (((i+1) % EVAL_PERIOD) == 0) or (i == (HYPER_PARAM_TRAIN_EPISODE-1)):
            avg_reward, confidence = evaluate(env, NUM_EVAL_EPISODES, ETA, weights)
            eval_results.append([i, avg_reward])
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_conf = confidence
                best_weights = weights.copy()
                best_episode = i
                
                if avg_reward < HYPER_PARAM_EXPECTED_GOOD_PERFORMANCE:
                    alpha_scaling = HYPER_PARAM_LR_SCALER
                print("A new best policy found at episode {}, average reward was {}".format(best_episode+1, best_avg_reward))
        
        episode_reward_list, weights = train(env, episode_reward_list, ETA, weights, v, alpha_scaling, alpha=alpha)    
    
    print([alpha, best_avg_reward, best_conf])
    best_perfs.append([alpha, best_avg_reward, best_conf])
    
np.save('perf_vs_alpha.npy', best_perfs)

# # -- save the best policy as well as the hyper-parameters --
# data = {'W': best_weights, 'N': ETA}

# # -- save policy for evaluation --
# path = 'weights.pkl'
# with open(path, "wb") as f:
#     pickle.dump(data, f)
    
# # -- save a backup policy --
# path = os.path.join(out_dir, 'weights.pkl')
# with open(path, "wb") as f:
#     pickle.dump(data, f)

# # -- save other logs as well as hyper-parameters --
# data['N'] = data['N'].tolist()
# data['W'] = data['W'].tolist()
# data.update({'seed': seed, 'best_reward': best_avg_reward, 'best_episode': best_episode})
# local_vars = locals()
# local_vars_keys = copy.deepcopy(list(local_vars.keys()))
# for k in local_vars_keys:
#     if 'HYPER_PARAM' in k:
#         data[k] = locals()[k]
        
# out_file = open(os.path.join(out_dir, 'result.json'), "w")
# json.dump(data, out_file, indent = 6)

# # -- plot rewards --
# plt.plot([i for i in range(1, HYPER_PARAM_TRAIN_EPISODE+1)], episode_reward_list, label='Episode reward')
# plt.plot([i for i in range(1, HYPER_PARAM_TRAIN_EPISODE+1)], running_average(episode_reward_list, 10), label='Average episode reward')
# plt.xlabel('Episodes')
# plt.ylabel('Total reward')
# plt.title('Total Reward vs Episodes')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.savefig(os.path.join(out_dir, 'episode_rewards.png'))
# plt.show()

# eval_results = np.array(eval_results)
# plt.plot(eval_results[:,0], eval_results[:,1], label='Evaluation results')
# plt.xlabel('Episodes')
# plt.ylabel('Average rewards')
# plt.title('Evaluation performances')
# plt.legend()
# plt.grid(alpha=0.3)
# plt.savefig(os.path.join(out_dir, 'evaluations.png'))
# plt.show()