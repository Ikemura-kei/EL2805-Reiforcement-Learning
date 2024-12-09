import numpy as np
from modules.maze import *
import tqdm
import os
import datetime
import shutil
import json


def q_learning(maze_env: MazeEnv, lambda_value: float = 0.98, max_eposide: int = 50000, eps_init: float = 0.2, visualize=False, alpha=2.0/3):
    start_pnt = (0, 0, 0, 6, 4)

    # -- initialize q-function --
    q_func_init_traj = np.zeros((max_eposide, maze_env.n_actions))
    q_func = np.zeros((maze_env.n_states, maze_env.n_actions))
    q_func += (1 - lambda_value)
    q_func[..., maze_env.AGENT_ACTION_STAY] = - (1 - lambda_value)

    # -- tracks number of visits per state --
    state_visits = np.zeros((maze_env.n_states, maze_env.n_actions))

    # -- we visualize the policy per some iterations --
    show_period = max_eposide // 100

    # -- track the best performances --
    best_win_rate = 0
    best_win_rate_episode = 0
    win_cnt = 0
    win_rate = 0
    win_rate_list = []
    with tqdm.tqdm(total=max_eposide) as pbar:
        # -- the main training loop --
        for episode in range(max_eposide):
            start = start_pnt
            s_idx = maze_env.states_to_idx[start]

            if ((episode+1) % show_period) == 0:
                win_rate = win_cnt / show_period
                win_rate_list.append((episode, win_rate))

                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_win_rate_episode = episode

                # print('At episode {}, current win rate {:.2f}%, best win rate {:.2f}% occurred at episode {}'.
                #       format(episode, win_rate * 100, best_win_rate * 100, best_win_rate_episode))
                win_cnt = 0

            # -- run an episode --
            t = 0
            win = False
            while True:
                if np.random.rand() <= eps_init:
                    # -- exploration --
                    a_idx = np.random.randint(0, maze_env.n_actions)
                else:
                    # -- greedy action --
                    a_idx = np.argmax(q_func[s_idx])

                s_prime_idx, reward, terminated, truncated, info = maze_env.step(
                    s_idx, a_idx, t, do_render=((((episode+1) % show_period) == 0) and visualize), show=visualize)

                # -- q function update --
                state_visits[s_idx, a_idx] += 1  # increment state visit
                # update the learning rate
                alpha_k = 1.0 / (state_visits[s_idx, a_idx] ** alpha)
                q_func[s_idx, a_idx] = q_func[s_idx, a_idx] + alpha_k * \
                    (reward + lambda_value *
                     np.max(q_func[s_prime_idx]) - q_func[s_idx, a_idx])

                if maze_env.idx_to_states[s_idx] == 'win':
                    win_cnt += 1
                    win = True

                if terminated:
                    break
                elif truncated:
                    break

                # -- write back current state --
                s_idx = s_prime_idx
                t += 1

            # -- record the q value at initial state --
            q_func_init_traj[episode] = q_func[maze_env.states_to_idx[start]]

            # -- update progress bar --
            pbar.update(1)
            pbar.set_description("Episode win?: {}, current win rate {:.2f}%, best win rate {:.2f}%".format(
                1 if win else 0, win_rate * 100, best_win_rate * 100))

    return q_func, q_func_init_traj, best_win_rate, win_rate_list


def sarsa(maze_env: MazeEnv, lambda_value: float = 0.98, max_eposide: int = 50000, eps_init: float = 0.2, visualize=False, alpha=2.0/3, delta=0.5):
    start_pnt = (0, 0, 0, 6, 4)

    # -- initialize q-function --
    q_func_init_traj = np.zeros((max_eposide, maze_env.n_actions))
    q_func = np.zeros((maze_env.n_states, maze_env.n_actions))
    q_func += (1 - lambda_value)
    q_func[..., maze_env.AGENT_ACTION_STAY] = - (1 - lambda_value)

    # -- tracks number of visits per state --
    state_visits = np.zeros((maze_env.n_states, maze_env.n_actions))

    # -- we visualize the policy per some iterations --
    show_period = max_eposide // 100

    # -- track the best performances --
    best_win_rate = 0
    best_win_rate_episode = 0
    win_cnt = 0
    win_rate = 0
    win_rate_list = []
    with tqdm.tqdm(total=max_eposide) as pbar:
        # -- the main training loop --
        for episode in range(max_eposide):
            eps = eps_init / ((episode+1)**delta) if delta > 0 else eps_init
            
            start = start_pnt
            s_idx = maze_env.states_to_idx[start]

            if ((episode+1) % show_period) == 0:
                win_rate = win_cnt / show_period
                win_rate_list.append((episode, win_rate))

                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_win_rate_episode = episode

                # print('At episode {}, current win rate {:.2f}%, best win rate {:.2f}% occurred at episode {}'.
                #       format(episode, win_rate * 100, best_win_rate * 100, best_win_rate_episode))
                win_cnt = 0

            # -- run an episode --
            t = 0
            win = False

            # -- choose a default previous action --
            if np.random.rand() <= eps:
                # -- exploration --
                a_idx = np.random.randint(0, maze_env.n_actions)
            else:
                # -- greedy action --
                a_idx = np.argmax(q_func[s_idx])

            while True:
                s_prime_idx, reward, terminated, truncated, info = maze_env.step(
                    s_idx, a_idx, t, do_render=((((episode+1) % show_period) == 0) and visualize), show=visualize)

                # -- q function update --
                state_visits[s_idx, a_idx] += 1  # increment state visit
                
                # -- choose the next action --
                if np.random.rand() <= eps:
                    # -- exploration --
                    a_prime_idx = np.random.randint(0, maze_env.n_actions)
                else:
                    # -- greedy action --
                    a_prime_idx = np.argmax(q_func[s_prime_idx])

                # -- update the learning rate --
                alpha_k = 1.0 / (state_visits[s_idx, a_idx] ** alpha)
                q_func[s_idx, a_idx] = q_func[s_idx, a_idx] + alpha_k * \
                    (reward + lambda_value *
                     q_func[s_prime_idx, a_prime_idx] - q_func[s_idx, a_idx])

                if maze_env.idx_to_states[s_idx] == 'win':
                    win_cnt += 1
                    win = True

                if terminated:
                    break
                elif truncated:
                    break

                # -- write back current state --
                s_idx = s_prime_idx
                a_idx = a_prime_idx
                t += 1

            # -- record the q value at initial state --
            q_func_init_traj[episode] = q_func[maze_env.states_to_idx[start]]

            # -- update progress bar --
            pbar.update(1)
            pbar.set_description("Episode win?: {}, current win rate {:.2f}%, best win rate {:.2f}%, eps {:.5f}".format(
                1 if win else 0, win_rate * 100, best_win_rate * 100, eps))

    return q_func, q_func_init_traj, best_win_rate, win_rate_list


if __name__ == "__main__":
    # -- create log dir --
    out_dir = './outputs/{}'.format(
        datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
    os.makedirs(out_dir, exist_ok=True)

    # -- save a copy of this code --
    shutil.copyfile(os.path.join(os.path.dirname(os.path.abspath(
        __file__)), 'main.py'), os.path.join(out_dir, 'main.py'))

    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 3],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]])

    fix_minotaur = False
    horizon = 5000
    maze_env = MazeEnv(maze, episode_len=horizon,
                       minotaur_fix=fix_minotaur, beta=0.65)
    algorithm = 'sarsa'  # 'sarsa' / 'q_learning'
    lambda_value = 0.98
    max_eposide = 50000
    eps_init = 0.2
    visualize = False
    alpha = 0.75
    delta = 0.6
    if algorithm == 'sarsa':
        q_func, q_func_init_traj, best_win_rate, win_rate_list = sarsa(maze_env, lambda_value, max_eposide,
                                                                       eps_init, visualize=visualize, alpha=alpha, delta=delta)
    elif algorithm == 'q_learning':
        q_func, q_func_init_traj, best_win_rate, win_rate_list = q_learning(maze_env, lambda_value, max_eposide,
                                                                            eps_init, visualize=visualize, alpha=alpha)
    else:
        raise NotImplementedError

    # -- save the q_function --
    np.save(os.path.join(out_dir, 'q_func.npy'), q_func)
    np.save(os.path.join(out_dir, 'q_func_init_traj.npy'), q_func_init_traj)
    np.save(os.path.join(out_dir, 'win_rate_list.npy'), np.array(win_rate_list))

    # -- save hyper-params --
    HYPER_PARAMS = {'fix_minotaur': fix_minotaur, 'horizon': horizon,
                    'lambda_value': lambda_value, 'max_eposide': max_eposide,
                    'eps_init': eps_init, 'visualize': visualize, 'best_win_rate': best_win_rate, 'alpha': alpha, 'algorithm': algorithm, 'delta': delta}
    with open(os.path.join(out_dir, 'params.json'), 'w') as file:
        json.dump(HYPER_PARAMS, file, indent=6)
