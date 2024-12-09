import numpy as np
import matplotlib.pyplot as plt
import os

configs = {1: False, 2: False, 3: False, 4: True}

if configs[1]:
    # -- 1. show value function trajectory of the default setting on q-learning (question i.2) --
    analysis_dir = os.path.join('.', 'analysis', 'value_func_traj')
    os.makedirs(analysis_dir, exist_ok=True)

    run = '2024-12-09 12:25:08'

    q_func_init_traj = np.load(os.path.join('outputs', run, 'q_func_init_traj.npy'))

    v_func = np.max(q_func_init_traj, axis=-1) # max over actions

    plt.plot(range(len(v_func)), v_func, 'y')
    plt.title('Value function trajectory')
    plt.xlabel('Episodes')
    plt.ylabel('Value function')
    plt.savefig(os.path.join(analysis_dir, 'value_func_traj.png'))
    plt.show()
    
    win_rates = np.load(os.path.join('outputs', run, 'win_rate_list.npy'))
    plt.plot(win_rates[:, 0], win_rates[:, 1], 'y')
    plt.title('Winning rates')
    plt.xlabel('Episodes')
    plt.ylabel('Winning rate')
    plt.savefig(os.path.join(analysis_dir, 'win_rates.png'))
    plt.show()
    
if configs[2]:
    # -- 2. analyze the effect of alpha on q-learning (question i.3)
    analysis_dir = os.path.join('.', 'analysis', 'effect_of_alpha')
    os.makedirs(analysis_dir, exist_ok=True)

    run_alpha1 = '2024-12-09 12:27:23' # alpha = 1/3
    run_alpha2 = '2024-12-09 12:25:08' # alpha = 2/3
    run_alpha3 = '2024-12-09 12:29:04' # alpha = 1

    q_func_init_traj_alpha1 = np.load(os.path.join('outputs', run_alpha1, 'q_func_init_traj.npy'))
    q_func_init_traj_alpha2 = np.load(os.path.join('outputs', run_alpha2, 'q_func_init_traj.npy'))
    q_func_init_traj_alpha3 = np.load(os.path.join('outputs', run_alpha3, 'q_func_init_traj.npy'))

    v_func_alpha1 = np.max(q_func_init_traj_alpha1, axis=-1) # max over actions
    v_func_alpha2 = np.max(q_func_init_traj_alpha2, axis=-1) # max over actions
    v_func_alpha3 = np.max(q_func_init_traj_alpha3, axis=-1) # max over actions

    plt.plot(range(len(v_func_alpha1)), v_func_alpha1, 'r')
    plt.plot(range(len(v_func_alpha2)), v_func_alpha2, 'g')
    plt.plot(range(len(v_func_alpha3)), v_func_alpha3, 'b')
    plt.title('Value function trajectory')
    plt.xlabel('Episodes')
    plt.ylabel('Value function')
    plt.legend(['alpha = 1/3', 'alpha = 2/3', 'alpha = 1'])
    plt.savefig(os.path.join(analysis_dir, 'value_func_traj.png'))
    plt.show()

    win_rates_alpha1 = np.load(os.path.join('outputs', run_alpha1, 'win_rate_list.npy'))
    win_rates_alpha2 = np.load(os.path.join('outputs', run_alpha2, 'win_rate_list.npy'))
    win_rates_alpha3 = np.load(os.path.join('outputs', run_alpha3, 'win_rate_list.npy'))
    plt.plot(win_rates_alpha1[:, 0], win_rates_alpha1[:, 1], 'r')
    plt.plot(win_rates_alpha2[:, 0], win_rates_alpha2[:, 1], 'g')
    plt.plot(win_rates_alpha3[:, 0], win_rates_alpha3[:, 1], 'b')
    plt.title('Winning rates')
    plt.xlabel('Episodes')
    plt.ylabel('Winning rate')
    plt.legend(['alpha = 1/3', 'alpha = 2/3', 'alpha = 1'])
    plt.savefig(os.path.join(analysis_dir, 'win_rates.png'))
    plt.show()
    
if configs[3]:
    # -- 3. sarsa value function (at initial state) visualization (question j.2) --
    analysis_dir = os.path.join('.', 'analysis', 'value_func_traj_sarsa')
    os.makedirs(analysis_dir, exist_ok=True)

    run_epsilon1 = '2024-12-09 13:16:55' # epsilon = 0.1
    run_epsilon2 = '2024-12-09 13:21:25' # epsilon = 0.2
    
    q_func_init_traj_epsilon1 = np.load(os.path.join('outputs', run_epsilon1, 'q_func_init_traj.npy'))
    q_func_init_traj_epsilon2 = np.load(os.path.join('outputs', run_epsilon2, 'q_func_init_traj.npy'))

    v_func_epsilon1 = np.max(q_func_init_traj_epsilon1, axis=-1) # max over actions
    v_func_epsilon2 = np.max(q_func_init_traj_epsilon2, axis=-1) # max over actions

    plt.plot(range(len(v_func_epsilon1)), v_func_epsilon1, 'r')
    plt.plot(range(len(v_func_epsilon2)), v_func_epsilon2, 'g')
    plt.title('Value function trajectory')
    plt.xlabel('Episodes')
    plt.ylabel('Value function')
    plt.legend(['epsilon = 0.1', 'epsilon = 0.2'])
    plt.savefig(os.path.join(analysis_dir, 'value_func_traj.png'))
    plt.show()
    
    win_rates_epsilon1 = np.load(os.path.join('outputs', run_epsilon1, 'win_rate_list.npy'))
    win_rates_epsilon2 = np.load(os.path.join('outputs', run_epsilon2, 'win_rate_list.npy'))
    plt.plot(win_rates_epsilon1[:, 0], win_rates_epsilon1[:, 1], 'r')
    plt.plot(win_rates_epsilon2[:, 0], win_rates_epsilon2[:, 1], 'g')
    plt.title('Winning rates')
    plt.xlabel('Episodes')
    plt.ylabel('Winning rate')
    plt.legend(['epsilon = 0.1', 'epsilon = 0.2'])
    plt.savefig(os.path.join(analysis_dir, 'win_rates.png'))
    plt.show()
    
if configs[4]:
    # -- compare epsilon decay on sarsa (question j.3) --
    analysis_dir = os.path.join('.', 'analysis', 'epsilon_decay_sarsa')
    os.makedirs(analysis_dir, exist_ok=True)

    run_epsilon0 = '2024-12-09 13:16:55' # no decay, alpha=2/3, eps_0=0.1
    run_epsilon1 = '2024-12-09 13:28:47' # delta=0.6, alpha=2/3, eps_0=0.2
    run_epsilon2 = '2024-12-09 13:31:22' # delta=0.85, alpha=2/3, eps_0=0.2
    run_epsilon3 = '2024-12-09 13:41:57' # delta=0.6, alpha=0.75, eps_0=0.2
    
    q_func_init_traj_epsilon0 = np.load(os.path.join('outputs', run_epsilon0, 'q_func_init_traj.npy'))
    q_func_init_traj_epsilon1 = np.load(os.path.join('outputs', run_epsilon1, 'q_func_init_traj.npy'))
    q_func_init_traj_epsilon2 = np.load(os.path.join('outputs', run_epsilon2, 'q_func_init_traj.npy'))
    q_func_init_traj_epsilon3 = np.load(os.path.join('outputs', run_epsilon3, 'q_func_init_traj.npy'))

    v_func_epsilon0 = np.max(q_func_init_traj_epsilon0, axis=-1) # max over actions
    v_func_epsilon1 = np.max(q_func_init_traj_epsilon1, axis=-1) # max over actions
    v_func_epsilon2 = np.max(q_func_init_traj_epsilon2, axis=-1) # max over actions
    v_func_epsilon3 = np.max(q_func_init_traj_epsilon3, axis=-1) # max over actions

    plt.plot(range(len(v_func_epsilon0)), v_func_epsilon0, 'y')
    plt.plot(range(len(v_func_epsilon1)), v_func_epsilon1, 'r')
    plt.plot(range(len(v_func_epsilon2)), v_func_epsilon2, 'g')
    plt.plot(range(len(v_func_epsilon3)), v_func_epsilon3, 'b')
    plt.title('Value function trajectory')
    plt.xlabel('Episodes')
    plt.ylabel('Value function')
    plt.legend(['no decay, alpha=2/3, eps_0=0.1', 'delta=0.6, alpha=2/3, eps_0=0.2', 'delta=0.85, alpha=2/3, eps_0=0.2', 'delta=0.6, alpha=0.75, eps_0=0.2'])
    plt.savefig(os.path.join(analysis_dir, 'value_func_traj.png'))
    plt.show()
    
    win_rates_epsilon0 = np.load(os.path.join('outputs', run_epsilon0, 'win_rate_list.npy'))
    win_rates_epsilon1 = np.load(os.path.join('outputs', run_epsilon1, 'win_rate_list.npy'))
    win_rates_epsilon2 = np.load(os.path.join('outputs', run_epsilon2, 'win_rate_list.npy'))
    win_rates_epsilon3 = np.load(os.path.join('outputs', run_epsilon3, 'win_rate_list.npy'))
    plt.plot(win_rates_epsilon0[:, 0], win_rates_epsilon0[:, 1], 'y')
    plt.plot(win_rates_epsilon1[:, 0], win_rates_epsilon1[:, 1], 'r')
    plt.plot(win_rates_epsilon2[:, 0], win_rates_epsilon2[:, 1], 'g')
    plt.plot(win_rates_epsilon3[:, 0], win_rates_epsilon3[:, 1], 'b')
    plt.title('Winning rates')
    plt.xlabel('Episodes')
    plt.ylabel('Winning rate')
    plt.legend(['no decay, alpha=2/3, eps_0=0.1', 'delta=0.6, alpha=2/3, eps_0=0.2', 'delta=0.85, alpha=2/3, eps_0=0.2', 'delta=0.6, alpha=0.75, eps_0=0.2'])
    plt.savefig(os.path.join(analysis_dir, 'win_rates.png'))
    plt.show()