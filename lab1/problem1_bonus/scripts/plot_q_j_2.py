import numpy as np
import matplotlib.pyplot as plt
import os

# -- analyze the effect of epsilon on q-learning (question i.2)
analysis_dir = os.path.join('.', 'analysis', 'q_j_2')
os.makedirs(analysis_dir, exist_ok=True)

class Info:
    run_name = ''
    legend = ''
    color = ''
    
    def __init__(self, run_name, legend, color):
        self.run_name = run_name
        self.legend = legend
        self.color = color
    
info_list = []
info_list.append(Info(run_name='q_j_2_epsilon=0.1_rand_q0_2024-12-09 19:01:30', legend='epsilon=0.1, rand_init', color='r'))
info_list.append(Info(run_name='q_j_2_epsilon=0.1_zeros_q0_2024-12-09 18:59:06', legend='epsilon=0.1, zeros_init', color='b'))
info_list.append(Info(run_name='q_j_2_epsilon=0.3_encourage_q0_2024-12-09 19:03:46', legend='epsilon=0.3, encourage_move_init', color='g'))
info_list.append(Info(run_name='q_j_2_epsilon=0.2_encourage_q0_2024-12-09 19:08:07', legend='epsilon=0.2, encourage_move_init', color='y'))
info_list.append(Info(run_name='q_j_2_epsilon=0.1_encourage_q0_2024-12-09 19:11:07', legend='epsilon=0.1, encourage_move_init', color='c'))

legends = []
for info in info_list:
    run = info.run_name
    q_func_init_traj = np.load(os.path.join('outputs', run, 'q_func_init_traj.npy'))
    v_func = np.max(q_func_init_traj, axis=-1) # max over actions
    plt.plot(range(len(v_func)), v_func, info.color)
    legends.append(info.legend)

plt.title('Value function trajectory')
plt.xlabel('Episodes')
plt.ylabel('Value function')
plt.legend(legends)
plt.grid(True)
plt.savefig(os.path.join(analysis_dir, 'value_func_traj.png'))
plt.show()

for info in info_list:
    run = info.run_name
    win_rates = np.load(os.path.join('outputs', run, 'win_rate_list.npy'))
    plt.plot(win_rates[:, 0], win_rates[:, 1], info.color)
    
plt.title('Winning rates')
plt.xlabel('Episodes')
plt.ylabel('Winning rate')
plt.legend(legends)
plt.grid(True)
plt.savefig(os.path.join(analysis_dir, 'win_rates.png'))
plt.show()