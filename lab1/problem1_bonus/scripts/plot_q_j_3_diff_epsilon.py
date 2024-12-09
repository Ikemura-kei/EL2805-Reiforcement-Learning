import numpy as np
import matplotlib.pyplot as plt
import os

# -- analyze the effect of epsilon on q-learning (question i.2)
analysis_dir = os.path.join('.', 'analysis', 'q_j_3_diff_epsilon')
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
info_list.append(Info(run_name='q_j_3_epsilon=0.1_delta_0.6_alpha=0.6666_2024-12-09 15:22:27', legend='epsilon=0.1, delta=0.6, alpha=2/3', color='b'))
info_list.append(Info(run_name='q_j_3_epsilon=0.2_delta_0.6_alpha=0.6666_2024-12-09 15:14:59', legend='epsilon=0.2, delta=0.6, alpha=2/3', color='r'))
info_list.append(Info(run_name='q_j_3_epsilon=0.3_delta_0.6_alpha=0.6666_2024-12-09 15:24:12', legend='epsilon=0.3, delta=0.6, alpha=2/3', color='k'))

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
plt.savefig(os.path.join(analysis_dir, 'win_rates.png'))
plt.show()