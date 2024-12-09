import numpy as np
import matplotlib.pyplot as plt
import os

# -- analyze the effect of epsilon on q-learning (question i.2)
analysis_dir = os.path.join('.', 'analysis', 'q_i_2')
os.makedirs(analysis_dir, exist_ok=True)

run1 = 'q_i_2_eps=0.1_zeros_q0_2024-12-09 18:58:16' # epsilon=0.1, zeros_init
run2 = 'q_i_2_eps=0.1_rand_q0_2024-12-09 19:00:08' # epsilon=0.1, rand_init
run3 = 'q_i_2_eps=0.1_encourage_q0_2024-12-09 19:04:01' # epsilon=0.1, encourage_move_init
run4 = 'q_i_2_eps=0.2_encourage_q0_2024-12-09 19:02:04' # epsilon=0.2, encourage_move_init

q_func_init_traj1 = np.load(os.path.join('outputs', run1, 'q_func_init_traj.npy'))
q_func_init_traj2 = np.load(os.path.join('outputs', run2, 'q_func_init_traj.npy'))
q_func_init_traj3 = np.load(os.path.join('outputs', run3, 'q_func_init_traj.npy'))
q_func_init_traj4 = np.load(os.path.join('outputs', run4, 'q_func_init_traj.npy'))

v_func1 = np.max(q_func_init_traj1, axis=-1) # max over actions
v_func2 = np.max(q_func_init_traj2, axis=-1) # max over actions
v_func3 = np.max(q_func_init_traj3, axis=-1) # max over actions
v_func4 = np.max(q_func_init_traj4, axis=-1) # max over actions

plt.plot(range(len(v_func1)), v_func1, 'r')
plt.plot(range(len(v_func2)), v_func2, 'g')
plt.plot(range(len(v_func3)), v_func3, 'b')
plt.plot(range(len(v_func4)), v_func4, 'y')
plt.title('Value function trajectory')
plt.xlabel('Episodes')
plt.ylabel('Value function')
plt.legend(['epsilon=0.1, zeros_init', 'epsilon=0.1, rand_init', 'epsilon=0.1, encourage_move_init', 'epsilon=0.2, encourage_move_init'])
plt.grid(True)
plt.savefig(os.path.join(analysis_dir, 'value_func_traj.png'))
plt.show()

win_rates1 = np.load(os.path.join('outputs', run1, 'win_rate_list.npy'))
win_rates2 = np.load(os.path.join('outputs', run2, 'win_rate_list.npy'))
win_rates3 = np.load(os.path.join('outputs', run3, 'win_rate_list.npy'))
win_rates4 = np.load(os.path.join('outputs', run4, 'win_rate_list.npy'))
plt.plot(win_rates1[:15, 0], win_rates1[:15, 1], 'r')
plt.plot(win_rates2[:15, 0], win_rates2[:15, 1], 'g')
plt.plot(win_rates3[:15, 0], win_rates3[:15, 1], 'b')
plt.plot(win_rates4[:15, 0], win_rates4[:15, 1], 'y')
plt.title('Winning rates')
plt.xlabel('Episodes')
plt.ylabel('Winning rate')
plt.legend(['epsilon=0.1, zeros_init', 'epsilon=0.1, rand_init', 'epsilon=0.1, encourage_move_init', 'epsilon=0.2, encourage_move_init'])
plt.grid(True)
plt.savefig(os.path.join(analysis_dir, 'win_rates.png'))
plt.show()