import matplotlib.pyplot as plt
import numpy as np
import pickle
def scale_back(x):
    low = np.tile(np.array([-1.2, -0.07])[:,None], (1,len(x[0])))
    high = np.tile(np.array([0.6, 0.07])[:,None], (1,len(x[0])))
    # return (x-low) / (high-low)
    return x * (high-low) + low

best_q_func_file = './outputs/2024-12-09 20:26:07/weights.pkl'

f = open(best_q_func_file, 'rb')
data = pickle.load(f)

w = data['W'] # (A, M)
eta = data['N'].astype(np.float32) # (M, 2)

state_1s = np.linspace(0., 1., 120) # (K, )
state_2s = np.linspace(0, 1., 120) # (K, )
state_1s_, state_2s_ = np.meshgrid(state_1s, state_2s) # (K, K) for both
state_1s_ = np.reshape(state_1s_, (state_1s.shape[0]**2, )) # (K*K, )
state_2s_ = np.reshape(state_2s_, (state_2s.shape[0]**2, )) # (K*K, )

states = np.stack([state_1s_, state_2s_], axis=0) # (2, K*K)

phis = eta @ states # (M, K,K)
q_func = w @ np.cos(np.pi * phis) # (A, K,K)
v_func = np.max(q_func,axis=0) # max over actions to get value function (K*K, )
states = scale_back(states)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(states[0], states[1], v_func, c=v_func, cmap='BrBG')
# ax.scatter3D(state_1s, state_2s, v_func, c=v_func, cmap='Spectral_r')
ax.view_init(60, 35)
ax.set_xlabel('position')
ax.set_ylabel('velocity')
ax.set_zlabel('value function')
plt.show()
exit()
legends = []

state_1 = -1.0
states = np.stack([np.array([state_1]*len(state_2s)), state_2s], axis=0) # (2, K)
phis = eta @ states # (M, K)
q_func = w @ np.cos(np.pi * phis) # (A, K)
v_func = np.max(q_func,axis=0) # max over actions to get value function (K, )
plt.plot(state_2s, v_func, 'r')
legends.append('position={:.1f}'.format(state_1))


state_1 = -0.2
states = np.stack([np.array([state_1]*len(state_2s)), state_2s], axis=0) # (2, K)
phis = eta @ states # (M, K)
q_func = w @ np.cos(np.pi * phis) # (A, K)
v_func = np.max(q_func,axis=0) # max over actions to get value function (K, )
plt.plot(state_2s, v_func, 'm')
legends.append('position={:.1f}'.format(state_1))


state_1 = 0.0
states = np.stack([np.array([state_1]*len(state_2s)), state_2s], axis=0) # (2, K)
phis = eta @ states # (M, K)
q_func = w @ np.cos(np.pi * phis) # (A, K)
v_func = np.max(q_func,axis=0) # max over actions to get value function (K, )
plt.plot(state_2s, v_func, 'b')
legends.append('position={:.1f}'.format(state_1))


state_1 = 0.2
states = np.stack([np.array([state_1]*len(state_2s)), state_2s], axis=0) # (2, K)
phis = eta @ states # (M, K)
q_func = w @ np.cos(np.pi * phis) # (A, K)
v_func = np.max(q_func,axis=0) # max over actions to get value function (K, )
plt.plot(state_2s, v_func, 'g')
legends.append('position={:.1f}'.format(state_1))

state_1 = 0.4
states = np.stack([np.array([state_1]*len(state_2s)), state_2s], axis=0) # (2, K)
phis = eta @ states # (M, K)
q_func = w @ np.cos(np.pi * phis) # (A, K)
v_func = np.max(q_func,axis=0) # max over actions to get value function (K, )
plt.plot(state_2s, v_func, 'k')
legends.append('position={:.1f}'.format(state_1))

plt.xlabel('Velocity')
plt.ylabel('Value function')
plt.legend(legends)
plt.title('Value function at different fixed positions and all velocities')
plt.show()