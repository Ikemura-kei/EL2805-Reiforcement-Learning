import numpy as np
from modules.maze import MazeEnv

q_func_path = './outputs/q_j_3_epsilon=0.3_delta_0.6_alpha=0.6666_2024-12-09 15:24:12/q_func.npy'

if __name__ == "__main__":
    q_func = np.load(q_func_path)
    
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 3],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]])

    horizon = 5000
    maze_test = MazeEnv(maze, episode_len=horizon, beta=0.65, minotaur_fix=False)

    for j in range(3):
        terminated = truncated = False
        start = (0, 0, 0, 6, 4)
        s_idx = maze_test.states_to_idx[start]
    
        for i in range(horizon):
            a_idx = np.argmax(q_func[s_idx])

            s_prime_idx, reward, terminated, truncated, info = maze_test.step(
                s_idx, a_idx, i, do_render=True, show=True)

            s_idx = s_prime_idx
            
            if terminated or truncated:
                break