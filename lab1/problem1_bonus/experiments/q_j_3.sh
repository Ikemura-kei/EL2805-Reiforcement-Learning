#! /bin/bash

python main.py \
    --algorithm 'sarsa' \
    --eps_init 0.2 \
    --alpha 0.6666666 \
    --delta 0.6 \
    --run_name 'q_j_3_epsilon=0.2_delta_0.6_alpha=0.6666' \
    --max_eposide 50000 \
    --q_func_init 'encourage_move' \

python main.py \
    --algorithm 'sarsa' \
    --eps_init 0.2 \
    --alpha 0.85 \
    --delta 0.6 \
    --run_name 'q_j_3_epsilon=0.2_delta_0.6_alpha=0.85' \
    --max_eposide 50000 \
    --q_func_init 'encourage_move' \

python main.py \
    --algorithm 'sarsa' \
    --eps_init 0.2 \
    --alpha 0.6666666 \
    --delta 0.8 \
    --run_name 'q_j_3_epsilon=0.2_delta_0.8_alpha=0.6666' \
    --max_eposide 50000 \
    --q_func_init 'encourage_move' \

python main.py \
    --algorithm 'sarsa' \
    --eps_init 0.2 \
    --alpha 0.6666666 \
    --delta 0.99 \
    --run_name 'q_j_3_epsilon=0.2_delta_0.99_alpha=0.6666' \
    --max_eposide 50000 \
    --q_func_init 'encourage_move' \

python main.py \
    --algorithm 'sarsa' \
    --eps_init 0.1 \
    --alpha 0.6666666 \
    --delta 0.6 \
    --run_name 'q_j_3_epsilon=0.1_delta_0.6_alpha=0.6666' \
    --max_eposide 50000 \
    --q_func_init 'encourage_move' \

python main.py \
    --algorithm 'sarsa' \
    --eps_init 0.3 \
    --alpha 0.6666666 \
    --delta 0.6 \
    --run_name 'q_j_3_epsilon=0.3_delta_0.6_alpha=0.6666' \
    --max_eposide 50000 \
    --q_func_init 'encourage_move' \