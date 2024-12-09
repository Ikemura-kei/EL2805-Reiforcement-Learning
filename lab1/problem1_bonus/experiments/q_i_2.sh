#! /bin/bash

python main.py \
    --algorithm 'q_learning' \
    --eps_init 0.1 \
    --alpha 0.666666 \
    --delta '-1' \
    --run_name 'q_i_2_eps=0.1_zeros_q0' \
    --max_eposide 50000 \
    --q_func_init 'zeros' \

python main.py \
    --algorithm 'q_learning' \
    --eps_init 0.1 \
    --alpha 0.666666 \
    --delta '-1' \
    --run_name 'q_i_2_eps=0.1_rand_q0' \
    --max_eposide 50000 \
    --q_func_init 'rand' \

python main.py \
    --algorithm 'q_learning' \
    --eps_init 0.2 \
    --alpha 0.666666 \
    --delta '-1' \
    --run_name 'q_i_2_eps=0.2_encourage_q0' \
    --max_eposide 50000 \
    --q_func_init 'encourage_move' \

python main.py \
    --algorithm 'q_learning' \
    --eps_init 0.1 \
    --alpha 0.666666 \
    --delta '-1' \
    --run_name 'q_i_2_eps=0.1_encourage_q0' \
    --max_eposide 50000 \
    --q_func_init 'encourage_move' \