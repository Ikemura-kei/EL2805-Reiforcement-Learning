#! /bin/bash

python main.py \
    --algorithm 'q_learning' \
    --eps_init 0.1 \
    --alpha 0.6 \
    --delta '-1' \
    --run_name 'q_i_3_alpha=0.6' \
    --max_eposide 50000 \
    --q_func_init 'encourage_move' \

python main.py \
    --algorithm 'q_learning' \
    --eps_init 0.1 \
    --alpha 0.85 \
    --delta '-1' \
    --run_name 'q_i_3_alpha=0.85' \
    --max_eposide 50000 \
    --q_func_init 'encourage_move' \