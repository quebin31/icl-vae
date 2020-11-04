#!/bin/bash

task="$1"
runid="$2"
seed="$3"
batch="16"
lr="0.000003"
rlambda="0.01"
epochs="10"

cmd="python3 src/main.py \
    --batch ${batch} \
    --lr ${lr} \
    --rlambda ${rlambda} \
    --epochs ${epochs}\
    --seed ${seed} \
    --task ${task} \
    --run-id ${runid} \
    --train --test"
    
pueue add -- "${cmd}"