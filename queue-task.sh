#!/bin/bash

task="$1"
runid="$2"
seed="$3"
batch="16"
lr="0.000004" 
rlambda="0.01"
epochs="100"
decay="0.8"
   
pueue add -- python3 src/main.py \
    --batch "${batch}" \
    --lr "${lr}" \
    --rlambda "${rlambda}" \
    --epochs "${epochs}" \
    --seed "${seed}" \
    --task "${task}" \
    --run-id "${runid}" \
    --decay "${decay}" \
    --train --test 