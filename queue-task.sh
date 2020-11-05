#!/bin/bash
  
pueue add -- python3 src/main.py \
    --config "$1" \
    --task "$2" \
    --run-id "$3" \
    --train --test 
