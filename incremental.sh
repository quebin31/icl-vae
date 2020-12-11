#!/bin/bash

set -e

runid="$1"
config="$2"
shift 2

for i in $(seq 1 50); do
    pipenv run python src/main.py --task=$i --config="$config" --prev-id="$runid" --train --test "$@"
    runid=$(cat .runid)
done 
