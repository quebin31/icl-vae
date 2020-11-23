#!/bin/bash

set -e

config="$1"
pipenv run python src/main.py --task=0 --config="$config" --train --test
