#!/bin/bash

set -e

./base.sh "$1"
runid=$(cat .runid)
./incremental.sh "$runid" "$1"
