#!/bin/bash

set -e

./base.sh "$1"
./incremental.sh "$(cat .runid)" "$1"
