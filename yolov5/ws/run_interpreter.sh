#!/bin/bash

echo "Sourcing env '$*'"
set -x
source set_env.sh "$1" "$2" "$3"
set +x
echo "Running interpreter"

python interpreter.py "$base_path"


