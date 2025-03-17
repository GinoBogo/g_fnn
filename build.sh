#!/usr/bin/env bash
# Author: Gino Francesco Bogo

set -euo pipefail

clear

if [ ! -d "build" ]; then
    mkdir build
fi

cd ./build

cmake ".."

make

echo '... Done!'
