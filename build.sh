#!/usr/bin/env bash
# Author: Gino Francesco Bogo

set -euo pipefail

clear

BUILD_TYPE=${1:-Debug}

echo "Build type: $BUILD_TYPE"

if [ ! -d "build" ]; then
    mkdir build
fi

cd ./build

cmake "../CmakeLists.txt" -B ./ -DCMAKE_BUILD_TYPE=$BUILD_TYPE

cmake --build ./ --config $BUILD_TYPE

cp ./compile_commands.json ../

echo '... Done!'
