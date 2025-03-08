#!/usr/bin/env bash

set -euo pipefail

clear

if [ -d "build" ]; then
    cmake --build ./build --target clean
fi

echo '... Done!'

