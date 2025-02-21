#!/usr/bin/env bash

clear

if [ -d "build" ]; then
    cmake --build ./build --target clean
fi

echo '... Done!'
