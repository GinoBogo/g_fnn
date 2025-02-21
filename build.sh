#!/usr/bin/env bash

clear

if [ ! -d "build" ]; then
    mkdir build
fi

cd ./build

cmake ..

make

echo '... Done!'