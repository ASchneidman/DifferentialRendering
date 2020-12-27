#!/bin/sh

git clone --recursive https://github.com/mitsuba-renderer/mitsuba2
cp mitsuba.conf mitsuba2

sudo apt install -y clang-9 libc++-9-dev libc++abi-9-dev cmake ninja-build
sudo apt install -y libz-dev libpng-dev libjpeg-dev libxrandr-dev libxinerama-dev libxcursor-dev
sudo apt install -y python3-dev python3-distutils python3-setuptools

mkdir build
cd build

export CC=clang-9
export CXX=clang++-9

cmake -GNinja -DMTS_ENABLE_GUI=ON -DMTS_USE_OPTIX_HEADERS=OFF -DMTS_OPTIX_PATH=/home/alex/Downloads/NVIDIA-OptiX-SDK-7.2.0-linux64-x86_64/include ..

ninja
