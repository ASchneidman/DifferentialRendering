# DifferentialRendering

Requirements:

1. Cuda 10.1 and up
2. Pytorch 1.7 and up
2. Torchvision 0.8.2 and up

# Steps

## Install ViLBERT
```
conda create -n vilbert-mt python=3.6
conda activate vilbert-mt

sudo apt-get install build-essential libcap-dev

pip install -r requirements.txt
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install notebook scikit-image

git clone https://github.com/facebookresearch/vilbert-multi-task.git

cd vilbert-multi-task/tools
rm -rf refer
git clone https://github.com/ASchneidman/refer.git
cd refer
make
cd ../../
```

Install apex from https://github.com/NVIDIA/apex

Install vqa_maskrcnn_benchmark

```
git clone https://gitlab.com/ASchneidman/vqa-maskrcnn-benchmark.git
cd vqa-maskrcnn-benchmark
python setup.py build develop
```

Download the pretrained visual feature extractor

```
cd vilbert-multi-task/data
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml
```

- Download pretrained ViLBERT

```
cd multi-task-vilbert
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/multi_task_model.bin
```


- Download pkl dictionaries

```
cd multi-task-vilbert/data
mkdir -p datasets/VQA/cache

aws s3 sync s3://diffrendering .
```

- To test the installation, you should be able to run the demo.ipynb in this project's root directory


### Tips and Tricks

- When installing cudatoolkit into your conda env, you might get it wrong the first time, when you install apex, you may get an error about how pytorch isn't built with the same cuda bindings that are installed. If you look around in the trace, you should be able to find the version number it needs

## Install mitsuba2

1. Make sure you are in the same python env as the one used for installing ViLBERT (probably vilbert-mt)

```
git clone --recursive https://github.com/mitsuba-renderer/mitsuba2
cp mitsuba.conf mitsuba2

sudo apt install -y clang-9 libc++-9-dev libc++abi-9-dev cmake ninja-build
sudo apt install -y libz-dev libpng-dev libjpeg-dev libxrandr-dev libxinerama-dev libxcursor-dev
sudo apt install -y python3-dev python3-distutils python3-setuptools

cd mitsuba2
```

2. Edit CMakeLists.txt
    - Add the following to line 27

```
set(PYTHON_LIBRARY /home/$USER/anaconda3/envs/vilbert-mt/lib/libpython3.6m.so)
set(PYTHON_INCLUDE_DIR /home/$USER/anaconda3/envs/vilbert-mt/include/python3.6m)
```

3. 

```
mkdir build
cd build

export CC=clang-9
export CXX=clang++-9

cmake -GNinja ..

ninja
```

### Tips and Tricks

- Ensure latest nvidia drivers are installed
- Had to download optix headers manually from https://developer.nvidia.com/designworks/optix/download
    - If you experience an issue where `optix.h` cannot be found when attempting to use mitsuba2, try
    downloading optix headers directly from nvidia: https://developer.nvidia.com/designworks/optix/download
    - Then, point cmake at the headers using the following:

```
cmake -GNinja -DMTS_USE_OPTIX_HEADERS=OFF -DMTS_OPTIX_PATH=[YOUR PATH TO THE OPTIX DOWNLOAD]/NVIDIA-OptiX-SDK-7.2.0-linux64-x86_64/include ..
```
