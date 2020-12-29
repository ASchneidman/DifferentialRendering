# DifferentialRendering

# Steps

## Install ViLBERT

- `sudo apt-get install build-essential libcap-dev`
- Follow github to install ViLBERT https://github.com/facebookresearch/vilbert-multi-task
- `pip install -r requirements.txt`
- `cd vilbert-multi-task`
- Set up `tools/refer` (optionally, clone from my fork instead: https://github.com/ASchneidman/refer.git)
    - `cd tools/refer`
    - Edit line 39 of `refer.py` from 
    `from external import mask`
    to 
    `from pycocotools import mask`
    - `make`
- Install vqa_maskrcnn_benchmark (follow these exact steps!!)

```
cd vilbert-multi-task
git clone https://gitlab.com/ASchneidman/vqa-maskrcnn-benchmark.git
cd vqa-maskrcnn-benchmark
python setup.py build develop
```

- Likely the wrong version of transformers got installed, correct it with

```
pip uninstall pytorch_transformers
pip install pytorch_transformers==1.2.0
```

- There's likely an issue with tensorboard from pip, install it from conda

```
pip uninstall tensorboardX
conda install tensorboardX
```

- Download the pretrained visual feature extractor
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


- Download datasets (you can delete these after getting the file we need: `trainval_label2ans.pkl`, i.e. once the VQA dataset is extracted, you can cancel the untar)

```
cd multi-task-vilbert/data
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/datasets.tar.gz
tar xf datasets.tar.gz
```

- To test the installation, you should be able to run the demo.ipynb in this project's root directory


### Tips and Tricks

- When installing cudatoolkit into your conda env, try 10.1. You might get it wrong the first time, when you install apex, you may get an error about how pytorch isn't built with the same cuda bindings that are installed. If you look around in the trace, you should be able to find the version number it needs

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
