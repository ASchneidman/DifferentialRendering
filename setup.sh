#!/bin/sh

# Sets up this project on an aws ec2 instance with the deep learning base ubuntu 18 AMI with conda installed

sudo apt-get install build-essential libcap-dev

export CONDA_ALWAYS_YES="true"
conda init bash

sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-10.1 /usr/local/cuda

conda create -n vilbert-mt python=3.6
conda activate vilbert-mt

git clone https://github.com/ASchneidman/DifferentialRendering.git
cd DifferentialRendering
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


git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../

git clone https://gitlab.com/ASchneidman/vqa-maskrcnn-benchmark.git
cd vqa-maskrcnn-benchmark
python setup.py build develop

cd ../data

wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml
mkdir -p datasets/VQA/cache
aws s3 sync s3://diffrendering datasets/VQA/cache/

cd ..
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/multi_task_model.bin



