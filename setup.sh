#!/bin/sh

sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-10.1 /usr/local/cuda

sleep 10m

wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
chmod +x Anaconda3-2020.11-Linux-x86_64.sh
/bin/bash Anaconda3-2020.11-Linux-x86_64.sh -b -p $HOME/Anaconda3

/home/ubuntu/anaconda3/bin/conda create -n vilbert-mt python=3.6
/home/ubuntu/anaconda3/bin/conda activate vilbert-mt

sudo apt-get install build-essential libcap-dev

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

cd data
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml
mkdir -p datasets/VQA/cache
aws s3 sync s3://diffrendering datasets/VQA/cache/

cd ..
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/multi_task_model.bin



