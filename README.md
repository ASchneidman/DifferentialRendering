# DifferentialRendering

# Steps


## Install ViLBERT

- `sudo apt-get install build-essential libcap-dev`
- Follow github to install ViLBERT https://github.com/facebookresearch/vilbert-multi-task
- `cd vilbert-multi-task`
- Set up `tools/refer` (optionally, clone from my fork instead: https://github.com/ASchneidman/refer.git)
    - `cd tools/refer`
    - `pip install pycocotools`
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

## Install mitsuba2 (read tips and tricks first)

1. Make sure you are in the same python env as the one used for installing ViLBERT (probably vilbert-mt)
2. Follow guide to set up mitsuba2: https://mitsuba2.readthedocs.io/en/latest/src/getting_started/cloning.html

### Tips and Tricks

- Edit CMakeLists.txt
    - Add the following to line 27

```
set(PYTHON_LIBRARY /home/$USER/anaconda3/envs/vilbert-mt/lib/libpython3.6m.so)
set(PYTHON_INCLUDE_DIR /home/$USER/anaconda3/envs/vilbert-mt/include/python3.6m)
```

- Ensure latest nvidia drivers are installed
- Had to download optix headers manually from https://developer.nvidia.com/designworks/optix/download
- Look at mitsuba_setup.sh to help out
