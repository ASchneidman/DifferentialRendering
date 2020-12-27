# DifferentialRendering

# Steps

## Install mitsuba2

1. Follow guide to set up mitsuba2: https://mitsuba2.readthedocs.io/en/latest/src/getting_started/cloning.html

### Tips and Tricks

- Ensure latest nvidia drivers are installed
- Had to download optix headers manually from https://developer.nvidia.com/designworks/optix/download
- Look at mitsuba_setup.sh to help out

## Install ViLBERT

- Follow github to install ViLBERT https://github.com/facebookresearch/vilbert-multi-task
- `cd vilbert-multi-task`
- Set up `tools/refer`
    - `cd tools/refer`
    - `pip install pycocotools`
    - Edit line 39 of `refer.py` from 
    `from external import mask`
    to 
    `from pycocotools import mask`
    - `make`
    - `cd ../..`
- Install vqa_maskrcnn_benchmark (follow these exact steps!!)

```
git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git
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
cd data
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml
```
- Download datasets (you can delete these after getting the file we need: `trainval_label2ans.pkl`, i.e. once the VQA dataset is extracted, you can cancel the untar)

```
cd data
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/datasets.tar.gz
tar xf datasets.tar.gz
```


### Tips and Tricks

- When installing cudatoolkit into your conda env, try 10.1. You might get it wrong the first time, when you install apex, you may get an error about how pytorch isn't built with the same cuda bindings that are installed. If you look around in the trace, you should be able to find the version number it needs