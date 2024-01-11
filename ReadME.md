# Q-SENN (Quantized Self-Explaining Neural Network).
This repository contains the code for the AAAI 2024 paper "Q-SENN: Quantized Self-Explaining Neural Network" by Thomas
Norrenbrock, Marco Rudolph and Bodo Rosenhahn. Additonally, the SLDD-model from "Take 5: 
Interpretable Image Classification with a Handful of Features" (NeurIPS Workshop) from the same authors is included.
# Usage
The code to create a Q-SENN model can be started from the file main.py.
To create a Q-SENN for Cub-2011 with resnet50 as backbone, one can execute main.py directly.
To change parameters, e.g.  the dataset, backbone, or create a sldd model, one can pass arguments to main.py.
See main.py --help for more details. Most parameters are hardcoded somewhere to be the default arguments
used for all experiments.

For Example, python main.py --dataset StanfordCars will start the creation of Q-SENN with resnet50
on StanfordCars using the default arguments in the paper.

# Environment
You will need the usual libaries for deep learning, e.g. pytorch, torchvision, numpy, etc.
The code ran with python 3.10.9 and torch 1.13.
Additionally glm-saga (https://github.com/MadryLab/glm_saga) is required, and can be installed via pip.

# Datasets
The datasets are not included in this repository. Currently supported datasets are:

Cub2011, ImageNet, StanfordCars, TravelingBirds

They have to be downloaded manually and put into the respective folder under ~/tmp/datasets (or wherever, if the code
is changed accordingly).
The default paths could be changed in the dataset_classes or for Imagenet in get_data.py

For Cub, the default path is: Path.home() / "tmp/Datasets/CUB200/CUB_200_2011/images"

If cropped images, like for PIP-Net, ProtoPool, etc. are desired, then the crop_root should be set to a folder containing the
cropped images in the expected structure, obtained by following ProtoTree's instructions:
https://github.com/M-Nauta/ProtoTree/blob/main/README.md#preprocessing-cub, default path is: PPCUB200  instead of CUB200
for Protopool. Using these images can be set via --cropGT.


# ImageNet
All experiments so far skipped the dense training from scratch on ImageNet. The pretrained models are used directly.
This can be replicated with the argument --do-dense False.

# Citation
If you use this code, please cite our paper:
Upcoming