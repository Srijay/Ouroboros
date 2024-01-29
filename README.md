# Ouroboros: Transmuting Protein Expression Perturbations to Cancer Histology Imaging with Generative-Predictive Modeling

![ouroboros](https://github.com/Srijay/Ouroboros/assets/6882352/77ab0e40-94f4-488c-8db1-6b57956dc989)


This repository contains code for using Ouroboros, our generative-predictive model.

Scroll down to the bottom to find instructions on downloading our pretrained weights 

# Set Up Environment
```
# create base conda environment
conda env create -f environment.yml

# activate environment
conda activate ouroboros

# install PyTorch with pip
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```
# Training

To train the model please update the hyper parameters in main.py and execute the following command:

```
python main.py --mode train
```

# Testing 
To test the model please update the hyper parameters in config.txt, give appropriate paths, and execute the following command:

```
python main.py --mode test --batch_size 1
```
