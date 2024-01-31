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

# Download data and extract patches

The link to download whole slide images will be active upon publication. After downloading whole slide images, create a folder named 'images' inside the 'data' folder, and run the script: 

```
python extract_patches.py
```

The script will create a folder named 'patches' inside the 'data' folder which will have patches of size 256×256 pixels from whole slide images.


# Training

The trained weights can be downloaded [here](https://drive.google.com/drive/folders/1VbB6Ep06hlrPlBrnXSzba6mancuU1iN9?usp=sharing). Put the trained weights inside the trained_models folder. To train the model please update the hyper parameters in main.py and execute the following command:

```
python main.py --mode train
```

# Testing 
To test the model, execute the following command:

```
python main.py --mode test --batch_size 1
```
