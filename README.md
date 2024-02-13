# Ouroboros: Transmuting Protein Expression Perturbations to Cancer Histology Imaging with Generative-Predictive Modeling

![ouroboros](https://github.com/Srijay/Ouroboros/assets/6882352/77ab0e40-94f4-488c-8db1-6b57956dc989)


This repository contains code for using Ouroboros, our generative-predictive model.

Scroll down to the bottom to find instructions on downloading our pretrained weights 

# Set Up Environment

Clone the repository, and execute the following commands to set up the environment.

```
cd Ouroboros

# create base conda environment
conda env create -f environment.yml

# activate environment
conda activate ouroboros

# install PyTorch with pip
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

# Download data and extract patches

The batch-corrected protein expression data is given in the 'data' folder with filename protein_expression_data.csv. The whole slie images can be downloaded from [here]https://drive.google.com/drive/folders/1onYsDRu6DbmxwmGHu0hY4cePKvK2YHyg?usp=sharing, and put it in the folder 'images' inside the 'data' folder. Run the script to extract patches: 

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

# Evaluation

The scripts used to evaluate the Ouroboros framework are kept under the 'evaluation' folder.
