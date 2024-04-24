# BERT Tuning for multi-class classification

This project addressed a multilabel classification problem using two fine-tuning approaches on BERT, full fine-tuning and prefix-tuning.

The dataset features a popular YouTube video collection, encompassing information from 40,895 videos formatted as a large CSV table. Each video record includes details such as the title, category ID, creation time, uploading channel, and interaction metrics like comment and like counts. The objective is to predict the category ID assigned by YouTube to each video based on its content.

**Fine-tuning Approaches**

Fine-tuning: This method involves adjusting the weights of a large language model across all layers to adapt it to a specific task. 

Prefix-tuning: This technique introduces a randomly initiated low-dimensional vector as "virtual words" to the input layer, updating only these virtual tokens during training. The learning process focuses only on these "virtual tokens" rather than the entire sentence's parameters. Prefix-tuning adds a set of parameters before the existing model and freezes all pretrained parameters, allowing each batch element at inference to run a different tuned model. This can occur solely at the input layer or throughout every layer.


## Results

With resources and time constraint, this exercise has been set up in an exploratory environment and achieved 93.46% of accuracy rate. It has obtained satisfactory results which suggest that this solution could possibly be implemented in a practical context.




## Installation

Here's how to install the project.

```bash
git clone https://github.com/yihan-zhou/bert-classification-tuning.git
pip install -r requirements.txt
```

## Run Training

Run the following command for full BERT finetune on the classification dataset.

```bash
bash script/run_og.sh
```
Run the following command for prefix BERT finetune on the classification dataset.

```bash
bash script/run_prefix.sh
```

## File Structure

This repository is organized as follows:


    ├── csv/                        # Contains CSV files for the dataset 
    │   ├── test_simple.csv         # Test dataset
    │   ├── train_simple.csv        # Train dataset
    │   ├── val_simple.csv          # Validation dataset
    │   └── USvideo.csv             # Original dataset where the test, train, val datasets generated
    ├── model/                      # Model scripts and related files
    │   ├── utils.py                # Model utils functions
    │   ├── prefix_encoder.py       # Prefix encoder layer
    │   ├── sequence_classification.py   # Bert classification model
    ├── script/                     # Scripts for running the model
    │   ├── run_og.sh               # Scripts for running the finetuning
    │   ├── run_prefix.sh           # Scripts for running the prefix finetuning
    ├── arguments.py                # Hyperparameters parser
    ├── dataset.py                  # Handles data loading and preprocessing
    ├── main.py                     # Main script to run the models
    ├── preprocessing.ipynb         # Jupyter notebook for data preprocessing, script to generate the test, train, val datasets
    ├── requirements.txt            # Required libraries and dependencies to run the project
    ├── README.md                   # Top-level README with project overview
    └── trainer.py                  # Training functions for the model



