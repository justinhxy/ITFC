# ISBI Algorithm for MCI Conversion Prediction

This repository contains the implementation of the algorithm presented in the paper titled "Title of the Paper" for MCI conversion prediction using MRI, PET, and clinical data.

## 1. Paper Algorithm Flowchart

![Algorithm Flowchart](f1.png)

## 2. Environment Setup

To run this project, ensure that you have the following software and dependencies installed:

- Python 3.x
- PyTorch (version)
- Other required libraries (e.g., NumPy, pandas, scikit-learn, etc.)

### Installation
pip install -r requirements.txt

## 3. Training and Inference Code
Training Code: To train the model, run the following command:
bash
python train.py
Inference Code: To perform inference on a trained model, use the following command:
bash
python inference.py --model_path <path_to_model>


## 4. Dataset Folder and Split Ratios
The dataset is organized as follows:
/dataset
    /MRI
    /PET
    /clinical_data
MRI: Directory containing the MRI scans in NIfTI format (.nii).
PET: Directory containing the PET scans in NIfTI format (.nii).
clinical_data: CSV files containing clinical data for the patients.
Data Split:
Training: 70%
Validation: 15%
Test: 15%
Ensure that the dataset is split accordingly for reproducible results.

## 5. Code Execution Example
Here is an example of how to run the training and inference code:

Training:
bash

python train.py --data_path <path_to_data> --epochs 50 --batch_size 32
Inference:
bash
python inference.py --model_path <path_to_trained_model> --data_path <path_to_data>

## 6. Experimental Results and Visualizations
The results of our experiments are summarized in the table below:
