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
|       |       | ADNI1 | ADNI2 |
| Methods          | Modality  | ACC   | SPE   | SEN   | AUC   | F1    | ACC   | SPE   | SEN   | AUC   | F1    |
|------------------|-----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| ResNet           | **M+P**   | 0.725 | 0.823 | 0.564 | 0.653 | 0.606 | 0.809 | 0.928 | 0.437 | 0.709 | 0.510 |
| JSRL             | **M+P**   | 0.582 | 0.779 | 0.354 | 0.571 | 0.580 | 0.650 | 0.590 | 0.720 | 0.694 | 0.602 |
| HOPE             | **M**     | 0.611 | 0.786 | 0.611 | 0.648 | 0.593 | 0.712 | 0.860 | 0.712 | 0.616 | 0.692 |
| VAPL             | **M+C**   | 0.630 | 0.564 | 0.693 | 0.635 | 0.651 | 0.835 | 0.843 | 0.745 | 0.865 | 0.671 |
| HFBSurv          | **M+P+C** | 0.921 | 0.904 | 0.937 | 0.920 | 0.916 | 0.954 | 0.977 | 0.909 | 0.943 | 0.932 |
| ITCFN (w/o MMG)  | **M+P+C** | 0.932 | 0.925 | 0.937 | 0.931 | 0.927 | **0.960** | **0.992** | **0.937** | **0.965** | **0.960** |
| **ITCFN (Ours)** | **M+P+C** | **0.947** | **0.949** | **0.944** | **0.946** | **0.944** | 0.954 | 0.980 | 0.906 | 0.943 | 0.931 |
