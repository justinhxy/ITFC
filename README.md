# ITCFN: Incomplete Triple-Modal Co-Attention Fusion Network for Mild Cognitive Impairment Conersion Prediction

This repository contains the implementation of the algorithm presented in the paper titled "Title of the Paper" for MCI conversion prediction using MRI, PET, and clinical data.

## 1. Paper Algorithm Flowchart

![Algorithm Flowchart](f1.png)

## 2. Environment Setup
 We conducted our experiments using the PyTorch 2.0 framework, utilizing a single NVIDIA A100 80 GB GPU for computational efficiency. The model was trained from scratch over two distinct stages, each consisting of 200 epochs, with a batch size of 8 to effectively manage the data. We optimized the model parameters using the Adam algorithm, setting the learning rate to 0.0001 to ensure precise adjustments during training.


### Installation
pip install -r requirements.txt

## 3. Training and Inference Code



## 4. Dataset Folder and Split Ratios
To ensure reproducible and comparable results, we employed 5-fold cross-validation in all experiments, validating the model's stability and generalization while maintaining a consistent random seed for data splitting.

The dataset for this study is obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI), specifically the ADNI-1 and ADNI-2 cohorts. To prevent duplication, subjects present in both datasets were removed from ADNI-2. We selected T1-weighted sMRI, FDG-PET, and clinical data, categorized into four groups: normal controls (NC), sMCI, pMCI, and AD. Demographic information of the dataset is shown in Table below. Additionally, PET data is missing for 82 pMCI and 95 sMCI cases in ADNI-1, and for 1 pMCI and 30 sMCI cases in ADNI-2.

| Variable         | ADNI1 - AD   | ADNI1 - pMCI | ADNI1 - sMCI | ADNI1 - NC  | ADNI2 - AD  | ADNI2 - pMCI | ADNI2 - sMCI | ADNI2 - NC  |
|------------------|--------------|--------------|--------------|-------------|-------------|--------------|--------------|-------------|
| Number (M/F)     | 88/83        | 90/61        | 136/72       | 103/104     | 89/67       | 43/38        | 156/125      | 132/165     |
| Age              | 75.35±7.47   | 74.63±7.18   | 74.75±7.63   | 75.92±5.12  | 74.75±8.09  | 72.60±7.27   | 71.29±7.43   | 72.80±6.01  |
| Education        | 14.64±3.19   | 15.66±2.92   | 15.61±3.11   | 15.91±2.87  | 15.72±2.75  | 16.29±2.55   | 16.31±2.61   | 16.61±2.5   |
| CDR-SB           | 4.32±1.58    | 1.85±0.98    | 1.38±0.75    | 0.03±0.12   | 4.51±1.67   | 2.18±0.95    | 1.33±0.82    | 0.04±0.15   |
| MMSE             | 23.23±2.03   | 26.59±1.7    | 27.33±1.77   | 29.14±0.98  | 23.12±2.07  | 27.1±1.82    | 28.21±1.63   | 28.99±1.26  |


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
| Methods          |ADNI1| Modality  | ACC   | SPE   | SEN   | AUC   | F1    |ADNI2| ACC   | SPE   | SEN   | AUC   | F1    |
|------------------|-------|-----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| ResNet           || **M+P**   | 0.725 | 0.823 | 0.564 | 0.653 | 0.606 || 0.809 | 0.928 | 0.437 | 0.709 | 0.510 |
| JSRL             || **M+P**   | 0.582 | 0.779 | 0.354 | 0.571 | 0.580 || 0.650 | 0.590 | 0.720 | 0.694 | 0.602 |
| HOPE             || **M**     | 0.611 | 0.786 | 0.611 | 0.648 | 0.593 || 0.712 | 0.860 | 0.712 | 0.616 | 0.692 |
| VAPL             || **M+C**   | 0.630 | 0.564 | 0.693 | 0.635 | 0.651 || 0.835 | 0.843 | 0.745 | 0.865 | 0.671 |
| HFBSurv          || **M+P+C** | 0.921 | 0.904 | 0.937 | 0.920 | 0.916 || 0.954 | 0.977 | 0.909 | 0.943 | 0.932 |
| ITCFN (w/o MMG)  || **M+P+C** | 0.932 | 0.925 | 0.937 | 0.931 | 0.927 || **0.960** | **0.992** | **0.937** | **0.965** | **0.960** |
| **ITCFN (Ours)** || **M+P+C** | **0.947** | **0.949** | **0.944** | **0.946** | **0.944** || 0.954 | 0.980 | 0.906 | 0.943 | 0.931 |



| Methods          |ADNI1| ACC   | SPE   | SEN   | AUC   | F1    |ADNI2| ACC   | SPE   | SEN   | AUC   | F1    |
|------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| None             || 0.889 | 0.904 | 0.877 | 0.904 | 0.895 || 0.941 | 0.963 | 0.918 | 0.932 | 0.934 |
| MMG only         || 0.889 | 0.890 | 0.891 | 0.890 | 0.879 || 0.948 | 0.976 | 0.914 | 0.935 | 0.932 |
| TCAF only        || 0.932 | 0.925 | 0.937 | 0.931 | 0.927 || **0.960** | **0.992** | **0.937** | **0.965** | **0.960** |
| MMG+TCAF         || **0.947** | **0.949** | **0.944** | **0.946** | **0.944** || 0.954 | 0.980 | 0.906 | 0.943 | 0.931 |
