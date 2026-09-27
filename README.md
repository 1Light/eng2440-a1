# ENG2440 Assignment 1: Pneumonia Classification Challenge

**Student:** Nasir Adem Degu  
**Course:** ENG 2440 Medical Imaging & AI in Healthcare

## Overview

This repository contains the implementation for Assignment 1: Pneumonia Classification Challenge using the RSNA Pneumonia Detection Challenge dataset.

The project implements an end-to-end binary classification pipeline for frontal chest radiographs, with the target defined as:

- **Positive:** lung opacity consistent with possible pneumonia
- **Negative:** no such annotated opacity

The main modeling experiment compares three levels of classification:

1. **Trivial baseline:** always predicts the majority class.
2. **Simple PyTorch baseline:** a single `nn.Linear` classifier using eight image-summary features.
3. **Pretrained CNN:** an ImageNet-pretrained DenseNet121 adapted for chest radiographs, first trained as a fixed feature extractor and then fine-tuned.

The notebook covers data inspection, leakage-safe patient-level splitting, preprocessing and augmentation, class-imbalance handling, controlled baseline-to-CNN comparison, multi-seed evaluation, operating-threshold analysis, calibration, subgroup analysis, Grad-CAM, and error analysis.

It also includes two additional experiments:

- **Prevalence shift and threshold transfer:** tests how the model's performance changes when pneumonia prevalence differs from the original test set.
- **Shortcut stress test with a control:** compares the effect of masking image borders with masking an approximately equal-area region inside the expected lung fields.

## Repository contents

```text
.
├── 25011150_ENG2440_A1.ipynb
├── requirements.txt
├── ai_use_declaration.txt
├── README.md
└── .gitignore
```

- `25011150_ENG2440_A1.ipynb`: fully executed assignment notebook.
- `requirements.txt`: direct Python dependencies used by the notebook.
- `ai_use_declaration.txt`: declaration of generative-AI assistance used during the assignment.
- `.gitignore`: prevents the medical-image dataset, supplied data files, and generated/local files from being committed.

## Dataset

The chest radiographs are from the **RSNA Pneumonia Detection Challenge**.

The medical images are intentionally **not included in this repository**. They must be obtained separately from the official RSNA dataset source.

The following two supporting CSV files are supplied with the assignment and are also **not included in this repository**:

- `assignment1_labels.csv`: annotation and binary-target information.
- `rsna_to_nih_mapping.csv`: mapping between RSNA examination identifiers and anonymized NIH patient identifiers.

Before running the notebook, place the supplied CSV files in the project root and arrange the DICOM files locally as:

```text
.
├── data/
│   └── images/
│       └── [DICOM files]
├── 25011150_ENG2440_A1.ipynb
├── assignment1_labels.csv
├── rsna_to_nih_mapping.csv
├── requirements.txt
├── ai_use_declaration.txt
├── README.md
└── .gitignore
```

The notebook expects:

```text
DATA_ROOT = data/images/
LABEL_FILE = assignment1_labels.csv
MAPPING_FILE = rsna_to_nih_mapping.csv
```

The `data/` directory, DICOM files, and supplied CSV files are excluded from Git.

## Setup and execution

The notebook can be run in Google Colab or in a local Jupyter-compatible environment.

### Google Colab

Open `25011150_ENG2440_A1.ipynb` in Google Colab.

Most required packages are already available in Colab. If any dependencies are missing, install them from a notebook cell with:

```python
%pip install -r requirements.txt
```

Ensure that the two supplied CSV files and the DICOM dataset are available at the paths shown above, then run the notebook cells sequentially from top to bottom.

### Local Jupyter / VS Code

Install the dependencies into the Python environment that will be used as the notebook kernel:

```bash
python -m pip install -r requirements.txt
```

Then open `25011150_ENG2440_A1.ipynb` in Jupyter, JupyterLab, VS Code, or another Jupyter-compatible environment.

Select the Python environment in which the requirements were installed as the notebook kernel, then run the cells sequentially from top to bottom.

## Dependencies

The direct Python dependencies are listed in `requirements.txt`:

```text
matplotlib
numpy
pandas
pydicom
scikit-learn
torch
torchvision
```

Python standard-library modules used by the notebook do not require separate installation.

## Reproducibility

The submitted notebook is fully executed and contains the outputs used in the assignment report.

Fixed random seeds are used throughout the notebook, including for data splitting, the simple baseline, augmentation examples, representative-example selection, prevalence-shift subsampling, and CNN training.

The headline CNN configuration is evaluated using three seeds:

```text
42
7
1234
```

Python, NumPy, PyTorch, CUDA, and DataLoader randomness are explicitly controlled where applicable, and deterministic cuDNN behavior is enabled.

Exact bitwise reproduction across different hardware, CUDA versions, PyTorch versions, or other software environments is not guaranteed.

## Data and clinical-use notice

The classification target is derived from expert image annotations of possible pneumonia and is not a confirmed clinical diagnosis.

This project was developed for coursework and is **not a clinical diagnostic system**.