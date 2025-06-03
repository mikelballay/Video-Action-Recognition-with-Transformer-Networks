# Optical Flow Action Recognition (UF)

Research pipeline developed at the University of Florida (UF) for optical flow extraction, embedding computation, and video action classification. It includes code for feature extraction (BN-Inception + NVIDIA Optical Flow), training scripts for a transformer (BERT) on embedding sequences, and evaluation of results.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Repository Structure](#repository-structure)  
3. [Requirements](#requirements)  
4. [Setup & Installation](#setup--installation)  
5. [Usage](#usage)  
   - [1. Optical Flow Extraction & Embedding](#1-optical-flow-extraction--embedding)  
   - [2. Training the Transformer Model](#2-training-the-transformer-model)  
   - [3. Evaluation](#3-evaluation)  
6. [Results & Metrics](#results--metrics)  
7. [Known Limitations](#known-limitations)  
8. [Report & Documentation](#report--documentation)  
9. [License](#license)  

---

## Project Overview

This repository contains a complete research pipeline for:
- **Optical Flow Extraction**: Compute optical flow from video frames using NVIDIA CUDA-based OpenCV.
- **Embedding Computation**: Extract per-frame features using a BN-Inception model pre-trained on TSN (Temporal Segment Networks).
- **Transformer Classification**: Train a BERT-style transformer on sequences of embeddings to classify actions in short video clips.
- **Evaluation**: Generate overall and per-class accuracy metrics on a validation set.

The code was developed and tested on a University of Florida virtual machine. Some heavy files (trained model weights, full datasets) are not included here, but example scripts and folder structure are provided so you can reproduce the pipeline if you place the appropriate data and weight files in the correct paths.

---

## Repository Structure
uf-action-recognition/
├── README.md
├── requirements.txt
├── docs/
│ └── UF_paper.pdf
├── data/
│ └── example.npy
├── embeddings/
│ └── example.npy
├── scripts/
│ ├── 01_optical_flow_extraction.py
│ ├── 02_training_transformer.py
│ ├── 03_evaluation.py
│ └── utils.py
└── LICENSE


- **`docs/UF_paper.pdf`**: Final research report, methodology, results, and conclusions.  
- **`data/` & `embeddings/`**: Contain example `.npy` files so you can see the expected array shapes.  
- **`scripts/`**: Main Python scripts—see details below.  
- **`requirements.txt`**: List of Python packages required to run the code.  

---

## Requirements

- Python 3.8 or higher  
- CUDA-enabled GPU (recommended) or CPU fallback  
- PyTorch (≥ 1.7)  
- torchvision  
- transformers  
- opencv-python (with CUDA support, if available)  
- numpy  
- scikit-learn  
- Pillow  
- tqdm  

Install dependencies via:

```bash
pip install -r requirements.txt

> **Note:** If you don’t have a CUDA-enabled GPU, the code will automatically fall back to
CPU, but runtime will be significantly slower.
---
## Setup & Installation
1. **Clone this repository**
 ```bash
 git clone https://github.com/TuUsuario/uf-action-recognition.git
 cd uf-action-recognition
 ```
2. **Install Python dependencies**
 ```bash
 pip install -r requirements.txt
 ```
3. **Prepare data directories**
 The pipeline expects the following folder structure for raw videos and embeddings:
 ```
 /path/to/DARPA/ecole/train/<class_label>/*.mp4
 /path/to/DARPA/ecole/val/<class_label>/*.mp4
 /path/to/DARPA_embeddings/train/<class_label>/*.npy
 /path/to/DARPA_embeddings/val/<class_label>/*.npy
 ```
 - Replace `/path/to/…` with your actual local path.
 - If you do not have access to the original DARPA data or precomputed embeddings, see
[Known Limitations](#known-limitations).
---
## Usage
### 1. Optical Flow Extraction & Embedding
**Script**: `scripts/01_optical_flow_extraction.py`
**Description**
- Reads a single video file (`.mp4`) from a class folder in the validation set.
- Computes optical flow using NVIDIA’s CUDA-accelerated OpenCV API.
- Extracts features from each flow tensor using a BN-Inception model (pretrained on TSN).
- Saves a `.npy` file of shape `(num_frames, 1024)` to the output folder.
**Usage Example**
```bash
python scripts/01_optical_flow_extraction.py <video_name> <class_label>
```
- **`<video_name>`**: Name of the video file without extension (e.g., `video001`).
- **`<class_label>`**: Name of the folder/class in `DARPA/ecole/val` (e.g., `squat`).
**Internal Variables to Adjust**
- `input_dir = "/home/mikel.ballay/cap4773_mikel/try/DARPA/ecole/val/"`
- `output_dir = "/home/mikel.ballay/cap4773_mikel/try/DARPA_embeddings/val/"`
---
### 2. Training the Transformer Model
**Script**: `scripts/02_training_transformer.py`
**Description**
- Defines a `VideoTransformer` class (BERT-style) with positional embeddings.
- Uses `PaddedVideoEmbeddingDataset` to load `.npy` embeddings from train set.
- Trains for a specified number of epochs, calculates training accuracy, and saves best
model weights to `model.pth`.
**Usage Example**
```bash
python scripts/02_training_transformer.py \
 --train_root "/home/mikel.ballay/cap4773_mikel/try/DARPA_embeddings/train/" \
 --epochs 15 \
 --batch_size 16 \
 --learning_rate 1e-4
```
**Arguments**
- `--train_root`: Path to the `train/` folder containing subfolders per class.
- `--epochs`: Number of training epochs (default: 15).
- `--batch_size`: Batch size for DataLoader (default: 16).
- `--learning_rate`: Learning rate for AdamW optimizer (default: 1e-4).
> **Note:** Adjust `embedding_dim` (default: 1024) and `max_seq_length` (default: 200)
inside the script if your embeddings have a different dimension or length.
---
### 3. Evaluation
**Script**: `scripts/03_evaluation.py`
**Description**
- Loads the saved transformer model (`model.pth`).
- Uses `PaddedVideoEmbeddingDataset` on the validation set to generate predictions.
- Computes overall accuracy and per-class accuracy, printing results to the console.
**Usage Example**
```bash
python scripts/03_evaluation.py \
 --val_root "/home/mikel.ballay/cap4773_mikel/try/DARPA_embeddings/val/" \
 --model_path "./model.pth"
```
**Arguments**
- `--val_root`: Path to the `val/` folder containing subfolders per class.
- `--model_path`: Path to the saved `model.pth`.
**Output**
```
Overall Validation Accuracy: 15.72%
Per-Class Accuracy:
 Class 'squat': 57.14%
 Class 'put_down': 55.00%
 ...
```
---
## Results & Metrics
- **Overall Validation Accuracy**
 Computed across all classes in the validation set. Example output:
 ```
 Overall Validation Accuracy: 15.72%
 ```
- **Per-Class Accuracy**
 Computed as (correct predictions for class) / (total samples of class) × 100. Example:
 ```
 Class 'squat': 57.14%
 Class 'put_down': 55.00%
 Class 'pick_up': 12.50%
 ...
```
- **Interpretation**
 - Higher per-class accuracy indicates the transformer is able to distinguish certain
actions more reliably (e.g., ‘squat’).
 - Lower overall accuracy suggests that, despite the architecture, more data or
additional modalities (RGB frames) might be needed to improve performance.
---
## Known Limitations
1. **Missing Weights & Data**
 - The pretrained TSN weights (`TSN-flow.pth.tar`) used for BN-Inception feature
extraction are not included.
 - Original video data from `/DARPA/ecole/train` and `/DARPA_embeddings/train` is not
included here.
 - To fully reproduce, you must place:
 - Raw `.mp4` videos in `DARPA/ecole/{train,val}/{class_label}/`
 - Precomputed `.npy` embeddings in `DARPA_embeddings/{train,val}/{class_label}/`
2. **Example Files Provided**
 - An `example.npy` file in `data/` and `embeddings/` shows the format of embedding
arrays. It contains random values but demonstrates expected shape `(num_frames, 1024)`.
3. **Reproducibility**
 - If you lack the exact DARPA dataset, you can test the pipeline on a smaller toy
dataset:
 1. Create a few short `.mp4` clips labeled in subfolders.
 2. Run `01_optical_flow_extraction.py` to generate embeddings.
 3. Use those embeddings to train and evaluate.
4. **Hardware Requirements**
 - A CUDA-enabled GPU is strongly recommended for optical flow extraction and embedding
computation. CPU fallback is supported but very slow.
---
## Report & Documentation
The full research report (June 2025) is available in PDF format:
- [docs/UF_paper.pdf](docs/UF_paper.pdf)
It includes:
- Motivation and objectives
- Methodology (optical flow, embedding, transformer architecture)
- Experimental setup and hyperparameters
- Detailed results and per-class analysis
- Discussion and future directions
---
## License
This repository is provided for academic and non-commercial use only. All code, data, and
documentation are © 2025 Mikel Ballay and collaborators. If you wish to request broader
permissions, please contact **mikel.ballay@gmail.com**



