# Anomaly Detection Project

A machine learning pipeline for detecting anomalies in web traffic data using Isolation Forest.

## Overview

This project processes web traffic data, performs feature engineering, and applies an Isolation Forest model for unsupervised anomaly detection. It includes data transformation, model training, and prediction with reporting of anomaly scores and key insights.

## Setting up and running locally

### Prerequisites
- Python 3.12
- Poetry (for dependency management)
- FastText Model:
    - The pipeline requires the English FastText model (`cc.en.300.bin`) to be available at `models/cc.en.300.bin`.
    - You can download it from the official FastText website: https://fasttext.cc/docs/en/crawl-vectors.html.
    - After downloading, extract the model into the `models/` directory.
- SVD Model – Ensure the `svd.pkl` file is available at `models/svd_model.pkl`.
- Optional: `pyenv` - useful for managing Python versions and virtual environments.

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd ddg-home-assignment

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

## Usage

To run the pipeline on an input file (e.g., `sample_input.tsv`):
```
poetry run python main.py sample_input.tsv
```
Output Files:
* Processed anomalies:
```bash
data/output/output_<timestamp>.tsv
```
* Report with summary stats:
```bash
data/output/output_<timestamp>_report.txt
```

## Summary on approach

The challenge was approached through a systematic methodology beginning with exploratory data analysis (EDA) to understand the underlying structure, distributions, and relationships within the dataset. This analysis was followed by comprehensive feature engineering, which included both basic transformations (entropy measures and frequency encoding) and advanced techniques (dimensionality reduction and character-level embeddings using a [FastText](https://fasttext.cc/) model).

Given the absence of labeled data, the Isolation Forest algorithm was selected as a robust baseline for unsupervised anomaly detection, as it is well-suited for identifying outliers in high-dimensional feature spaces.

KS and PSI checks showed the model generalizes well, and UMAP visualization revealed that anomalies mostly appear at the edges of clusters

## Project Structure

```
ddg-home-assignment/
├── data/                    
│   ├── sample_input.tsv                # Sample input data
│   ├── output/                         # Generated outputs
│       ├── output_<timestamp>.tsv          # Detected anomalies
│       ├── output_<timestamp>_report.txt   # Summary report
├── models/                  
│   ├── cc.en.300.bin                   # FastText embeddings
│   ├── iso_forest.pkl                  # Anomaly detection model for predictions
│   └── svd.pkl                         # SVD model for dimension reduction
├── notebooks/               
│   ├── 00_data_prep.ipynb              # Data preparation
│   ├── 01_eda.ipynb                    # EDA
│   ├── 02_feature_engineering.ipynb    # Feature engineering
│   └── 03_modelling.ipynb              # Model training & validation
├── transform.py                        # Feature transformation pipeline
├── predict.py                          # Prediction pipeline
├── helpers.py                          # Utility functions
├── constants.py                        # Configurations & constants
├── main.py                             # Entry point for execution
├── pyproject.toml                      # Dependency management
└── poetry.lock                         # Locked dependencies
```
