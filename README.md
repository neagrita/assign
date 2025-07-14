# Anomaly Detection Project

A machine learning pipeline for detecting anomalies in web traffic data using Isolation Forest.

## Overview

This project implements an end-to-end anomaly detection system that processes web traffic data, extracts features, trains an Isolation Forest model, and provides prediction capabilities for identifying anomalous events.

## Setting up and running locally

### Prerequisites
- Python 3.12
- Poetry (for dependency management)
- FastText Model:
    - The pipeline requires the English FastText model (`cc.en.300.bin`) to be available at `models/cc.en.300.bin`.
    - You can download it from the official FastText website: https://fasttext.cc/docs/en/crawl-vectors.html.
    - After downloading, extract the model into the `models/` directory.
    - ⚠️ Note: The pipeline includes a fallback to programmatically download this model if it's missing. However, be aware that the file is 7.2 GB, so manual download is recommended.
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

TBD


## Summary on approach

The challenge was approached through a systematic methodology beginning with exploratory data analysis (EDA) to understand the underlying structure, distributions, and relationships within the dataset. This analysis was followed by comprehensive feature engineering, which included both basic transformations (entropy measures and frequency encoding) and advanced techniques (dimensionality reduction and character-level embeddings using a [FastText](https://fasttext.cc/) model).

Given the absence of labeled data, the Isolation Forest algorithm was selected as a robust baseline for unsupervised anomaly detection, as it is well-suited for identifying outliers in high-dimensional feature spaces.

## Project Structure

```
ddg-home-assignment/
├── data/                            # All data files
│   ├── sample_input.tsv.tsv             # Sample input file, first 10 observations
│   ├── output.tsv                       # Classified anomalies
├── models/                          # Model files
│   ├── cc.en.300.bin                    # FastText model (needed for feature engineering)
│   └── svd.pkl                          # SVD model (needed for feature engineering)
├── notebooks/                       # Jupyter notebooks and related files
│   ├── 00_data_prep.ipynb               # Data preparation
│   ├── 01_eda.ipynb                     # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb     # Feature engineering
│   └── 03_modelling.ipynb               # Model training and evaluation
├── transform.py                     # Feature transformation pipeline
├── predict.py                       # Prediction pipeline
├── helpers.py                       # Utility functions
├── constants.py                     # Constants and configurations
├── pyproject.toml                   # Python dependencies
└── poetry.lock                      # Locked dependencies
```
