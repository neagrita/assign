# Anomaly Detection Project

A machine learning pipeline for detecting anomalies in web traffic data using Isolation Forest.

## Overview

This project implements an end-to-end anomaly detection system that processes web traffic data, extracts features, trains an Isolation Forest model, and provides prediction capabilities for identifying anomalous events.

## Setting up and running locally

### Prerequisites
- Python 3.12
- Poetry (for dependency management)
- Download the FastText model to `models/cc.en.300.bin`
- Ensure the SVD model is available at `models/svd.pkl`
- Optional: pyenv for virtual environment management

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

The challenge was approached by first performing exploratory data analysis (EDA) to understand the structure, distributions, and relationships within the dataset. This was followed by targeted feature engineering - including basic transformations such as entropy measures and frequency encoding, as well as more advanced techniques like dimensionality reduction and character-level embeddings using a [FastText](https://fasttext.cc/) model.

Given the lack of labeled data, the Isolation Forest algorithm was selected as a strong baseline for unsupervised anomaly detection, well-suited to identifying outliers in high-dimensional space.

## Project Structure

```
ddg-home-assignment/
├── data/                            # All data files
│   ├── bot-hunter-dataset-top10.tsv     # Sample input file, first 10 observations
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
