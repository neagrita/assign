# Unsupervised anomaly detection for web traffic events

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

# Install dependencies (minimal setup for running the model)
poetry install

# OR install all dependencies (including those for notebooks and development)
poetry install --with eda,dev,modelling
```

## Usage

To run the pipeline on an input file (e.g., `sample_input.tsv`):
```
poetry run python main.py data/sample_input.tsv
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

## Input Data Requirements
- File format: TSV (tab-separated values)
- Expected columns:
  - `datetime` (string, format: YYYY-MM-DD HH:MM:SS)
  - `region` (categorical)
  - `browser` (categorical)
  - `device` (categorical)
  - `url_params` (text)
- The pipeline assumes header row is not present.

## Summary on approach

The challenge was approached through a systematic methodology beginning with exploratory data analysis (EDA) to understand the underlying structure, distributions, and relationships within the dataset. This analysis was followed by comprehensive feature engineering, which included both basic transformations (entropy measures and frequency encoding) and advanced techniques (dimensionality reduction and character-level embeddings using a [FastText](https://fasttext.cc/) model).

Given the absence of labeled data, the Isolation Forest algorithm was selected as a robust baseline for unsupervised anomaly detection, as it is well-suited for identifying outliers in high-dimensional feature spaces.

Validation with KS and PSI confirmed stable performance, while UMAP visualization showed anomalies mostly on cluster edges.

## Project Structure

```
assign/
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
