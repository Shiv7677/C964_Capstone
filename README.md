# Gene Mutation Prediction Project

## Overview
This project focuses on predicting the impact of genetic mutations and classifying them into various categories, such as neutral/non-neutral and driver/non-driver mutations. The project integrates machine learning models, genomic datasets, and analytical pipelines to provide insights into genetic mutations, which can aid in understanding their roles in diseases like cancer.

## Features
- **Mutation Status Prediction**: Classify genetic mutations as neutral or non-neutral.
- **Driver Mutation Prediction**: Classify mutations as driver or non-driver based on additional features such as Q-values and B-scores.
- **Q-Value Calculation**: Uses the Benjamini-Hochberg procedure to calculate Q-values from p-values.
- **B-Score Threshold Categorization**: Post-prediction classification into cancer drivers, potential drivers, or passengers based on B-score thresholds.

## Inputs and Outputs
### Inputs
1. **Gene Name**: Name of the gene.
2. **Mutation**: Type of mutation.
3. **Mutability**: Measure of mutation likelihood.
4. **Count**: Number of occurrences.
5. **B_Score** (optional): Calculated if not provided.

### Outputs
1. **Neutral/Non-Neutral Prediction**
2. **Driver/Non-Driver Prediction**
3. **Q-Values**
4. **Categorization**: Cancer Drivers, Potential Drivers, or Passengers

## Dataset
The project uses a finalized dataset named `combined_dataset_with_recalculated_q_values.csv`. This dataset contains the following features:
- Gene Name
- Mutation
- Mutability
- Count
- B_Score
- Q-Values

## Models
1. **Mutation Prediction Model**:
   - Classifies mutations as neutral or non-neutral.
   - Two versions available:
     - `mutation_prediction_model.pkl`: Baseline model.
     - `mutation_prediction_model_v2.pkl`: Improved model with better preprocessing and feature optimization.

2. **Driver Prediction Model**:
   - Classifies mutations as drivers or non-drivers based on additional features.

## Key Files
- **`app.py`**: Implements the integrated pipeline for predicting mutation status and performing Q-value calculations.
- **Models**:
  - `mutation_prediction_model.pkl`
  - `mutation_prediction_model_v2.pkl`
  - `mutation_impact_model.pkl`
  - `genetic_disorder_model.pkl`
- **Dataset**: `combined_dataset_with_recalculated_q_values.csv`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gene-mutation-prediction.git
   cd gene-mutation-prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have Python 3.8 or higher installed.

## Usage
1. Place your dataset in the root directory.
2. Run the application:
   ```bash
   python app.py
   ```
3. Provide the required inputs via the interface or API endpoints.

## How It Works
1. The pipeline starts with input preprocessing.
2. The mutation prediction model predicts whether the mutation is neutral or non-neutral.
3. Q-values are calculated using the Benjamini-Hochberg procedure.
4. The driver prediction model determines if the mutation is a driver or non-driver.
5. Post-prediction, mutations are categorized into cancer drivers, potential drivers, or passengers based on B-score thresholds.

## B-Score Categorization
- **Cancer Drivers**: B-score below the first threshold.
- **Potential Drivers**: B-score between the two thresholds.
- **Passengers**: B-score above the second threshold.

## Contribution
Contributions are welcome! Please open an issue or submit a pull request.


## Contact
For questions or feedback, please contact patelshiv7677@gmail.com.

