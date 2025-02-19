# Santander Product Recommendation

## Overview
This project analyzes customer transaction data from the **Santander Product Recommendation** competition on Kaggle. The goal is to predict which financial products a customer will purchase in the following month using **machine learning models**, particularly **CatBoost**.

## Features
- **Data Preprocessing**: Cleansing and transforming raw data, handling missing values.
- **Feature Selection**: Using **ANOVA F-test** to identify key predictive features.
- **Exploratory Data Analysis (EDA)**: Understanding customer purchase behavior.
- **Machine Learning Modeling**:
  - **CatBoostClassifier** (Primary Model)
  - **XGBoostClassifier**
  - **RandomForestClassifier**
- **Evaluation**: Analyzing model performance and improving predictive accuracy.
- **Recommendation Output**: Generates predictions and saves results as a Kaggle submission `.csv` file.

## Dataset
The dataset consists of **1.5 years of customer transaction data**. Each row represents a customer's financial product holdings at a given month.

- Source: [Kaggle Santander Product Recommendation](https://www.kaggle.com/c/santander-product-recommendation)
- Data includes:
  - **Customer demographics** (age, gender, etc.)
  - **Banking history** (account types, transaction channels, etc.)
  - **Monthly product purchases** (binary labels for 24 product categories)

## Preprocessing Steps
- **Handling Missing Values**:
  - Replacing missing categorical values with mode or dropping columns with excessive missing values.
  - Encoding categorical variables using **LabelEncoder**.
  - Converting date fields to timestamps.
- **Defining Purchase Events**:
  - Creating a synthetic previous month’s data to track new purchases.
- **Feature Selection**:
  - Removing low-variance features.
  - Using **ANOVA F-test** to select top-performing features.

## Model Training
- **Algorithms**:
  - **CatBoostClassifier** (Primary Model)
  - **XGBoostClassifier**
  - **RandomForestClassifier**
- **Feature Selection**: Trained on the top 7 most informative features.
- **Data Split**: Using a rolling time-based split (train on previous months, predict future months).
- **Performance Improvement**:
  - Removed redundant purchase history records.
  - Fine-tuned CatBoost hyperparameters.

## Recommendation Output
- **Function: `get_recommendations()`**
  - Generates predictions for test customers.
  - Saves results in **Kaggle submission format (`ncodpers`, `added_products`)**.
  - Outputs predictions as a `.csv` file.

## Results
- **Visualization of Purchase Trends**:
  - Identified seasonal trends and stable vs. fluctuating products.
- **Model Performance**:
  - Achieved a **Kaggle leaderboard ranking of ~150**.
  - Data preprocessing improved accuracy significantly.
- **Key Findings**:
  - Gender, age, and customer activity index are strong predictors.
  - Some products have cyclical purchasing patterns (e.g., tax payments, loans).

## Installation & Usage
### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Required Libraries:
  ```bash
  pip install numpy pandas scikit-learn catboost xgboost matplotlib seaborn
  ```

### Running the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/santander-product-recommendation.git
   cd santander-product-recommendation
   ```
2. Run the Jupyter notebook:
   ```bash
   jupyter notebook
   ```
3. Execute the preprocessing and training scripts.

## Future Work
- Experiment with **LightGBM**.
- Implement deep learning approaches (e.g., **LSTMs** for sequential predictions).
- Fine-tune feature engineering for better insights.

## License
This project is for academic purposes and is based on publicly available Kaggle competition data. It does not contain or represent any real Santander customer data.

