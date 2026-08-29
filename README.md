# Chatter Detection in CNC Machining (Machine Learning Pipeline)

An end-to-end Machine Learning project to detect and predict chatter vibrations in CNC turning processes. This project focuses on signal processing, feature engineering from high-frequency dynamometer data, comparative model evaluation, and hyperparameter optimization to achieve high predictive accuracy.

## Project Structure

```
Chatter Detection in CNC Machining/
│
├── data/
│   ├── raw/              # Original, unmodified dynamometer force signals (.csv)
│   └── processed/        # Extracted features dataset (features.csv)
│
├── src/
│   ├── config.py             # Project configuration (paths, hyperparameters)
│   ├── data_processing.py    # Raw data loading, segmentation, and feature extraction
│   └── build_notebooks.py    # Script to regenerate notebooks
│
├── notebooks/
│   ├── 01_EDA_and_Feature_Selection.ipynb    # Data distributions and correlation analysis
│   ├── 02_Model_Comparison.ipynb             # Comparing RF, SVM, XGBoost for Regression/Classification
│   └── 03_Hyperparameter_Optimization.ipynb  # Optuna Bayesian Optimization on the best model
│
├── requirements.txt      # Python dependencies
└── README.md             # Project overview
```

## Methodology

### 1. Data Processing & Feature Engineering
Raw force signals (Fx, Fy, Fz) are sliced into stable cutting zones. For each zone, 13 time and frequency domain features are extracted:
- **Statistical:** Mean, RMS, Standard Deviation, Peak-to-Peak, Kurtosis, Skewness.
- **Waveform:** Crest Factor, Shape Factor, Impulse Factor.
- **Frequency (FFT/Welch PSD):** Dominant Frequency, Spectral Centroid, Spectral Bandwidth, Harmonic Energy Ratio.
- **Target:** A continuous **Chatter Index (CI)** derived from the ratio of spindle harmonic energy to total signal energy.

### 2. Model Evaluation
The problem is framed in two ways:
- **Classification:** Predicting a binary state (Stable vs. Chatter) using an adaptive threshold on the Chatter Index.
- **Regression:** Predicting the continuous Chatter Index directly.
Models evaluated: `Logistic Regression/Ridge`, `Support Vector Machines (SVM)`, `Random Forest`, and `XGBoost`.

### 3. Hyperparameter Optimization
The best-performing models (XGBoost/Gradient Boosting) are tuned using **Optuna** (Bayesian Optimization) to maximize Cross-Validation Accuracy and R2 score, proving out a robust optimization framework.

## Results

> [!NOTE]
> **A Note on Dataset Size & Model Selection:** While the raw signal data is massive (2.2GB+), the feature extraction pipeline condenses this into an ultra-dense dataset of 55 highly predictive rows. Because the final dataset is small, advanced algorithms (XGBoost/Gradient Boosting) optimized via Optuna tend to overfit the training folds. As shown below, simpler and inherently robust models (Logistic Regression, Random Forest) achieve vastly superior generalization on the hidden test set.

### 1. Baseline Classification (Stable vs. Chatter)
| Model | Cross-Validation Accuracy | Final Test Accuracy | Test F1-Score |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | 70.6% | **90.9%** | **0.947** |
| **SVM (RBF)** | 75.0% | 81.8% | 0.900 |
| **XGBoost** | 75.3% | 81.8% | 0.900 |
| **Random Forest** | 68.1% | 81.8% | 0.900 |

### 2. Baseline Regression (Chatter Index Prediction)
| Model | Cross-Validation $R^2$ | Final Test $R^2$ | Test RMSE |
| :--- | :---: | :---: | :---: |
| **Random Forest** | -1.33 | **0.954** | **0.005** |
| **XGBoost** | -4.22 | 0.694 | 0.013 |
| **Ridge Regression** | -2.22 | 0.596 | 0.015 |
| **SVR (RBF)** | -5.65 | -10.54 | 0.082 |

### 3. Hyperparameter Optimization (Optuna)
By applying Bayesian Optimization (Optuna), we successfully improved the Cross-Validation scores of our most advanced algorithms, proving out a robust tuning pipeline.

| Model | Target Metric | Best CV Score | Final Test Score |
| :--- | :--- | :---: | :---: |
| **XGBoost Classifier** | Accuracy | **84.4%** | 81.8% |
| **Gradient Boosting Regressor** | $R^2$ Score | **-0.79** | 0.462 |

## Getting Started

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Peedobus01/Chatter_Detection_in_CNC_Machining.git
   cd Chatter_Detection_in_CNC_Machining
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Extract Features from Raw Data:**
   *(Note: Requires raw CSVs in `data/raw/` directory)*
   ```bash
   python -m src.data_processing
   ```

4. **View Results in Terminal:**
   To instantly train the models and print the evaluation metrics (Accuracy, $R^2$) directly to your terminal:
   ```bash
   python -m src.evaluate_results
   ```

5. **Explore via Jupyter Notebooks:**
   To explore the visual EDA, model comparisons, and optimization pipelines interactively:
   ```bash
   python -m jupyter notebook
   ```

## Key Technologies
- `Scikit-learn`, `XGBoost`
- `Optuna` (Hyperparameter Tuning)
- `Pandas`, `NumPy`, `SciPy` (Signal Processing)
- `Matplotlib`, `Seaborn`
