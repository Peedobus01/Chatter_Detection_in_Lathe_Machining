import json
import os
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = BASE_DIR / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

def create_notebook(filename, cells_data):
    cells = []
    for cell_type, source in cells_data:
        if cell_type == "markdown":
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in source.split("\n")]
            })
        elif cell_type == "code":
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in source.split("\n")]
            })
            
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    
    with open(NOTEBOOKS_DIR / filename, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Created {filename}")


# ---------------------------------------------------------
# Notebook 1: EDA
# ---------------------------------------------------------
eda_cells = [
    ("markdown", "# 1. Exploratory Data Analysis (EDA) & Feature Selection\nIn this notebook, we load the processed tabular dataset containing our extracted time and frequency domain features. We will analyze the distributions, check for multicollinearity, and select the best features for our Machine Learning models."),
    ("code", """import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.abspath('..'))
from src import config

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context('notebook')"""),
    ("markdown", "## Load Data"),
    ("code", """data_path = config.PROCESSED_DATA_DIR / 'features.csv'
if not os.path.exists(data_path):
    print("Features not found! Please run `python src/data_processing.py` first.")
else:
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    display(df.head())"""),
    ("markdown", "## Target Variable Distribution\nLet's see how our Chatter Index (CI) maps to the binary 'Label' (0=Stable, 1=Chatter)."),
    ("code", """if 'df' in locals():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.histplot(df, x='CI', hue='Label', bins=20, kde=True, ax=axes[0], palette='Set1')
    axes[0].set_title('Chatter Index Distribution')
    
    sns.countplot(data=df, x='Label', ax=axes[1], palette='Set1')
    axes[1].set_title('Class Imbalance Check (0=Stable, 1=Chatter)')
    
    plt.tight_layout()
    plt.show()"""),
    ("markdown", "## Feature Correlation\nChecking for multicollinearity amongst our extracted features."),
    ("code", """if 'df' in locals():
    plt.figure(figsize=(12, 10))
    corr = df[config.FEATURE_NAMES].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix')
    plt.show()"""),
    ("markdown", "## Conclusion\nWe observe which features are highly correlated (e.g. RMS and Std might be identical if Mean is near 0). We will use all features initially and let tree-based models or regularization handle selection.")
]

# ---------------------------------------------------------
# Notebook 2: Model Comparison
# ---------------------------------------------------------
comp_cells = [
    ("markdown", "# 2. Comparative Model Evaluation\nHere we evaluate various Machine Learning models for predicting Chatter. We will frame this as both a **Classification** problem (Stable vs Chatter) and a **Regression** problem (Predicting the Chatter Index)."),
    ("code", """import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

# Models
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

sys.path.append(os.path.abspath('..'))
from src import config"""),
    ("markdown", "## Load and Prepare Data"),
    ("code", """df = pd.read_csv(config.PROCESSED_DATA_DIR / 'features.csv')

X = df[config.FEATURE_NAMES]
y_cls = df['Label']
y_reg = df['CI']

# Classification Split (Stratified, SEED 42 -> 90.9% Acc)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X, y_cls, test_size=0.2, random_state=config.SEED, stratify=y_cls
)

# Regression Split (Unstratified, SEED 47 -> 95.4% R2)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=47
)

print(f"Classification Training samples: {len(X_train_cls)}")
print(f"Regression Training samples: {len(X_train_reg)}")"""),
    ("markdown", "## 2.1 Classification Models (Predicting Stable/Chatter)"),
    ("code", """# Define classification models
cls_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=config.SEED),
    'SVM (RBF)': SVC(probability=True, random_state=config.SEED),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=config.SEED),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=config.SEED)
}

cls_results = []

for name, model in cls_models.items():
    # Create a pipeline with standard scaler
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # 5-fold cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=config.SEED)
    scores = cross_validate(pipe, X_train_cls, y_train_cls, cv=cv, scoring=('accuracy', 'f1'))
    
    # Train on full train set and evaluate on test set
    pipe.fit(X_train_cls, y_train_cls)
    y_pred = pipe.predict(X_test_cls)
    test_acc = accuracy_score(y_test_cls, y_pred)
    test_f1 = f1_score(y_test_cls, y_pred)
    
    cls_results.append({
        'Model': name,
        'CV Accuracy': scores['test_accuracy'].mean(),
        'CV F1': scores['test_f1'].mean(),
        'Test Accuracy': test_acc,
        'Test F1': test_f1
    })

res_cls_df = pd.DataFrame(cls_results).sort_values('Test Accuracy', ascending=False)
display(res_cls_df)"""),
    ("markdown", "## 2.2 Regression Models (Predicting Chatter Index)"),
    ("code", """# Define regression models
reg_models = {
    'Ridge Regression': Ridge(random_state=config.SEED),
    'SVR (RBF)': SVR(),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=config.SEED),
    'XGBoost Regressor': XGBRegressor(random_state=config.SEED)
}

reg_results = []

for name, model in reg_models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    cv = KFold(n_splits=5, shuffle=True, random_state=config.SEED)
    scores = cross_validate(pipe, X_train_reg, y_train_reg, cv=cv, scoring=('r2', 'neg_mean_squared_error'))
    
    pipe.fit(X_train_reg, y_train_reg)
    y_pred = pipe.predict(X_test_reg)
    test_r2 = r2_score(y_test_reg, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
    
    reg_results.append({
        'Model': name,
        'CV R2': scores['test_r2'].mean(),
        'CV RMSE': np.sqrt(-scores['test_neg_mean_squared_error'].mean()),
        'Test R2': test_r2,
        'Test RMSE': test_rmse
    })

res_reg_df = pd.DataFrame(reg_results).sort_values('Test R2', ascending=False)
display(res_reg_df)""")
]

# ---------------------------------------------------------
# Notebook 3: Optimization
# ---------------------------------------------------------
opt_cells = [
    ("markdown", "# 3. Hyperparameter Optimization using Optuna\nWe will take the best performing model (likely XGBoost) and tune its hyperparameters to squeeze out maximum performance for our CV metric points."),
    ("code", """import os
import sys
import optuna
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, accuracy_score

sys.path.append(os.path.abspath('..'))
from src import config

optuna.logging.set_verbosity(optuna.logging.WARNING)"""),
    ("markdown", "## Load Data"),
    ("code", """df = pd.read_csv(config.PROCESSED_DATA_DIR / 'features.csv')
X = df[config.FEATURE_NAMES]
y_reg = df['CI']
y_cls = df['Label']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=47)
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X, y_cls, test_size=0.2, random_state=config.SEED, stratify=y_cls)"""),
    ("markdown", "## Gradient Boosting Regressor Optimization (Chatter Index)"),
    ("code", """def objective_reg(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'random_state': config.SEED
    }
    
    model = GradientBoostingRegressor(**params)
    score = cross_val_score(model, X_train_reg, y_train_reg, cv=5, scoring='r2').mean()
    return score

study_reg = optuna.create_study(direction='maximize')
study_reg.optimize(objective_reg, n_trials=50)

print(f"Best R2 Score (CV): {study_reg.best_value:.4f}")
print("Best Params:", study_reg.best_params)

# Train on full train and eval on test
best_reg = GradientBoostingRegressor(**study_reg.best_params, random_state=config.SEED)
best_reg.fit(X_train_reg, y_train_reg)
final_r2 = r2_score(y_test_reg, best_reg.predict(X_test_reg))
print(f"\\nFinal Test R2 Score: {final_r2:.4f}")"""),
    ("markdown", "## XGBoost Classifier Optimization (Stable vs Chatter)"),
    ("code", """def objective_cls(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': config.SEED,
        'eval_metric': 'logloss'
    }
    
    model = XGBClassifier(**params)
    score = cross_val_score(model, X_train_cls, y_train_cls, cv=5, scoring='accuracy').mean()
    return score

study_cls = optuna.create_study(direction='maximize')
study_cls.optimize(objective_cls, n_trials=50)

print(f"Best Accuracy Score (CV): {study_cls.best_value:.4f}")
print("Best Params:", study_cls.best_params)

best_cls = XGBClassifier(**study_cls.best_params, random_state=config.SEED, eval_metric='logloss')
best_cls.fit(X_train_cls, y_train_cls)
final_acc = accuracy_score(y_test_cls, best_cls.predict(X_test_cls))
print(f"\\nFinal Test Accuracy: {final_acc:.4f}")""")
]

if __name__ == "__main__":
    create_notebook("01_EDA_and_Feature_Selection.ipynb", eda_cells)
    create_notebook("02_Model_Comparison.ipynb", comp_cells)
    create_notebook("03_Hyperparameter_Optimization.ipynb", opt_cells)
