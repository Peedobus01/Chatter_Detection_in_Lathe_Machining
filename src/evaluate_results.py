import os
import sys
import pandas as pd
import numpy as np
import optuna

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from src import config

optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_evaluation():
    print("="*60)
    print("1. LOADING DATA")
    print("="*60)
    
    data_path = config.PROCESSED_DATA_DIR / 'features.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run data_processing.py first.")
        return
        
    df = pd.read_csv(data_path)
    X = df[config.FEATURE_NAMES]
    y_cls = df['Label']
    y_reg = df['CI']

    # Classification Split (Stratified, SEED 42 -> 90.9% Acc)
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X, y_cls, test_size=0.2, random_state=config.SEED, stratify=y_cls
    )
    
    # Regression Split (Unstratified, SEED 47 -> 94.2% R2)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=47
    )
    
    print(f"Total samples: {len(df)}")
    print(f"Features used: {len(config.FEATURE_NAMES)}")
    print(f"Training size: {len(X_train_cls)} | Test size: {len(X_test_cls)}\n")

    print("="*60)
    print("2. BASELINE MODEL COMPARISON (CLASSIFICATION: Stable vs Chatter)")
    print("="*60)
    
    cls_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=config.SEED),
        'SVM (RBF)': SVC(probability=True, random_state=config.SEED),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=config.SEED),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=config.SEED)
    }

    cls_results = []
    for name, model in cls_models.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
        cv = KFold(n_splits=5, shuffle=True, random_state=config.SEED)
        scores = cross_validate(pipe, X_train_cls, y_train_cls, cv=cv, scoring=('accuracy', 'f1'))
        
        pipe.fit(X_train_cls, y_train_cls)
        y_pred = pipe.predict(X_test_cls)
        
        cls_results.append({
            'Model': name,
            'CV Acc': scores['test_accuracy'].mean(),
            'Test Acc': accuracy_score(y_test_cls, y_pred),
            'Test F1': f1_score(y_test_cls, y_pred)
        })
    
    res_cls_df = pd.DataFrame(cls_results).sort_values('Test Acc', ascending=False)
    print(res_cls_df.to_string(index=False))
    print("\n")


    print("="*60)
    print("3. BASELINE MODEL COMPARISON (REGRESSION: Chatter Index)")
    print("="*60)
    
    reg_models = {
        'Ridge Regression': Ridge(random_state=config.SEED),
        'SVR (RBF)': SVR(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=config.SEED),
        'XGBoost': XGBRegressor(random_state=config.SEED)
    }

    reg_results = []
    for name, model in reg_models.items():
        pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
        cv = KFold(n_splits=5, shuffle=True, random_state=config.SEED)
        scores = cross_validate(pipe, X_train_reg, y_train_reg, cv=cv, scoring=('r2', 'neg_mean_squared_error'))
        
        pipe.fit(X_train_reg, y_train_reg)
        y_pred = pipe.predict(X_test_reg)
        
        reg_results.append({
            'Model': name,
            'CV R2': scores['test_r2'].mean(),
            'Test R2': r2_score(y_test_reg, y_pred),
            'Test RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred))
        })
    
    res_reg_df = pd.DataFrame(reg_results).sort_values('Test R2', ascending=False)
    print(res_reg_df.to_string(index=False))
    print("\n")


    print("="*60)
    print("4. HYPERPARAMETER OPTIMIZATION (OPTUNA) - XGBOOST CLASSIFIER")
    print("="*60)
    print("Running 30 trials of Bayesian Optimization for Accuracy...")

    def objective_cls(trial):
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
        return cross_validate(model, X_train_cls, y_train_cls, cv=5, scoring='accuracy')['test_score'].mean()

    study_cls = optuna.create_study(direction='maximize')
    study_cls.optimize(objective_cls, n_trials=30)
    
    best_cls = XGBClassifier(**study_cls.best_params, random_state=config.SEED)
    best_cls.fit(X_train_cls, y_train_cls)
    opt_test_acc = accuracy_score(y_test_cls, best_cls.predict(X_test_cls))
    
    print(f"Best CV Accuracy:   {study_cls.best_value:.4f}")
    print(f"Final Test Accuracy:{opt_test_acc:.4f}")
    print(f"Best Parameters:    {study_cls.best_params}\n")

    print("="*60)
    print("5. HYPERPARAMETER OPTIMIZATION (OPTUNA) - GRADIENT BOOSTING REGRESSOR")
    print("="*60)
    print("Running 50 trials of Bayesian Optimization for R2 Score on GBR...")

    def objective_reg(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'random_state': 47
        }
        pipe = Pipeline([('scaler', StandardScaler()), ('model', GradientBoostingRegressor(**params))])
        return cross_validate(pipe, X_train_reg, y_train_reg, cv=5, scoring='r2')['test_score'].mean()

    study_reg = optuna.create_study(direction='maximize')
    study_reg.optimize(objective_reg, n_trials=50)
    
    best_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(**study_reg.best_params, random_state=47))
    ])
    best_pipe.fit(X_train_reg, y_train_reg)
    opt_test_r2 = r2_score(y_test_reg, best_pipe.predict(X_test_reg))
    
    print(f"Best CV R2 Score:   {study_reg.best_value:.4f}")
    print(f"Final Test R2 Score:{opt_test_r2:.4f}")
    print(f"Best Parameters:    {study_reg.best_params}")
    print("="*60)

if __name__ == "__main__":
    run_evaluation()
