#!/usr/bin/env python3
"""
XGBoost Training Pipeline for Financial Time Series Prediction
Implements comprehensive ML pipeline with imbalanced data handling,
hyperparameter optimization, and time series cross-validation.
"""

# Import Libraries
import pandas as pd
import numpy as np
import warnings
import logging
import os
from datetime import datetime
import joblib
from typing import Dict, List, Tuple, Optional

# ML Libraries
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, f1_score, accuracy_score,
    precision_score, recall_score, average_precision_score
)
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.utils.class_weight import compute_class_weight

# Imbalanced Data Handling
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logging.warning("imbalanced-learn not available. Install with: pip install imbalanced-learn")

# Hyperparameter Optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available. Install with: pip install optuna")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/xgb_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveXGBoostPipeline:
    """
    Comprehensive XGBoost pipeline for financial time series prediction.
    """
    
    def __init__(self, data_path: str = 'merged_labeled_data.csv'):
        """
        Initialize the XGBoost pipeline.
        
        Args:
            data_path: Path to the labeled data CSV file
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_params = None
        
        # Create directories
        os.makedirs('logs/xgb_models', exist_ok=True)
        os.makedirs('logs/xgb_results', exist_ok=True)
        os.makedirs('logs/xgb_plots', exist_ok=True)
        
        logger.info("XGBoost Pipeline initialized")
    
    def load_and_analyze_data(self) -> None:
        """
        Load and perform initial analysis of the data.
        """
        logger.info("=== PHASE 1: DATA LOADING AND ANALYSIS ===")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Data loaded: {self.df.shape}")
        
        # Convert date columns
        if 'decision_date' in self.df.columns:
            self.df['decision_date'] = pd.to_datetime(self.df['decision_date'])
            date_range = f"{self.df['decision_date'].min()} to {self.df['decision_date'].max()}"
            logger.info(f"Date range: {date_range}")
        
        logger.info(f"Columns: {len(self.df.columns)}")
        
        # Analyze labels
        if 'label' in self.df.columns:
            label_dist = self.df['label'].value_counts().sort_index()
            logger.info(f"Label distribution:\n{label_dist}")
            
            for label, count in label_dist.items():
                percentage = (count / len(self.df)) * 100
                logger.info(f"Label {label}: {count} samples ({percentage:.2f}%)")
        
        # Check for missing values
        missing_values = self.df.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Missing values found: {missing_values}")
        else:
            logger.info("No missing values found")
        
        # Analyze data types
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        logger.info(f"Numeric features: {len(numeric_cols)}")
        logger.info(f"Categorical features: {len(categorical_cols)}")
        
        # Save analysis
        analysis_results = {
            'shape': self.df.shape,
            'columns': len(self.df.columns),
            'numeric_features': len(numeric_cols),
            'categorical_features': len(categorical_cols),
            'missing_values': missing_values,
            'label_distribution': label_dist.to_dict() if 'label' in self.df.columns else None
        }
        
        with open('logs/xgb_results/data_analysis.txt', 'w') as f:
            for key, value in analysis_results.items():
                f.write(f"{key}: {value}\n")
        
        logger.info("Data analysis completed and saved")
    
    def prepare_features(self) -> None:
        """
        Prepare features and target variables for training.
        """
        logger.info("=== PHASE 2: FEATURE PREPARATION ===")
        
        # First, convert date columns to datetime to exclude them properly
        date_cols = ['decision_date', 'entry_date', 'end_date', 'datetime', 'date', 'time_converted']
        for col in date_cols:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                    logger.info(f"Converted {col} to datetime")
                except Exception as e:
                    logger.info(f"Could not convert {col} to datetime: {e}")
        
        # Exclude non-feature columns (including all date/time columns)
        exclude_cols = ['decision_date', 'entry_date', 'end_date', 'datetime', 'date', 'time_converted', 'label', 'time', 'barrier_touched']
        
        # Get only numeric columns after datetime conversion
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        logger.info(f"Found {len(numeric_cols)} numeric columns")
        
        # Remove excluded columns from numeric columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        logger.info(f"Selected {len(feature_cols)} feature columns after exclusion")
        
        # Additional verification - ensure all selected columns are truly numeric
        verified_cols = []
        for col in feature_cols:
            try:
                # Test conversion to numeric
                test_data = self.df[col].dropna().head(5)
                if len(test_data) > 0:
                    pd.to_numeric(test_data, errors='raise')
                    verified_cols.append(col)
                else:
                    logger.info(f"Skipping {col}: no valid data")
            except Exception as e:
                logger.info(f"Excluding {col}: {str(e)[:50]}")
        
        self.feature_names = verified_cols
        logger.info(f"Final verified features: {len(self.feature_names)}")
        
        # Prepare features and target
        self.X = self.df[self.feature_names].copy()
        self.y = self.df['label'].copy()
        
        logger.info(f"Features prepared: {self.X.shape}")
        logger.info(f"Target prepared: {self.y.shape}")
        logger.info(f"Feature columns: {len(self.feature_names)}")
        
        # Debug: Check data types in self.X
        logger.info(f"Data types in X: {self.X.dtypes.value_counts()}")
        
        # Debug: Check for any remaining string columns
        string_cols = self.X.select_dtypes(include=['object']).columns.tolist()
        if string_cols:
            logger.error(f"Found string columns in X: {string_cols}")
            for col in string_cols:
                logger.error(f"Sample values in {col}: {self.X[col].head(3).tolist()}")
        else:
            logger.info("No string columns found in X - all numeric!")
        
        # Handle missing values if any
        if self.X.isnull().sum().sum() > 0:
            logger.warning("Filling missing values with median")
            self.X = self.X.fillna(self.X.median())
        
        # Encode labels to ensure proper format
        unique_labels = sorted(self.y.unique())
        logger.info(f"Unique labels: {unique_labels}")
        
        # Convert to 0, 1, 2 format if needed
        if set(unique_labels) == {-1, 0, 1}:
            label_mapping = {-1: 0, 0: 1, 1: 2}
            self.y = self.y.map(label_mapping)
            logger.info("Labels remapped: -1->0, 0->1, 1->2")
        
        logger.info(f"Final label distribution: {self.y.value_counts().sort_index()}")
    
    def create_time_series_splits(self, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time series cross-validation splits with walk-forward validation.
        """
        logger.info("=== PHASE 3: TIME SERIES CROSS-VALIDATION SETUP ===")
        
        # Ensure data is sorted by date
        self.df = self.df.sort_values('decision_date').reset_index(drop=True)
        
        # Create TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=5)  # 5-day gap to prevent data leakage
        
        splits = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X)):
            train_start = self.df.iloc[train_idx[0]]['decision_date']
            train_end = self.df.iloc[train_idx[-1]]['decision_date']
            val_start = self.df.iloc[val_idx[0]]['decision_date']
            val_end = self.df.iloc[val_idx[-1]]['decision_date']
            
            logger.info(f"Fold {fold+1}: Train {train_start} to {train_end}, Val {val_start} to {val_end}")
            logger.info(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")
            
            splits.append((train_idx, val_idx))
        
        return splits
    
    def handle_imbalanced_data(self, X_train: np.ndarray, y_train: np.ndarray, 
                             method: str = 'smote') -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle imbalanced data using various sampling techniques.
        """
        if not IMBLEARN_AVAILABLE:
            logger.warning("imbalanced-learn not available, skipping resampling")
            return X_train, y_train
        
        logger.info(f"Handling imbalanced data using {method}")
        
        # Original distribution
        unique, counts = np.unique(y_train, return_counts=True)
        logger.info(f"Original distribution: {dict(zip(unique, counts))}")
        
        try:
            if method == 'smote':
                sampler = SMOTE(random_state=42, k_neighbors=min(5, min(counts)-1))
            elif method == 'adasyn':
                sampler = ADASYN(random_state=42, n_neighbors=min(5, min(counts)-1))
            elif method == 'borderline':
                sampler = BorderlineSMOTE(random_state=42, k_neighbors=min(5, min(counts)-1))
            elif method == 'smoteenn':
                sampler = SMOTEENN(random_state=42)
            elif method == 'smotetomek':
                sampler = SMOTETomek(random_state=42)
            else:
                logger.warning(f"Unknown method {method}, using SMOTE")
                sampler = SMOTE(random_state=42, k_neighbors=min(5, min(counts)-1))
            
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            
            # New distribution
            unique, counts = np.unique(y_resampled, return_counts=True)
            logger.info(f"Resampled distribution: {dict(zip(unique, counts))}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"Error in resampling: {e}")
            return X_train, y_train
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, 
                               n_trials: int = 100) -> Dict:
        """
        Optimize XGBoost hyperparameters using Optuna.
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, using default parameters")
            return {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0,
                'reg_lambda': 1
            }
        
        logger.info(f"=== PHASE 4: HYPERPARAMETER OPTIMIZATION ({n_trials} trials) ===")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Calculate class weights
            classes = np.unique(y_train)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            scale_pos_weight = class_weights[1] / class_weights[0] if len(classes) == 2 else None
            
            if scale_pos_weight:
                params['scale_pos_weight'] = scale_pos_weight
            
            model = xgb.XGBClassifier(**params)
            
            # Use cross-validation for evaluation
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, 
                                      scoring='f1_macro', n_jobs=-1)
            
            return cv_scores.mean()
        
        try:
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42),
                pruner=MedianPruner()
            )
            
            study.optimize(objective, n_trials=n_trials, timeout=1800)  # 30 minutes max
            
            logger.info(f"Best trial: {study.best_trial.number}")
            logger.info(f"Best value: {study.best_value:.4f}")
            logger.info(f"Best params: {study.best_params}")
            
            return study.best_params
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {e}")
            return {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0,
                'reg_lambda': 1
            }
    
    def train_and_evaluate(self) -> None:
        """
        Main training and evaluation pipeline.
        """
        logger.info("=== STARTING XGBOOST TRAINING PIPELINE ===")
        
        # Load and analyze data
        self.load_and_analyze_data()
        
        # Prepare features
        self.prepare_features()
        
        # Create time series splits
        splits = self.create_time_series_splits(n_splits=5)
        
        # Results storage
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            logger.info(f"\n=== FOLD {fold + 1} ===")
            
            # Split data
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Handle imbalanced data
            X_train_resampled, y_train_resampled = self.handle_imbalanced_data(
                X_train_scaled, y_train, method='smote'
            )
            
            # Optimize hyperparameters (only for first fold to save time)
            if fold == 0:
                self.best_params = self.optimize_hyperparameters(
                    X_train_resampled, y_train_resampled, n_trials=50
                )
            
            # Train model
            logger.info("=== PHASE 5: MODEL TRAINING ===")
            
            # Calculate class weights
            classes = np.unique(y_train_resampled)
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train_resampled)
            
            # Setup model parameters
            model_params = self.best_params.copy()
            model_params.update({
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'mlogloss'
            })
            
            # Train XGBoost model
            model = xgb.XGBClassifier(**model_params)
            model.fit(
                X_train_resampled, y_train_resampled,
                eval_set=[(X_val_scaled, y_val)],
                verbose=False
            )
            
            # Make predictions
            y_pred = model.predict(X_val_scaled)
            y_pred_proba = model.predict_proba(X_val_scaled)
            
            # Evaluate fold results
            fold_result = {
                'fold': fold + 1,
                'accuracy': accuracy_score(y_val, y_pred),
                'f1_macro': f1_score(y_val, y_pred, average='macro'),
                'f1_weighted': f1_score(y_val, y_pred, average='weighted'),
                'precision_macro': precision_score(y_val, y_pred, average='macro'),
                'recall_macro': recall_score(y_val, y_pred, average='macro')
            }
            
            # ROC-AUC for multiclass
            try:
                if len(np.unique(y_val)) > 2:
                    fold_result['roc_auc'] = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
                else:
                    fold_result['roc_auc'] = roc_auc_score(y_val, y_pred_proba[:, 1])
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
                fold_result['roc_auc'] = 0.0
            
            fold_results.append(fold_result)
            
            logger.info(f"Fold {fold + 1} Results:")
            for metric, value in fold_result.items():
                if metric != 'fold':
                    logger.info(f"  {metric}: {value:.4f}")
            
            # Save best model based on F1-macro
            if fold == 0 or fold_result['f1_macro'] > max([r['f1_macro'] for r in fold_results[:-1]]):
                self.best_model = model
                logger.info(f"New best model saved (F1-macro: {fold_result['f1_macro']:.4f})")
        
        # Calculate average results
        avg_results = {}
        for metric in ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro', 'roc_auc']:
            values = [r[metric] for r in fold_results]
            avg_results[f'{metric}_mean'] = np.mean(values)
            avg_results[f'{metric}_std'] = np.std(values)
        
        logger.info("\n=== AVERAGE RESULTS ACROSS FOLDS ===")
        for metric, value in avg_results.items():
            logger.info(f"{metric}: {value:.4f}")
        
        # Save results
        results_df = pd.DataFrame(fold_results)
        results_df.to_csv('logs/xgb_results/fold_results.csv', index=False)
        
        with open('logs/xgb_results/average_results.txt', 'w') as f:
            for metric, value in avg_results.items():
                f.write(f"{metric}: {value:.4f}\n")
        
        # Save final model
        if self.best_model:
            joblib.dump(self.best_model, 'logs/xgb_models/best_xgb_model.pkl')
            joblib.dump(self.scaler, 'logs/xgb_models/scaler.pkl')
            
            # Save feature names
            with open('logs/xgb_models/feature_names.txt', 'w') as f:
                for feature in self.feature_names:
                    f.write(f"{feature}\n")
            
            logger.info("Best model and scaler saved")
        
        logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")

def main():
    """
    Main execution function.
    """
    try:
        # Initialize pipeline
        pipeline = ComprehensiveXGBoostPipeline()
        
        # Run training and evaluation
        pipeline.train_and_evaluate()
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

if __name__ == "__main__":
    main()