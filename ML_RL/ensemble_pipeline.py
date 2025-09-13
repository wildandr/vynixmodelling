#!/usr/bin/env python3
"""
Ensemble Methods Pipeline for Financial Time Series Prediction
Implements stacking, bagging, and voting classifiers to improve model performance.
"""

import pandas as pd
import numpy as np
import warnings
import logging
import os
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ML Libraries
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    f1_score, accuracy_score, precision_score, recall_score
)
from sklearn.ensemble import (
    VotingClassifier, BaggingClassifier, RandomForestClassifier,
    ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_class_weight

# Imbalanced Data Handling
try:
    from imblearn.over_sampling import SMOTE
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    logging.warning("imbalanced-learn not available")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/ensemble_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

class EnsemblePipeline:
    """
    Ensemble methods pipeline for improved model performance.
    """
    
    def __init__(self, data_path: str = 'merged_labeled_data.csv'):
        """
        Initialize the ensemble pipeline.
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.ensemble_models = {}
        
        # Create directories
        os.makedirs('logs/ensemble_models', exist_ok=True)
        os.makedirs('logs/ensemble_results', exist_ok=True)
        os.makedirs('logs/ensemble_plots', exist_ok=True)
        
        logger.info("Ensemble Pipeline initialized")
    
    def load_and_prepare_data(self) -> None:
        """
        Load and prepare data for ensemble training.
        """
        logger.info("=== LOADING AND PREPARING DATA ===")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Data loaded: {self.df.shape}")
        
        # Convert date columns
        date_cols = ['decision_date', 'entry_date', 'end_date', 'datetime', 'date', 'time_converted']
        for col in date_cols:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                    logger.info(f"Converted {col} to datetime")
                except Exception as e:
                    logger.info(f"Could not convert {col} to datetime: {e}")
        
        # Prepare features
        exclude_cols = ['decision_date', 'entry_date', 'end_date', 'datetime', 'date', 'time_converted', 'label', 'time', 'barrier_touched']
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Verify columns are numeric
        verified_cols = []
        for col in feature_cols:
            try:
                test_data = self.df[col].dropna().head(5)
                if len(test_data) > 0:
                    pd.to_numeric(test_data, errors='raise')
                    verified_cols.append(col)
            except Exception:
                continue
        
        self.feature_names = verified_cols
        self.X = self.df[self.feature_names].copy()
        self.y = self.df['label'].copy()
        
        # Handle missing values
        if self.X.isnull().sum().sum() > 0:
            self.X = self.X.fillna(self.X.median())
        
        # Encode labels
        unique_labels = sorted(self.y.unique())
        if set(unique_labels) == {-1, 0, 1}:
            label_mapping = {-1: 0, 0: 1, 1: 2}
            self.y = self.y.map(label_mapping)
            logger.info("Labels remapped: -1->0, 0->1, 1->2")
        
        logger.info(f"Features prepared: {self.X.shape}")
        logger.info(f"Final label distribution: {self.y.value_counts().sort_index()}")
    
    def create_base_models(self) -> Dict:
        """
        Create base models for ensemble methods.
        """
        logger.info("=== CREATING BASE MODELS ===")
        
        # Calculate class weights for imbalanced data
        classes = np.unique(self.y)
        class_weights = compute_class_weight('balanced', classes=classes, y=self.y)
        class_weight_dict = dict(zip(classes, class_weights))
        
        base_models = {
            'xgb': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'et': ExtraTreesClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            ),
            'lr': LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000,
                n_jobs=-1
            )
        }
        
        logger.info(f"Created {len(base_models)} base models")
        return base_models
    
    def create_ensemble_models(self, base_models: Dict) -> Dict:
        """
        Create ensemble models using base models.
        """
        logger.info("=== CREATING ENSEMBLE MODELS ===")
        
        ensemble_models = {}
        
        # 1. Voting Classifier (Hard Voting)
        ensemble_models['voting_hard'] = VotingClassifier(
            estimators=[(name, model) for name, model in base_models.items()],
            voting='hard',
            n_jobs=-1
        )
        
        # 2. Voting Classifier (Soft Voting)
        # Remove SVC for soft voting as it doesn't support predict_proba by default
        soft_voting_models = {k: v for k, v in base_models.items() if k != 'svc'}
        ensemble_models['voting_soft'] = VotingClassifier(
            estimators=[(name, model) for name, model in soft_voting_models.items()],
            voting='soft',
            n_jobs=-1
        )
        
        # 3. Bagging with XGBoost
        ensemble_models['bagging_xgb'] = BaggingClassifier(
            estimator=base_models['xgb'],
            n_estimators=10,
            random_state=42,
            n_jobs=-1
        )
        
        # 4. Bagging with Random Forest
        ensemble_models['bagging_rf'] = BaggingClassifier(
            estimator=base_models['rf'],
            n_estimators=10,
            random_state=42,
            n_jobs=-1
        )
        
        # 5. Stacking Classifier
        ensemble_models['stacking'] = StackingClassifier(
            estimators=[(name, model) for name, model in base_models.items() if name != 'lr'],
            final_estimator=base_models['lr'],
            cv=3,
            n_jobs=-1
        )
        
        logger.info(f"Created {len(ensemble_models)} ensemble models")
        return ensemble_models
    
    def evaluate_models(self, models: Dict, X_train: np.ndarray, X_val: np.ndarray, 
                       y_train: np.ndarray, y_val: np.ndarray) -> Dict:
        """
        Evaluate all models and return results.
        """
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training and evaluating {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val)
                
                # Calculate metrics
                result = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'f1_macro': f1_score(y_val, y_pred, average='macro'),
                    'f1_weighted': f1_score(y_val, y_pred, average='weighted'),
                    'precision_macro': precision_score(y_val, y_pred, average='macro'),
                    'recall_macro': recall_score(y_val, y_pred, average='macro')
                }
                
                # ROC-AUC if model supports predict_proba
                try:
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_val)
                        if len(np.unique(y_val)) > 2:
                            result['roc_auc'] = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
                        else:
                            result['roc_auc'] = roc_auc_score(y_val, y_pred_proba[:, 1])
                    else:
                        result['roc_auc'] = 0.0
                except Exception:
                    result['roc_auc'] = 0.0
                
                results[name] = result
                
                logger.info(f"{name} - F1-macro: {result['f1_macro']:.4f}, Accuracy: {result['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        return results
    
    def run_ensemble_pipeline(self) -> None:
        """
        Run the complete ensemble pipeline.
        """
        logger.info("=== STARTING ENSEMBLE PIPELINE ===")
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Sort by date for time series split
        self.df = self.df.sort_values('decision_date').reset_index(drop=True)
        
        # Create time series splits
        tscv = TimeSeriesSplit(n_splits=3, gap=5)  # Fewer splits for ensemble to save time
        
        # Create base and ensemble models
        base_models = self.create_base_models()
        ensemble_models = self.create_ensemble_models(base_models)
        
        # Combine all models for evaluation
        all_models = {**base_models, **ensemble_models}
        
        # Results storage
        all_results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X)):
            logger.info(f"\n=== FOLD {fold + 1} ===")
            
            # Split data
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Handle imbalanced data with SMOTE
            if IMBLEARN_AVAILABLE:
                try:
                    smote = SMOTE(random_state=42, k_neighbors=min(5, min(np.bincount(y_train))-1))
                    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
                    logger.info(f"SMOTE applied: {X_train_scaled.shape} -> {X_train_resampled.shape}")
                except Exception as e:
                    logger.warning(f"SMOTE failed: {e}, using original data")
                    X_train_resampled, y_train_resampled = X_train_scaled, y_train
            else:
                X_train_resampled, y_train_resampled = X_train_scaled, y_train
            
            # Evaluate all models
            fold_results = self.evaluate_models(
                all_models, X_train_resampled, X_val_scaled, y_train_resampled, y_val
            )
            
            # Add fold information
            for model_name, metrics in fold_results.items():
                result_row = {'fold': fold + 1, 'model': model_name, **metrics}
                all_results.append(result_row)
        
        # Calculate average results
        results_df = pd.DataFrame(all_results)
        avg_results = results_df.groupby('model').agg({
            'accuracy': ['mean', 'std'],
            'f1_macro': ['mean', 'std'],
            'f1_weighted': ['mean', 'std'],
            'precision_macro': ['mean', 'std'],
            'recall_macro': ['mean', 'std'],
            'roc_auc': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        avg_results.columns = ['_'.join(col).strip() for col in avg_results.columns]
        
        logger.info("\n=== ENSEMBLE RESULTS SUMMARY ===")
        
        # Sort by F1-macro mean
        avg_results_sorted = avg_results.sort_values('f1_macro_mean', ascending=False)
        
        for model_name in avg_results_sorted.index:
            f1_mean = avg_results_sorted.loc[model_name, 'f1_macro_mean']
            f1_std = avg_results_sorted.loc[model_name, 'f1_macro_std']
            acc_mean = avg_results_sorted.loc[model_name, 'accuracy_mean']
            roc_mean = avg_results_sorted.loc[model_name, 'roc_auc_mean']
            
            logger.info(f"{model_name:15} - F1: {f1_mean:.4f}±{f1_std:.4f}, Acc: {acc_mean:.4f}, ROC: {roc_mean:.4f}")
        
        # Save results
        results_df.to_csv('logs/ensemble_results/detailed_results.csv', index=False)
        avg_results.to_csv('logs/ensemble_results/average_results.csv')
        
        # Find and save best model
        best_model_name = avg_results_sorted.index[0]
        best_f1_score = avg_results_sorted.loc[best_model_name, 'f1_macro_mean']
        
        logger.info(f"\nBest model: {best_model_name} (F1-macro: {best_f1_score:.4f})")
        
        # Train best model on full dataset for saving
        logger.info("Training best model on full dataset...")
        X_scaled = self.scaler.fit_transform(self.X)
        
        if IMBLEARN_AVAILABLE:
            try:
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_scaled, self.y)
            except Exception:
                X_resampled, y_resampled = X_scaled, self.y
        else:
            X_resampled, y_resampled = X_scaled, self.y
        
        best_model = all_models[best_model_name]
        best_model.fit(X_resampled, y_resampled)
        
        # Save best model
        joblib.dump(best_model, f'logs/ensemble_models/best_ensemble_model_{best_model_name}.pkl')
        joblib.dump(self.scaler, 'logs/ensemble_models/ensemble_scaler.pkl')
        
        with open('logs/ensemble_models/ensemble_feature_names.txt', 'w') as f:
            for feature in self.feature_names:
                f.write(f"{feature}\n")
        
        logger.info("Best ensemble model saved")
        logger.info("=== ENSEMBLE PIPELINE COMPLETED ===")

def main():
    """
    Main execution function.
    """
    try:
        pipeline = EnsemblePipeline()
        pipeline.run_ensemble_pipeline()
    except Exception as e:
        logger.error(f"Ensemble pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()