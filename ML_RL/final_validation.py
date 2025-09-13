#!/usr/bin/env python3
"""
Final Model Validation and Testing Pipeline
Comprehensive performance analysis and model comparison.
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    f1_score, accuracy_score, precision_score, recall_score,
    average_precision_score, roc_curve, precision_recall_curve
)

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
        logging.FileHandler('logs/final_validation.log')
    ]
)
logger = logging.getLogger(__name__)

class FinalValidationPipeline:
    """
    Final validation pipeline for comprehensive model evaluation.
    """
    
    def __init__(self, data_path: str = 'merged_labeled_data.csv'):
        """
        Initialize the final validation pipeline.
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.models = {}
        self.scalers = {}
        
        # Create directories
        os.makedirs('logs/final_validation', exist_ok=True)
        os.makedirs('logs/final_plots', exist_ok=True)
        
        logger.info("Final Validation Pipeline initialized")
    
    def load_and_prepare_data(self) -> None:
        """
        Load and prepare data for final validation.
        """
        logger.info("=== LOADING AND PREPARING DATA FOR FINAL VALIDATION ===")
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Data loaded: {self.df.shape}")
        
        # Convert date columns
        date_cols = ['decision_date', 'entry_date', 'end_date', 'datetime', 'date', 'time_converted']
        for col in date_cols:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                except Exception:
                    continue
        
        # Prepare features (same logic as training pipelines)
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
    
    def load_trained_models(self) -> None:
        """
        Load all trained models for comparison.
        """
        logger.info("=== LOADING TRAINED MODELS ===")
        
        model_paths = {
            'xgb_single': 'logs/xgb_models/best_xgb_model.pkl',
            'ensemble_bagging': 'logs/ensemble_models/best_ensemble_model_bagging_xgb.pkl'
        }
        
        scaler_paths = {
            'xgb_single': 'logs/xgb_models/scaler.pkl',
            'ensemble_bagging': 'logs/ensemble_models/ensemble_scaler.pkl'
        }
        
        for model_name, model_path in model_paths.items():
            try:
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded {model_name} model")
                else:
                    logger.warning(f"Model file not found: {model_path}")
            except Exception as e:
                logger.error(f"Error loading {model_name}: {e}")
        
        for scaler_name, scaler_path in scaler_paths.items():
            try:
                if os.path.exists(scaler_path):
                    self.scalers[scaler_name] = joblib.load(scaler_path)
                    logger.info(f"Loaded {scaler_name} scaler")
                else:
                    logger.warning(f"Scaler file not found: {scaler_path}")
            except Exception as e:
                logger.error(f"Error loading {scaler_name} scaler: {e}")
        
        logger.info(f"Loaded {len(self.models)} models and {len(self.scalers)} scalers")
    
    def create_holdout_test_set(self, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a holdout test set for final evaluation.
        """
        logger.info("=== CREATING HOLDOUT TEST SET ===")
        
        # Sort by date to ensure temporal split
        self.df = self.df.sort_values('decision_date').reset_index(drop=True)
        
        # Use temporal split - last 20% of data as test set
        split_idx = int(len(self.df) * (1 - test_size))
        
        train_df = self.df.iloc[:split_idx]
        test_df = self.df.iloc[split_idx:]
        
        X_train = train_df[self.feature_names]
        y_train = train_df['label']
        X_test = test_df[self.feature_names]
        y_test = test_df['label']
        
        # Encode labels for test set
        if set(y_train.unique()) == {-1, 0, 1}:
            label_mapping = {-1: 0, 0: 1, 1: 2}
            y_train = y_train.map(label_mapping)
            y_test = y_test.map(label_mapping)
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        logger.info(f"Train date range: {train_df['decision_date'].min()} to {train_df['decision_date'].max()}")
        logger.info(f"Test date range: {test_df['decision_date'].min()} to {test_df['decision_date'].max()}")
        
        # Check label distributions
        logger.info(f"Train label distribution: {y_train.value_counts().sort_index()}")
        logger.info(f"Test label distribution: {y_test.value_counts().sort_index()}")
        
        return X_train.values, X_test.values, y_train.values, y_test.values
    
    def evaluate_model_comprehensive(self, model, scaler, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict:
        """
        Comprehensive evaluation of a single model.
        """
        logger.info(f"=== EVALUATING {model_name.upper()} ===")
        
        # Scale test data
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
        
        # Calculate comprehensive metrics
        results = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'f1_micro': f1_score(y_test, y_pred, average='micro'),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted')
        }
        
        # ROC-AUC for multiclass
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_test)) > 2:
                    results['roc_auc_ovr'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                    results['roc_auc_ovo'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')
                else:
                    results['roc_auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC for {model_name}: {e}")
                results['roc_auc_ovr'] = 0.0
                results['roc_auc_ovo'] = 0.0
        
        # Per-class metrics
        class_report = classification_report(y_test, y_pred, output_dict=True)
        for class_label in ['0', '1', '2']:
            if class_label in class_report:
                results[f'f1_class_{class_label}'] = class_report[class_label]['f1-score']
                results[f'precision_class_{class_label}'] = class_report[class_label]['precision']
                results[f'recall_class_{class_label}'] = class_report[class_label]['recall']
        
        # Log key metrics
        logger.info(f"{model_name} Results:")
        logger.info(f"  Accuracy: {results['accuracy']:.4f}")
        logger.info(f"  F1-Macro: {results['f1_macro']:.4f}")
        logger.info(f"  F1-Weighted: {results['f1_weighted']:.4f}")
        logger.info(f"  ROC-AUC (OvR): {results.get('roc_auc_ovr', 0):.4f}")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, model_name)
        
        # Generate classification report
        report = classification_report(y_test, y_pred)
        logger.info(f"\nClassification Report for {model_name}:\n{report}")
        
        # Save detailed results
        with open(f'logs/final_validation/{model_name}_detailed_report.txt', 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Test Set Size: {len(y_test)}\n")
            f.write(f"Test Date: {datetime.now()}\n\n")
            
            f.write("=== METRICS ===\n")
            for metric, value in results.items():
                if metric != 'model_name':
                    f.write(f"{metric}: {value:.4f}\n")
            
            f.write("\n=== CLASSIFICATION REPORT ===\n")
            f.write(report)
            
            f.write("\n=== CONFUSION MATRIX ===\n")
            f.write(str(cm))
        
        return results
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str) -> None:
        """
        Plot and save confusion matrix.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Sell', 'Hold', 'Buy'],
                   yticklabels=['Sell', 'Hold', 'Buy'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'logs/final_plots/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved for {model_name}")
    
    def plot_model_comparison(self, results_list: List[Dict]) -> None:
        """
        Plot comparison of all models.
        """
        logger.info("=== CREATING MODEL COMPARISON PLOTS ===")
        
        # Prepare data for plotting
        models = [r['model_name'] for r in results_list]
        metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [r[metric] for r in results_list]
            
            axes[i].bar(models, values, alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Remove empty subplot
        axes[-1].remove()
        
        plt.tight_layout()
        plt.savefig('logs/final_plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Model comparison plot saved")
    
    def generate_final_report(self, results_list: List[Dict]) -> None:
        """
        Generate comprehensive final report.
        """
        logger.info("=== GENERATING FINAL REPORT ===")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results_list)
        
        # Save detailed results
        results_df.to_csv('logs/final_validation/final_model_comparison.csv', index=False)
        
        # Find best model
        best_model_idx = results_df['f1_macro'].idxmax()
        best_model = results_df.iloc[best_model_idx]
        
        # Generate summary report
        with open('logs/final_validation/FINAL_REPORT.txt', 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FINAL MODEL VALIDATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Test Set Size: {len(self.y)}\n\n")
            
            f.write("=== MODEL COMPARISON SUMMARY ===\n")
            for _, row in results_df.iterrows():
                f.write(f"\n{row['model_name'].upper()}:\n")
                f.write(f"  Accuracy: {row['accuracy']:.4f}\n")
                f.write(f"  F1-Macro: {row['f1_macro']:.4f}\n")
                f.write(f"  F1-Weighted: {row['f1_weighted']:.4f}\n")
                f.write(f"  Precision-Macro: {row['precision_macro']:.4f}\n")
                f.write(f"  Recall-Macro: {row['recall_macro']:.4f}\n")
                if 'roc_auc_ovr' in row:
                    f.write(f"  ROC-AUC (OvR): {row['roc_auc_ovr']:.4f}\n")
            
            f.write(f"\n=== BEST MODEL ===\n")
            f.write(f"Model: {best_model['model_name']}\n")
            f.write(f"F1-Macro Score: {best_model['f1_macro']:.4f}\n")
            f.write(f"Accuracy: {best_model['accuracy']:.4f}\n")
            
            f.write(f"\n=== RECOMMENDATIONS ===\n")
            f.write(f"1. Best performing model: {best_model['model_name']}\n")
            f.write(f"2. Key strength: F1-Macro score of {best_model['f1_macro']:.4f}\n")
            
            if best_model['f1_macro'] > 0.65:
                f.write("3. Model performance: GOOD - Ready for production consideration\n")
            elif best_model['f1_macro'] > 0.55:
                f.write("3. Model performance: MODERATE - Consider further optimization\n")
            else:
                f.write("3. Model performance: NEEDS IMPROVEMENT - Requires significant optimization\n")
            
            f.write(f"\n=== FILES GENERATED ===\n")
            f.write("- final_model_comparison.csv: Detailed metrics comparison\n")
            f.write("- confusion_matrix_*.png: Confusion matrices for each model\n")
            f.write("- model_comparison.png: Visual comparison of all models\n")
            f.write("- *_detailed_report.txt: Individual model reports\n")
        
        logger.info("Final report generated")
        logger.info(f"Best model: {best_model['model_name']} (F1-Macro: {best_model['f1_macro']:.4f})")
    
    def run_final_validation(self) -> None:
        """
        Run the complete final validation pipeline.
        """
        logger.info("=== STARTING FINAL VALIDATION PIPELINE ===")
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Load trained models
        self.load_trained_models()
        
        if not self.models:
            logger.error("No models loaded. Please train models first.")
            return
        
        # Create holdout test set
        X_train, X_test, y_train, y_test = self.create_holdout_test_set()
        
        # Evaluate all models
        results_list = []
        
        for model_name, model in self.models.items():
            if model_name in self.scalers:
                scaler = self.scalers[model_name]
                results = self.evaluate_model_comprehensive(model, scaler, X_test, y_test, model_name)
                results_list.append(results)
            else:
                logger.warning(f"No scaler found for {model_name}, skipping")
        
        if results_list:
            # Generate comparison plots
            self.plot_model_comparison(results_list)
            
            # Generate final report
            self.generate_final_report(results_list)
            
            logger.info("=== FINAL VALIDATION COMPLETED SUCCESSFULLY ===")
        else:
            logger.error("No models were successfully evaluated")

def main():
    """
    Main execution function.
    """
    try:
        pipeline = FinalValidationPipeline()
        pipeline.run_final_validation()
    except Exception as e:
        logger.error(f"Final validation pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()