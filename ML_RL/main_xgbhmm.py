# Import Section
import numpy as np
import pandas as pd
import warnings
import logging
from utils.get_ticker import *
from utils.load_data import *
from fundamental_feature_engineering import *
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import time
import optuna
from triple_barrier.triplebarrier import apply_triple_barrier_labeling

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

# Dataframe Init
original_df = pd.read_csv('/root/vynixmodelling/dataset/TSLA_original.csv')

# Logging Init
# logging location: /root/vynixmodelling/ML_RL/logs
# log name format: {log_name}_{time}.log
from datetime import datetime
import os

# Create logs/main directory if it doesn't exist
os.makedirs('/root/vynixmodelling/ML_RL/logs/main', exist_ok=True)
os.makedirs('/root/vynixmodelling/ML_RL/logs/xgb_results', exist_ok=True)

# Generate datetime-based log filename
current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'/root/vynixmodelling/ML_RL/logs/main/main_{current_datetime}.log'

# Configure logging to both file and console
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Clear existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# File handler for logging to file
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Console handler for logging to terminal
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Configure for real-time logging
file_handler.stream = open(file_handler.baseFilename, 'a', buffering=1)  # Line buffering

# Preprocessing
df = original_df.copy()
logging.info(f"Original dataframe shape: {df.shape}")

# Exclude rows with NaN or null values
non_null_df = df.dropna()

# Print total rows in the dataframe without NaN or null values
logging.info(f"Total rows in df without NaN or null values: {len(non_null_df)}")

# Convert 'time' column to datetime format and print the first few rows
non_null_df['time_converted'] = pd.to_datetime(non_null_df['time'], unit='s').dt.strftime('%d%m%Y')
# logging.info(non_null_df[['time', 'time_converted']].head(1))
# logging.info(non_null_df.head)
# logging.info(non_null_df.tail)
# logging.info(non_null_df.info())
# logging.info(non_null_df.describe())
# logging.info(non_null_df.columns)

# Data Gathering with Fundamental data
# 1. Process Fundamental Data from Local Files
# Menggunakan fungsi baru untuk membaca data lokal berdasarkan CIK
pivoted_df = process_fundamental_data_local("TSLA")

# Jika gagal memuat dari lokal, fallback ke API
if pivoted_df is None:
    logging.info("Fallback to API method...")
    fundamental_data = get_fundamental_data("TSLA")
    pivoted_df = process_fundamental_data(fundamental_data, "TSLA")
    logging.info("Data loaded from API successfully")
else:
    # Baca kembali CSV yang sudah diproses untuk mendapatkan DataFrame
    pivoted_df = pd.read_csv('/root/vynixmodelling/dataset/data_fundamental/TSLA_time.csv', index_col=0)
    logging.info(f"Data loaded from local files successfully: {pivoted_df.shape}")

# 3. Fundamental Data Financial Feature Engineering
from fundamental_feature_engineering import apply_feature_engineering
enhanced_fundamental_df = apply_feature_engineering(pivoted_df, "TSLA")
# logging.info(enhanced_fundamental_df.head(1))

# 4. Teknikal and Fundamental Data Preprocessing
from technical_fundamental_preprocessing import preprocess_technical_fundamental_data

# Process data using complete pipeline
logging.info("Processing technical and fundamental data...")
output_path = '/root/vynixmodelling/dataset/main_processed_data.csv'

filtered_df = preprocess_technical_fundamental_data(
    technical_df=non_null_df,
    fundamental_df=enhanced_fundamental_df,
    output_path=output_path,
    start_period='2012-Q2',
    end_period='2025-Q2'
)

logging.info(f"Data processing completed: {filtered_df.shape}")
logging.info(f"Processed data saved to: {output_path}")

# Display final results
logging.info(f"\nFinal processed data shape: {filtered_df.shape}")
logging.info(f"Date range: {filtered_df['date'].min()} to {filtered_df['date'].max()}")
logging.info(f"Technical columns: {len([col for col in filtered_df.columns if any(tech in col.lower() for tech in ['open', 'high', 'low', 'close', 'volume'])])}")
logging.info(f"Fundamental columns: {len([col for col in filtered_df.columns if col not in ['time', 'date', 'datetime'] and not any(tech in col.lower() for tech in ['open', 'high', 'low', 'close', 'volume'])])}")
logging.info("\nPreprocessing completed successfully!")
logging.info(f'Data completeness: {(filtered_df.count().sum() / (len(filtered_df) * len(filtered_df.columns)) * 100):.2f}%')

# 5. Implementasi Labelling Triple Barrier Method ke filtered_df
# Semua parameter di init di file ini. main.py.
from triple_barrier.triplebarrier import apply_triple_barrier_labeling
from triple_barrier.visualizebarrier import generate_triple_barrier_visualizations

# Triple Barrier Method Parameters - dapat dikonfigurasi sesuai kebutuhan
TRIPLE_BARRIER_PARAMS = {
    'volatility_window': 20,           # Window untuk menghitung volatilitas
    'upper_barrier_multiplier': 1.0,   # Multiplier untuk upper barrier
    'lower_barrier_multiplier': 1.0,   # Multiplier untuk lower barrier
    'time_barrier_days': 5,            # Maksimum periode untuk menunggu barrier touch
    'verbose': True                    # Tampilkan statistik hasil
}

# Visualization Parameters
VISUALIZATION_PARAMS = {
    'output_dir': '/root/vynixmodelling/ML_RL/logs/visualization',
    'window_size': 50,                 # Jumlah data sebelum dan sesudah untuk visualisasi
    'save_html': True,                 # Simpan dalam format HTML
    'save_png': False,                  # Simpan dalam format PNG (akan menginstal kaleido otomatis)
    'verbose': True                    # Tampilkan log proses
}

logging.info("\n=== Applying Triple Barrier Method ===")
logging.info("Starting Triple Barrier Method labeling...")

# Aplikasikan Triple Barrier Method
triple_barrier_df = apply_triple_barrier_labeling(
    data=filtered_df,
    **TRIPLE_BARRIER_PARAMS
)

# Simpan hasil Triple Barrier ke CSV
triple_barrier_output_path = '/root/vynixmodelling/ML_RL/triple_barrier_results.csv'
triple_barrier_df.to_csv(triple_barrier_output_path, index=False)

# Generate visualisasi Triple Barrier
logging.info("\n=== Generating Triple Barrier Visualizations ===")
logging.info("Generating Triple Barrier visualizations...")

visualization_files = generate_triple_barrier_visualizations(
    data=filtered_df,
    triple_barrier_df=triple_barrier_df,
    **VISUALIZATION_PARAMS
)

logging.info(f"Visualizations generated: {len(visualization_files)} files")
for file_type, file_path in visualization_files.items():
    logging.info(f"{file_type}: {file_path}")

logging.info(f"\n=== Triple Barrier Implementation Complete ===")
logging.info(f"Labels generated: {len(triple_barrier_df)} samples")
logging.info(f"Results saved to: {triple_barrier_output_path}")
# logging.info(f"Visualizations saved to: {VISUALIZATION_PARAMS['output_dir']}")
# logging.info(f"Total visualization files: {len(visualization_files)}")

# Summary statistik final
# 6. Merge Triple Barrier labels dengan filtered_df berdasarkan date
# merged_label_df = triple_barrier_df.copy() + filtered_df.copy(), gunakan triple_barrier_df sebagai
# data utama yang sudah ber-label, lalu tarik data filtered_df sesuai dengan 'date' dari 'decision_date' triple_barrier_df

logging.info("\n=== Merging Triple Barrier Labels with Features ===")
logging.info("Merging triple_barrier_df with filtered_df based on decision_date...")

# Konversi decision_date ke datetime jika belum
triple_barrier_df['decision_date'] = pd.to_datetime(triple_barrier_df['decision_date'])
filtered_df['date'] = pd.to_datetime(filtered_df['date'])

# Merge berdasarkan decision_date dari triple_barrier_df dengan date dari filtered_df
merged_label_df = triple_barrier_df.merge(
    filtered_df, 
    left_on='decision_date', 
    right_on='date', 
    how='left'  # Gunakan left join untuk mempertahankan semua data triple_barrier_df
)

# Hapus kolom duplikat 'date' karena sudah ada 'decision_date'
if 'date' in merged_label_df.columns:
    merged_label_df = merged_label_df.drop('date', axis=1)

logging.info(f"Triple Barrier data shape: {triple_barrier_df.shape}")
logging.info(f"Filtered data shape: {filtered_df.shape}")
logging.info(f"Merged data shape: {merged_label_df.shape}")
logging.info(f"Successfully merged: {len(merged_label_df[merged_label_df.notna().all(axis=1)])} complete rows")

# Simpan hasil merge
merged_output_path = '/root/vynixmodelling/ML_RL/merged_labeled_data.csv'
merged_label_df.to_csv(merged_output_path, index=False)
logging.info(f"Merged labeled data saved to: {merged_output_path}")

logging.info(f"\n=== Final Summary ===")
logging.info(f"Original processed data shape: {filtered_df.shape}")
logging.info(f"Triple Barrier labels: {len(triple_barrier_df)} samples")
logging.info(f"Merged labeled data shape: {merged_label_df.shape}")
logging.info(f"Date range: {filtered_df['date'].min()} to {filtered_df['date'].max()}")
logging.info(f"Date range: {triple_barrier_df['decision_date'].min()} to {triple_barrier_df['decision_date'].max()}")
logging.info(f"Technical + Fundamental features: {len(filtered_df.columns)} columns")
logging.info(f"Total features in merged data: {len(merged_label_df.columns)} columns")
logging.info(f"Triple Barrier parameters used: {TRIPLE_BARRIER_PARAMS}")
logging.info("\nAll preprocessing, labeling, and merging completed successfully!")

# 6. Training Data Preparation.
# 6.1 Drop Column
# Dataframe yang digunakan sebagai data training: merged_label_df. kolom yang perlu di drop:
# 'decision_date', 'entry_date', 'end_date', 'end_price', 
# 'return', 'barrier_touched', 'value_at_barrier_touched', 
# 'time_converted', 'datetime','time', 'time_barrier'
# Target Col: 'label'

# Drop unnecessary columns for training
columns_to_drop = [
    'decision_date', 'entry_date', 'end_date', 'end_price',
    'return', 'barrier_touched', 'value_at_barrier_touched',
    'time_converted', 'datetime', 'time', 'time_barrier'
]

# Filter out columns that do not exist in the DataFrame
existing_columns_to_drop = [col for col in columns_to_drop if col in merged_label_df.columns]

if existing_columns_to_drop:
    training_df = merged_label_df.drop(columns=existing_columns_to_drop)
    logging.info(f"Dropped columns: {existing_columns_to_drop}")
else:
    training_df = merged_label_df.copy()
    logging.info("No specified columns to drop were found in the DataFrame.")

logging.info(f"Training DataFrame shape after dropping columns: {training_df.shape}")
logging.info(f"Training DataFrame columns: {training_df.columns.tolist()}")

# 6.2 Time-based Train Val Test Split
logging.info("\n=== Splitting Data into Train, Validation, and Test Sets (Time-based) ===")
logging.info("Splitting data into train, validation, and test sets using time-based approach...")

# First, we need to preserve decision_date for time-based splitting
# Create a copy of merged_label_df with decision_date preserved
merged_with_date = merged_label_df.copy()
merged_with_date['decision_date'] = pd.to_datetime(merged_with_date['decision_date'])

# Define Q3 2024 start date (July 1, 2024)
q3_2024_start = pd.Timestamp('2024-07-01')

# Split data based on time
test_mask = merged_with_date['decision_date'] >= q3_2024_start
train_val_mask = ~test_mask

# Create test set from Q3 2024 onwards
test_data = merged_with_date[test_mask].copy()
logging.info(f"Test data period: {test_data['decision_date'].min()} to {test_data['decision_date'].max()}")
logging.info(f"Test data shape: {test_data.shape}")

# Create train+validation set from data before Q3 2024
train_val_data = merged_with_date[train_val_mask].copy()
logging.info(f"Train+Val data period: {train_val_data['decision_date'].min()} to {train_val_data['decision_date'].max()}")
logging.info(f"Train+Val data shape: {train_val_data.shape}")

# Now drop the datetime columns for training
columns_to_drop_final = [
    'decision_date', 'entry_date', 'end_date', 'end_price',
    'return', 'barrier_touched', 'value_at_barrier_touched',
    'time_converted', 'datetime', 'time', 'time_barrier'
]

# Apply column dropping to test set
existing_cols_test = [col for col in columns_to_drop_final if col in test_data.columns]
if existing_cols_test:
    test_data_clean = test_data.drop(columns=existing_cols_test)
else:
    test_data_clean = test_data.copy()

# Apply column dropping to train+val set
existing_cols_train_val = [col for col in columns_to_drop_final if col in train_val_data.columns]
if existing_cols_train_val:
    train_val_data_clean = train_val_data.drop(columns=existing_cols_train_val)
else:
    train_val_data_clean = train_val_data.copy()

# Extract features and labels for test set
X_test = test_data_clean.drop('label', axis=1)
y_test = test_data_clean['label']

# Extract features and labels for train+val set
X_train_val = train_val_data_clean.drop('label', axis=1)
y_train_val = train_val_data_clean['label']

# Split train+val into train (80%) and validation (20%) - remove stratify for time series
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.20, random_state=42
)

# Update training_df to match the time-based split approach
training_df = train_val_data_clean.copy()

logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
logging.info(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

# Save the split data to CSV files
output_dir = '/root/vynixmodelling/dataset/training_data/'
import os
os.makedirs(output_dir, exist_ok=True)

X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
X_val.to_csv(os.path.join(output_dir, 'X_val.csv'), index=False)
y_val.to_csv(os.path.join(output_dir, 'y_val.csv'), index=False)

logging.info(f"Train, test, and validation data saved to {output_dir}")

# 7. Training
# 7.1 XGBoost. Hanya gunakan data Train dan Val dulu.
logging.info("\n=== Training XGBoost Model ===")
logging.info("Initializing and training XGBoost Classifier...")

# Check unique labels
logging.info(f"Unique labels in training data: {sorted(y_train.unique())}")
logging.info(f"Label distribution in training data: {y_train.value_counts().sort_index()}")
logging.info(f"Unique labels: {sorted(y_train.unique())}")

# Check unique labels in training data
unique_labels = sorted(y_train.unique())
logging.info(f"Unique labels found in training data: {unique_labels}")

# Create dynamic mapping based on actual labels present
if len(unique_labels) == 2:
    # Binary classification case
    label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
    reverse_label_mapping = {0: unique_labels[0], 1: unique_labels[1]}
    num_classes = 2
    objective = 'binary:logistic'
    eval_metric = 'logloss'
else:
    # Multi-class case (3 classes)
    label_mapping = {-1: 0, 0: 1, 1: 2}
    reverse_label_mapping = {0: -1, 1: 0, 2: 1}
    num_classes = 3
    objective = 'multi:softmax'
    eval_metric = 'mlogloss'

logging.info(f"Label mapping: {label_mapping}")
logging.info(f"Number of classes: {num_classes}")

# Apply mapping to training and validation labels
y_train_mapped = y_train.map(label_mapping)
y_val_mapped = y_val.map(label_mapping)

logging.info(f"Mapped labels in training data: {sorted(y_train_mapped.unique())}")
logging.info(f"Mapped label distribution: {y_train_mapped.value_counts().sort_index()}")

# Initialize XGBoost Classifier with dynamic configuration
xgb_model = XGBClassifier(
    objective=objective,
    num_class=num_classes if num_classes > 2 else None,  # Only set for multi-class
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric=eval_metric,
    random_state=42
)

# Train the model
logging.info("Training XGBoost model...")
xgb_model.fit(X_train, y_train_mapped,
              eval_set=[(X_train, y_train_mapped), (X_val, y_val_mapped)],
              verbose=True)

logging.info("XGBoost Model training completed.")

# Evaluate on Validation Set
logging.info("Evaluating XGBoost Model on validation set...")

# Make predictions using mapped labels
y_pred_val_mapped = xgb_model.predict(X_val)
y_proba_val = xgb_model.predict_proba(X_val)

# Convert predictions back to original labels
y_pred_val = pd.Series(y_pred_val_mapped).map(reverse_label_mapping)

logging.info(f"Predicted labels (mapped): {sorted(y_pred_val_mapped)}")
logging.info(f"Predicted labels (original): {sorted(y_pred_val.unique())}")
logging.info(f"Validation labels (original): {sorted(y_val.unique())}")

# Calculate metrics using original labels
accuracy_val = accuracy_score(y_val, y_pred_val)
precision_val = precision_score(y_val, y_pred_val, average='weighted')
recall_val = recall_score(y_val, y_pred_val, average='weighted')
f1_val = f1_score(y_val, y_pred_val, average='weighted')

# Calculate ROC AUC based on number of classes
if num_classes == 2:
    # For binary classification, use probability of positive class
    y_val_mapped_for_roc = y_val.map(label_mapping)
    roc_auc_val = roc_auc_score(y_val_mapped_for_roc, y_proba_val[:, 1])
else:
    # For multi-class, use all probabilities
    y_val_mapped_for_roc = y_val.map(label_mapping)
    roc_auc_val = roc_auc_score(y_val_mapped_for_roc, y_proba_val, multi_class='ovr', average='weighted')

# Display results
logging.info(f"Validation Accuracy: {accuracy_val:.4f}")
logging.info(f"Validation Precision: {precision_val:.4f}")
logging.info(f"Validation Recall: {recall_val:.4f}")
logging.info(f"Validation F1-Score: {f1_val:.4f}")
logging.info(f"Validation ROC AUC: {roc_auc_val:.4f}")

logging.info(f"Validation Accuracy: {accuracy_val:.4f}")
logging.info(f"Validation Precision: {precision_val:.4f}")
logging.info(f"Validation Recall: {recall_val:.4f}")
logging.info(f"Validation F1-Score: {f1_val:.4f}")
logging.info(f"Validation ROC AUC: {roc_auc_val:.4f}")

# Display confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
logging.info("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_val, y_pred_val)
logging.info(f"Confusion Matrix:\n{cm}")
logging.info(f"\nClassification Report:\n{classification_report(y_val, y_pred_val)}")

logging.info("XGBoost Model evaluation on validation set completed.")

# 7.2 Evaluate on Training Set for comparison
logging.info("\n=== Evaluating XGBoost Model on Training Set ===")
logging.info("Evaluating XGBoost Model on training set...")

# Make predictions on training set
y_pred_train_mapped = xgb_model.predict(X_train)
y_pred_train = pd.Series(y_pred_train_mapped).map(reverse_label_mapping)

# Calculate training metrics
accuracy_train = accuracy_score(y_train, y_pred_train)
precision_train = precision_score(y_train, y_pred_train, average='weighted')
recall_train = recall_score(y_train, y_pred_train, average='weighted')
f1_train = f1_score(y_train, y_pred_train, average='weighted')

logging.info(f"Training Accuracy: {accuracy_train:.4f}")
logging.info(f"Training Precision: {precision_train:.4f}")
logging.info(f"Training Recall: {recall_train:.4f}")
logging.info(f"Training F1-Score: {f1_train:.4f}")

# 7.3 Save the trained model
logging.info("\n=== Saving XGBoost Model ===")
logging.info("Saving trained XGBoost model...")

import joblib
import os

# Create model directory
model_dir = '/root/vynixmodelling/ML_RL/logs/xgb_models/'
os.makedirs(model_dir, exist_ok=True)

# Save model
model_path = os.path.join(model_dir, 'xgb_model_trained.pkl')
joblib.dump(xgb_model, model_path)

# Save label mappings
mapping_path = os.path.join(model_dir, 'label_mappings.pkl')
joblib.dump({
    'label_mapping': label_mapping,
    'reverse_label_mapping': reverse_label_mapping
}, mapping_path)

logging.info(f"Model saved to: {model_path}")
logging.info(f"Label mappings saved to: {mapping_path}")

# 7.4 Feature Importance Analysis
logging.info("Analyzing feature importance...")

# Get feature importance
feature_importance = xgb_model.feature_importances_
feature_names = X_train.columns

# Create feature importance dataframe
import pandas as pd
feature_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Display top 20 most important features
logging.info("Top 20 Most Important Features:")
logging.info(feature_importance_df.head(20))

# Save feature importance
feature_importance_path = os.path.join(model_dir, 'feature_importance.csv')
feature_importance_df.to_csv(feature_importance_path, index=False)
logging.info(f"Feature importance saved to: {feature_importance_path}")

logging.info("XGBoost training and evaluation completed successfully.")

# 7.5 Testing model menggunakan data test dengan inferensi menggunakan model yang sudah dibuat sebelumnya
logging.info("Testing XGBoost Model on test set...")

# Make predictions on test set using the trained model
y_pred_test_mapped = xgb_model.predict(X_test)
y_proba_test = xgb_model.predict_proba(X_test)

# Convert predictions back to original labels
y_pred_test = pd.Series(y_pred_test_mapped).map(reverse_label_mapping)

logging.info(f"Test set size: {len(X_test)} samples")
logging.info(f"Predicted labels (mapped): {sorted(set(y_pred_test_mapped))}")
logging.info(f"Predicted labels (original): {sorted(y_pred_test.unique())}")
logging.info(f"Test labels (original): {sorted(y_test.unique())}")

# Calculate test metrics using original labels
accuracy_test = accuracy_score(y_test, y_pred_test)
precision_test = precision_score(y_test, y_pred_test, average='weighted')
recall_test = recall_score(y_test, y_pred_test, average='weighted')
f1_test = f1_score(y_test, y_pred_test, average='weighted')

# Calculate ROC AUC for test set based on number of classes
if num_classes == 2:
    # For binary classification, use probability of positive class
    y_test_mapped_for_roc = y_test.map(label_mapping)
    roc_auc_test = roc_auc_score(y_test_mapped_for_roc, y_proba_test[:, 1])
else:
    # For multi-class, use all probabilities
    y_test_mapped_for_roc = y_test.map(label_mapping)
    roc_auc_test = roc_auc_score(y_test_mapped_for_roc, y_proba_test, multi_class='ovr', average='weighted')

# Display test results
logging.info(f"Test Accuracy: {accuracy_test:.4f}")
logging.info(f"Test Precision: {precision_test:.4f}")
logging.info(f"Test Recall: {recall_test:.4f}")
logging.info(f"Test F1-Score: {f1_test:.4f}")
logging.info(f"Test ROC AUC: {roc_auc_test:.4f}")

# Display test confusion matrix
logging.info("\n=== Test Set Confusion Matrix ===")
cm_test = confusion_matrix(y_test, y_pred_test)
logging.info(f"Test Confusion Matrix:\n{cm_test}")
logging.info(f"\nTest Classification Report:\n{classification_report(y_test, y_pred_test)}")

# Compare performance across all sets
logging.info("\n=== Performance Comparison Across All Sets ===")
performance_comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
    'Training': [accuracy_train, precision_train, recall_train, f1_train, 0.0],  # ROC AUC not calculated for training
    'Validation': [accuracy_val, precision_val, recall_val, f1_val, roc_auc_val],
    'Test': [accuracy_test, precision_test, recall_test, f1_test, roc_auc_test]
})

logging.info(performance_comparison.round(4))

# Save performance comparison
performance_path = os.path.join(model_dir, 'performance_comparison.csv')
performance_comparison.to_csv(performance_path, index=False)
logging.info(f"Performance comparison saved to: {performance_path}")

# Save test predictions for further analysis
test_predictions_dict = {
    'actual_label': y_test.values,
    'predicted_label': y_pred_test.values,
}

# Add probability columns based on number of classes
if num_classes == 2:
    test_predictions_dict['probability_class_0'] = y_proba_test[:, 0]  # Probability for first class
    test_predictions_dict['probability_class_1'] = y_proba_test[:, 1]  # Probability for second class
else:
    test_predictions_dict['probability_class_0'] = y_proba_test[:, 0]  # Probability for class -1 (mapped to 0)
    test_predictions_dict['probability_class_1'] = y_proba_test[:, 1]  # Probability for class 0 (mapped to 1)
    test_predictions_dict['probability_class_2'] = y_proba_test[:, 2]  # Probability for class 1 (mapped to 2)

test_predictions_df = pd.DataFrame(test_predictions_dict)

# Add prediction confidence (max probability)
test_predictions_df['prediction_confidence'] = y_proba_test.max(axis=1)

# Add correct/incorrect prediction flag
test_predictions_df['correct_prediction'] = (test_predictions_df['actual_label'] == test_predictions_df['predicted_label'])

# Save test predictions
test_predictions_path = os.path.join(model_dir, 'test_predictions.csv')
test_predictions_df.to_csv(test_predictions_path, index=False)
logging.info(f"Test predictions saved to: {test_predictions_path}")

# Analyze prediction confidence
logging.info("\n=== Prediction Confidence Analysis ===")
logging.info(f"Average prediction confidence: {test_predictions_df['prediction_confidence'].mean():.4f}")
logging.info(f"Confidence for correct predictions: {test_predictions_df[test_predictions_df['correct_prediction']]['prediction_confidence'].mean():.4f}")
logging.info(f"Confidence for incorrect predictions: {test_predictions_df[~test_predictions_df['correct_prediction']]['prediction_confidence'].mean():.4f}")

# Show prediction distribution
logging.info("\n=== Test Set Prediction Distribution ===")
logging.info("Actual vs Predicted Label Distribution:")
logging.info(pd.crosstab(y_test, y_pred_test, margins=True))

logging.info("XGBoost model testing on test set completed successfully.")

# 8. Hyperparameter Tuning. Hanya gunakan data Train dan Val terlebih dahulu saja.
logging.info("\n" + "="*80)
logging.info("8. HYPERPARAMETER TUNING")
logging.info("="*80)

# Import libraries for hyperparameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import time
import os
from datetime import datetime
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.info("Warning: Plotly not available. Visualization will be skipped.")
import warnings
warnings.filterwarnings('ignore')

# 8.1 Hyperparameter pada parameter-parameter Model XGboost saja.
logging.info("\n8.1 XGBoost Hyperparameter Tuning with Optuna")
logging.info("-" * 50)

# Define objective function for Optuna
def xgb_objective(trial):
    # Suggest hyperparameters with optimal distributions
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5, log=False),
        'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear']),
        'tree_method': trial.suggest_categorical('tree_method', ['auto', 'exact', 'approx', 'hist'])
    }
    
    # Set objective and eval_metric based on number of classes
    if num_classes == 2:
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'logloss'
    else:
        params['objective'] = 'multi:softmax'
        params['num_class'] = num_classes
        params['eval_metric'] = 'mlogloss'
    
    params['random_state'] = 42
    params['n_jobs'] = -1
    params['verbosity'] = 0  # Suppress XGBoost output
    
    # Create and train model
    model = XGBClassifier(**params)
    
    # Use cross-validation for robust evaluation
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(
        model, X_train, y_train_mapped, 
        cv=3, scoring='accuracy', n_jobs=-1
    )
    
    # Return mean CV score
    return cv_scores.mean()

# Enhanced progress callback class with early stopping and comprehensive logging
class EnhancedProgressCallback:
    def __init__(self, study_name="Hyperparameter Tuning", patience=10, min_improvement=0.001):
        self.study_name = study_name
        self.start_time = time.time()
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_score = float('-inf')
        self.trials_without_improvement = 0
        self.trial_scores = []
        
    def __call__(self, study, trial):
        elapsed_time = time.time() - self.start_time
        current_score = trial.value if trial.value is not None else 0.0
        self.trial_scores.append(current_score)
        
        # Check for improvement
        if current_score > self.best_score + self.min_improvement:
            self.best_score = current_score
            self.trials_without_improvement = 0
        else:
            self.trials_without_improvement += 1
        
        # Progress display
        avg_score = sum(self.trial_scores[-5:]) / min(5, len(self.trial_scores))  # Last 5 trials average
        logging.info(f"{self.study_name} - Trial {trial.number + 1}: Current = {current_score:.4f}, Best = {study.best_value:.4f}, Avg(5) = {avg_score:.4f}, Time = {elapsed_time:.1f}s")
        
        # Early stopping check
        if self.trials_without_improvement >= self.patience:
            logging.info(f"\n\nEarly stopping triggered after {self.patience} trials without improvement >= {self.min_improvement}")
            study.stop()
        
        # Log significant improvements
        if trial.number > 0 and current_score == study.best_value:
            logging.info(f"\n*** New best score achieved: {current_score:.4f} ***")
            logging.info(f"Parameters: {trial.params}")

# Create Optuna study with pruning
logging.info("Starting XGBoost hyperparameter tuning with Optuna...")
start_time = time.time()

# Create study with persistent storage
storage = optuna.storages.RDBStorage(url="sqlite:///xgboost_study.db")
study = optuna.create_study(
    study_name="xgboost_hyperparameters",
    direction="maximize",
    storage=storage,
    load_if_exists=True
)

# Add enhanced progress callback with early stopping
xgb_callback = EnhancedProgressCallback(
    study_name="XGBoost Hyperparameters", 
    patience=20, 
    min_improvement=0.002
)

# Optimize with enhanced callbacks
study.optimize(
    xgb_objective, 
    n_trials=100,  # More trials for better optimization
    callbacks=[xgb_callback],
    show_progress_bar=True
)

# Create visualizations
# visualize_study_results(study, "xgboost_hyperparameters")  # Function not defined, commented out

xgb_tuning_time = time.time() - start_time
logging.info(f"XGBoost tuning completed in {xgb_tuning_time:.2f} seconds")

# Get best parameters and score
best_xgb_params = study.best_params
best_xgb_score = study.best_value

logging.info(f"\nBest XGBoost Parameters:")
for param, value in best_xgb_params.items():
    logging.info(f"  {param}: {value}")
logging.info(f"Best Cross-Validation Score: {best_xgb_score:.4f}")
logging.info(f"Number of trials: {len(study.trials)}")
logging.info(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")

# Train best XGBoost model with optimal parameters
if num_classes == 2:
    best_xgb_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        **best_xgb_params
    )
else:
    best_xgb_model = XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=-1,
        **best_xgb_params
    )

best_xgb_model.fit(X_train, y_train_mapped)

# Evaluate on validation set
y_val_pred_tuned = best_xgb_model.predict(X_val)
y_val_proba_tuned = best_xgb_model.predict_proba(X_val)

# Calculate validation metrics for tuned model
val_accuracy_tuned = accuracy_score(y_val_mapped, y_val_pred_tuned)
val_precision_tuned = precision_score(y_val_mapped, y_val_pred_tuned, average='weighted')
val_recall_tuned = recall_score(y_val_mapped, y_val_pred_tuned, average='weighted')
val_f1_tuned = f1_score(y_val_mapped, y_val_pred_tuned, average='weighted')

if num_classes == 2:
    val_roc_auc_tuned = roc_auc_score(y_val_mapped, y_val_proba_tuned[:, 1])
else:
    val_roc_auc_tuned = roc_auc_score(y_val_mapped, y_val_proba_tuned, multi_class='ovr')

logging.info(f"\nTuned XGBoost Validation Performance:")
logging.info(f"Accuracy: {val_accuracy_tuned:.4f}")
logging.info(f"Precision: {val_precision_tuned:.4f}")
logging.info(f"Recall: {val_recall_tuned:.4f}")
logging.info(f"F1-Score: {val_f1_tuned:.4f}")
logging.info(f"ROC AUC: {val_roc_auc_tuned:.4f}")

# Compare with original model (use validation accuracy from section 7.4)
logging.info(f"\nComparison with Original Model:")
original_val_accuracy = 0.6295  # From previous validation results
logging.info(f"Original Validation Accuracy: {original_val_accuracy:.4f}")
logging.info(f"Tuned Validation Accuracy: {val_accuracy_tuned:.4f}")
logging.info(f"Improvement: {val_accuracy_tuned - original_val_accuracy:.4f}")

# Save tuned XGBoost model
tuned_xgb_model_path = 'logs/xgb_models/tuned_xgb_model.pkl'
joblib.dump(best_xgb_model, tuned_xgb_model_path)
logging.info(f"\nTuned XGBoost model saved to: {tuned_xgb_model_path}")

# Save XGBoost tuning feature importance
xgb_tuning_feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': best_xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

xgb_tuning_feature_importance_path = 'logs/xgb_models/feature_importance_xgb_tuning.csv'
xgb_tuning_feature_importance.to_csv(xgb_tuning_feature_importance_path, index=False)
logging.info(f"XGBoost tuning feature importance saved to: {xgb_tuning_feature_importance_path}")

# Print XGBoost tuning feature importance
logging.info("\n=== XGBoost Tuning Feature Importance ===")
logging.info(xgb_tuning_feature_importance.head(20))
logging.info(f"\nFull feature importance saved to: {xgb_tuning_feature_importance_path}")

# 8.2 Melakukan tuning juga pada volatility_window, upper_barrier_multiplier, lower_barrier_multiplier, time_barrier_days.
logging.info("\n8.2 Combined XGBoost and Barrier Parameters Tuning with Optuna")
logging.info("-" * 50)

# Define objective function for combined XGBoost and Barrier parameters
def combined_objective(trial):
    # XGBoost hyperparameters
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0, step=0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.1),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0, step=0.1),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0, step=0.1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0, step=0.1)
    }
    
    # Barrier parameters - For 1:1 risk-reward ratio, upper and lower multipliers must be equal
    barrier_multiplier = trial.suggest_float('barrier_multiplier', 0.5, 4.0, step=0.1)
    
    barrier_params = {
        'volatility_window': trial.suggest_int('volatility_window', 5, 60, step=5),
        'upper_barrier_multiplier': barrier_multiplier,  # Same value for 1:1 risk-reward
        'lower_barrier_multiplier': barrier_multiplier,  # Same value for 1:1 risk-reward
        'time_barrier_days': trial.suggest_int('time_barrier_days', 3, 28),
        'verbose': False
    }
    
    try:
        # Apply triple barrier labeling with current parameters
        temp_triple_barrier_df = apply_triple_barrier_labeling(
            data=filtered_df,
            **barrier_params
        )
        
        # Merge with filtered_df
        temp_triple_barrier_df['decision_date'] = pd.to_datetime(temp_triple_barrier_df['decision_date'])
        temp_merged_df = temp_triple_barrier_df.merge(
            filtered_df, 
            left_on='decision_date', 
            right_on='date', 
            how='left'
        )
        
        if 'date' in temp_merged_df.columns:
            temp_merged_df = temp_merged_df.drop('date', axis=1)
        
        # Drop unnecessary columns for training
        temp_columns_to_drop = [
            'decision_date', 'entry_date', 'end_date', 'end_price',
            'return', 'barrier_touched', 'value_at_barrier_touched',
            'time_converted', 'datetime', 'time', 'time_barrier'
        ]
        temp_existing_columns_to_drop = [col for col in temp_columns_to_drop if col in temp_merged_df.columns]
        
        if temp_existing_columns_to_drop:
            df_with_labels_new = temp_merged_df.drop(columns=temp_existing_columns_to_drop)
        else:
            df_with_labels_new = temp_merged_df.copy()
        
        # Remove rows with NaN labels
        df_with_labels_new = df_with_labels_new.dropna(subset=['label'])
        
        if len(df_with_labels_new) < 100:  # Skip if too few samples
            return 0.0  # Return poor score for pruning
        
        # Apply time-based split to hyperparameter tuning data (same as initial approach)
        merged_new_df = merged_label_df.copy()
        merged_new_df['decision_date'] = pd.to_datetime(merged_new_df['decision_date'])
        
        # Define Q3 2024 start date (July 1, 2024) - same as initial split
        q3_2024_start_new = pd.Timestamp('2024-07-01')
        
        # Split data based on time (same logic as initial split)
        test_mask_new = merged_new_df['decision_date'] >= q3_2024_start_new
        train_val_mask_new = ~test_mask_new
        
        # Create test set from Q3 2024 onwards
        test_data_new = merged_new_df[test_mask_new].copy()
        
        # Create train+validation set from data before Q3 2024
        train_val_data_new = merged_new_df[train_val_mask_new].copy()
        
        # Drop datetime columns for training
        columns_to_drop_new = [
            'decision_date', 'entry_date', 'end_date', 'end_price',
            'return', 'barrier_touched', 'value_at_barrier_touched',
            'time_converted', 'datetime', 'time', 'time_barrier'
        ]
        
        # Apply column dropping to test set
        existing_cols_test_new = [col for col in columns_to_drop_new if col in test_data_new.columns]
        if existing_cols_test_new:
            test_data_clean_new = test_data_new.drop(columns=existing_cols_test_new)
        else:
            test_data_clean_new = test_data_new.copy()
        
        # Apply column dropping to train+val set
        existing_cols_train_val_new = [col for col in columns_to_drop_new if col in train_val_data_new.columns]
        if existing_cols_train_val_new:
            train_val_data_clean_new = train_val_data_new.drop(columns=existing_cols_train_val_new)
        else:
            train_val_data_clean_new = train_val_data_new.copy()
        
        # Extract features and labels for test set
        X_test_new = test_data_clean_new.drop('label', axis=1)
        y_test_new = test_data_clean_new['label']
        
        # Extract features and labels for train+val set
        X_train_val_new = train_val_data_clean_new.drop('label', axis=1)
        y_train_val_new = train_val_data_clean_new['label']
        
        # Split train+val into train (80%) and validation (20%) - remove stratify for time series
        X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(
            X_train_val_new, y_train_val_new, test_size=0.20, random_state=42
        )
        
        # Map labels
        unique_labels_new = sorted(y_train_new.unique())
        num_classes_new = len(unique_labels_new)
        
        if num_classes_new == 2:
            label_mapping_new = {unique_labels_new[0]: 0, unique_labels_new[1]: 1}
        else:
            label_mapping_new = {label: idx for idx, label in enumerate(unique_labels_new)}
        
        y_train_mapped_new = y_train_new.map(label_mapping_new)
        y_val_mapped_new = y_val_new.map(label_mapping_new)
        
        # Train model with suggested XGBoost parameters
        if num_classes_new == 2:
            temp_model = XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                **xgb_params
            )
        else:
            temp_model = XGBClassifier(
                objective='multi:softmax',
                num_class=num_classes_new,
                eval_metric='mlogloss',
                random_state=42,
                **xgb_params
            )
        
        temp_model.fit(X_train_new, y_train_mapped_new)
        
        # Evaluate on validation set
        y_val_pred_new = temp_model.predict(X_val_new)
        val_accuracy_new = accuracy_score(y_val_mapped_new, y_val_pred_new)
        
        return val_accuracy_new
        
    except Exception as e:
        # Return poor score for failed trials
        return 0.0

def create_study_with_storage(study_name, direction='maximize'):
    """Create Optuna study with persistent storage"""
    # Create studies directory if it doesn't exist
    import os
    os.makedirs('logs/optuna_studies', exist_ok=True)
    
    # Create database path
    db_path = f'logs/optuna_studies/{study_name}.db'
    storage = f'sqlite:///{db_path}'
    
    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True
    )
    
    logging.info(f"Study '{study_name}' created/loaded with storage: {db_path}")
    return study

# Create Optuna study for Triple Barrier parameters
logging.info("Starting combined XGBoost and Barrier parameters optimization with Optuna...")
start_time = time.time()

# Create Optuna study for combined parameters
combined_study = create_study_with_storage(
    study_name="combined_xgb_barrier_parameters",
    direction='maximize'
)

# Create callback for early stopping and progress tracking
combined_callback = EnhancedProgressCallback(study_name="Combined XGBoost and Barrier Parameters", patience=15, min_improvement=0.005)

# Optimize combined parameters
combined_study.optimize(
    combined_objective, 
    n_trials=100,  # Increased for better parameter space exploration
    callbacks=[combined_callback],
    show_progress_bar=True
)

# Calculate tuning time
combined_tuning_time = time.time() - start_time
logging.info(f"\nCombined tuning completed in {combined_tuning_time:.2f} seconds")

# Get best combined parameters
best_combined_params = combined_study.best_params
best_combined_score = combined_study.best_value

# Separate XGBoost and barrier parameters
best_xgb_params_combined = {
    'n_estimators': best_combined_params['n_estimators'],
    'max_depth': best_combined_params['max_depth'],
    'learning_rate': best_combined_params['learning_rate'],
    'subsample': best_combined_params['subsample'],
    'colsample_bytree': best_combined_params['colsample_bytree'],
    'reg_alpha': best_combined_params['reg_alpha'],
    'reg_lambda': best_combined_params['reg_lambda'],
    'min_child_weight': best_combined_params['min_child_weight'],
    'gamma': best_combined_params['gamma']
}

best_barrier_params = {
    'volatility_window': best_combined_params['volatility_window'],
    'upper_barrier_multiplier': best_combined_params['barrier_multiplier'],
    'lower_barrier_multiplier': best_combined_params['barrier_multiplier'],
    'time_barrier_days': best_combined_params['time_barrier_days']
}

logging.info(f"\nBest Combined XGBoost Parameters:")
for param, value in best_xgb_params_combined.items():
    logging.info(f"  {param}: {value}")

logging.info(f"\nBest Combined Barrier Parameters:")
for param, value in best_barrier_params.items():
    logging.info(f"  {param}: {value}")
logging.info(f"Best Combined Validation Score: {best_combined_score:.4f}")

# Compare with original model
logging.info(f"\nComparison with Original Model:")
logging.info(f"Original Validation Accuracy: 0.6295")
logging.info(f"XGBoost Only Tuned Validation Accuracy: {best_xgb_score:.4f}")
logging.info(f"Combined Tuned Validation Accuracy: {best_combined_score:.4f}")
logging.info(f"Improvement over original: {best_combined_score - 0.6295:.4f}")
logging.info(f"Improvement over XGBoost only: {best_combined_score - best_xgb_score:.4f}")

# Create the best combined model using the best parameters
# Since barrier parameters don't create a model object, we save the parameters instead
best_combined_model = {
    'xgb_params': best_xgb_params_combined,
    'barrier_params': best_barrier_params
}

# Save best combined model
best_combined_model_path = 'logs/xgb_models/best_combined_model.pkl'
joblib.dump(best_combined_model, best_combined_model_path)
logging.info(f"\nBest combined model parameters saved to: {best_combined_model_path}")

# Create and train the final combined model to get feature importance
if num_classes == 2:
    combined_model_for_importance = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        **best_xgb_params_combined
    )
else:
    combined_model_for_importance = XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes,
        eval_metric='mlogloss',
        random_state=42,
        **best_xgb_params_combined
    )

# Train the model to get feature importance
combined_model_for_importance.fit(X_train, y_train_mapped)

# Save barrier tuning feature importance
barrier_tuning_feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': combined_model_for_importance.feature_importances_
}).sort_values('importance', ascending=False)

barrier_tuning_feature_importance_path = 'logs/xgb_models/feature_importance_barrier_tuning.csv'
barrier_tuning_feature_importance.to_csv(barrier_tuning_feature_importance_path, index=False)
logging.info(f"Barrier tuning feature importance saved to: {barrier_tuning_feature_importance_path}")

# Print barrier tuning feature importance
logging.info("\n=== Barrier Tuning Feature Importance ===")
logging.info(barrier_tuning_feature_importance.head(20))
logging.info(f"\nFull feature importance saved to: {barrier_tuning_feature_importance_path}")

# Save best parameters
best_params_final = {
    'xgb_params_only': best_xgb_params,
    'xgb_params_combined': best_xgb_params_combined,
    'barrier_params': best_barrier_params,
    'best_xgb_only_score': best_xgb_score,
    'best_combined_score': best_combined_score
}

best_params_path = 'logs/xgb_results/best_hyperparameters.pkl'
joblib.dump(best_params_final, best_params_path)
logging.info(f"Best hyperparameters saved to: {best_params_path}")

# 8.3 Melakukan testing dataset test dengan model yang disimpan dari proses 8.1 dan 8.2
logging.info("\n8.3 Final Testing with Best Hyperparameters")
logging.info("-" * 50)

# Recreate the dataset with best barrier parameters
logging.info("Recreating dataset with best barrier parameters...")
final_params = {
    'volatility_window': best_barrier_params['volatility_window'],
    'upper_barrier_multiplier': best_barrier_params['upper_barrier_multiplier'],
    'lower_barrier_multiplier': best_barrier_params['lower_barrier_multiplier'],
    'time_barrier_days': best_barrier_params['time_barrier_days'],
    'verbose': False
}

# Apply triple barrier labeling with best parameters
final_triple_barrier_df = apply_triple_barrier_labeling(
    data=filtered_df,
    **final_params
)

# Merge with filtered_df
final_triple_barrier_df['decision_date'] = pd.to_datetime(final_triple_barrier_df['decision_date'])
final_merged_df = final_triple_barrier_df.merge(
    filtered_df, 
    left_on='decision_date', 
    right_on='date', 
    how='left'
)

if 'date' in final_merged_df.columns:
    final_merged_df = final_merged_df.drop('date', axis=1)

# Drop unnecessary columns for training
final_columns_to_drop = [
    'decision_date', 'entry_date', 'end_date', 'end_price',
    'return', 'barrier_touched', 'value_at_barrier_touched',
    'time_converted', 'datetime', 'time', 'time_barrier'
]
final_existing_columns_to_drop = [col for col in final_columns_to_drop if col in final_merged_df.columns]

if final_existing_columns_to_drop:
    df_with_labels_final = final_merged_df.drop(columns=final_existing_columns_to_drop)
else:
    df_with_labels_final = final_merged_df.copy()

# Remove rows with NaN labels
df_with_labels_final = df_with_labels_final.dropna(subset=['label'])

logging.info(f"Final dataset shape: {df_with_labels_final.shape}")
logging.info(f"Label distribution:")
logging.info(df_with_labels_final['label'].value_counts().sort_index())

# Apply time-based split to final dataset (same as initial approach)
final_merged_df['decision_date'] = pd.to_datetime(final_merged_df['decision_date'])

# Define Q3 2024 start date (July 1, 2024) - same as initial split
q3_2024_start_final = pd.Timestamp('2024-07-01')

# Split data based on time (same logic as initial split)
test_mask_final = final_merged_df['decision_date'] >= q3_2024_start_final
train_val_mask_final = ~test_mask_final

# Create test set from Q3 2024 onwards
test_data_final = final_merged_df[test_mask_final].copy()
logging.info(f"Final test data period: {test_data_final['decision_date'].min()} to {test_data_final['decision_date'].max()}")
logging.info(f"Final test data shape: {test_data_final.shape}")

# Create train+validation set from data before Q3 2024
train_val_data_final = final_merged_df[train_val_mask_final].copy()
logging.info(f"Final train+val data period: {train_val_data_final['decision_date'].min()} to {train_val_data_final['decision_date'].max()}")
logging.info(f"Final train+val data shape: {train_val_data_final.shape}")

# Drop datetime columns for final training
final_columns_to_drop_clean = [
    'decision_date', 'entry_date', 'end_date', 'end_price',
    'return', 'barrier_touched', 'value_at_barrier_touched',
    'time_converted', 'datetime', 'time', 'time_barrier'
]

# Apply column dropping to final test set
existing_cols_test_final = [col for col in final_columns_to_drop_clean if col in test_data_final.columns]
if existing_cols_test_final:
    test_data_clean_final = test_data_final.drop(columns=existing_cols_test_final)
else:
    test_data_clean_final = test_data_final.copy()

# Apply column dropping to final train+val set
existing_cols_train_val_final = [col for col in final_columns_to_drop_clean if col in train_val_data_final.columns]
if existing_cols_train_val_final:
    train_val_data_clean_final = train_val_data_final.drop(columns=existing_cols_train_val_final)
else:
    train_val_data_clean_final = train_val_data_final.copy()

# Extract features and labels for final test set
X_test_final = test_data_clean_final.drop('label', axis=1)
y_test_final = test_data_clean_final['label']

# Extract features and labels for final train+val set
X_train_val_final = train_val_data_clean_final.drop('label', axis=1)
y_train_val_final = train_val_data_clean_final['label']

# Split train+val into train (80%) and validation (20%) - remove stratify for time series
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_train_val_final, y_train_val_final, test_size=0.20, random_state=42
)

logging.info(f"\nFinal data splits:")
logging.info(f"Training set: {X_train_final.shape[0]} samples")
logging.info(f"Validation set: {X_val_final.shape[0]} samples")
logging.info(f"Test set: {X_test_final.shape[0]} samples")

# Map labels
unique_labels_final = sorted(y_train_final.unique())
num_classes_final = len(unique_labels_final)

if num_classes_final == 2:
    label_mapping_final = {unique_labels_final[0]: 0, unique_labels_final[1]: 1}
else:
    label_mapping_final = {label: idx for idx, label in enumerate(unique_labels_final)}

y_train_mapped_final = y_train_final.map(label_mapping_final)
y_val_mapped_final = y_val_final.map(label_mapping_final)
y_test_mapped_final = y_test_final.map(label_mapping_final)

logging.info(f"\nFinal label mapping: {label_mapping_final}")
logging.info(f"Number of classes: {num_classes_final}")

# Train final model with best combined hyperparameters
logging.info("\nTraining final model with best combined hyperparameters...")
if num_classes_final == 2:
    final_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        **best_xgb_params_combined
    )
else:
    final_model = XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes_final,
        eval_metric='mlogloss',
        random_state=42,
        **best_xgb_params_combined
    )

# Train the final model
final_model.fit(X_train_final, y_train_mapped_final)

# Evaluate on all sets
logging.info("\nEvaluating final model on all datasets...")

# Training set evaluation
y_train_pred_final = final_model.predict(X_train_final)
y_train_proba_final = final_model.predict_proba(X_train_final)

train_accuracy_final = accuracy_score(y_train_mapped_final, y_train_pred_final)
train_precision_final = precision_score(y_train_mapped_final, y_train_pred_final, average='weighted')
train_recall_final = recall_score(y_train_mapped_final, y_train_pred_final, average='weighted')
train_f1_final = f1_score(y_train_mapped_final, y_train_pred_final, average='weighted')

if num_classes_final == 2:
    train_roc_auc_final = roc_auc_score(y_train_mapped_final, y_train_proba_final[:, 1])
else:
    train_roc_auc_final = roc_auc_score(y_train_mapped_final, y_train_proba_final, multi_class='ovr')

# Validation set evaluation
y_val_pred_final = final_model.predict(X_val_final)
y_val_proba_final = final_model.predict_proba(X_val_final)

val_accuracy_final = accuracy_score(y_val_mapped_final, y_val_pred_final)
val_precision_final = precision_score(y_val_mapped_final, y_val_pred_final, average='weighted')
val_recall_final = recall_score(y_val_mapped_final, y_val_pred_final, average='weighted')
val_f1_final = f1_score(y_val_mapped_final, y_val_pred_final, average='weighted')

if num_classes_final == 2:
    val_roc_auc_final = roc_auc_score(y_val_mapped_final, y_val_proba_final[:, 1])
else:
    val_roc_auc_final = roc_auc_score(y_val_mapped_final, y_val_proba_final, multi_class='ovr')

# Test set evaluation
y_test_pred_final = final_model.predict(X_test_final)
y_test_proba_final = final_model.predict_proba(X_test_final)

test_accuracy_final = accuracy_score(y_test_mapped_final, y_test_pred_final)
test_precision_final = precision_score(y_test_mapped_final, y_test_pred_final, average='weighted')
test_recall_final = recall_score(y_test_mapped_final, y_test_pred_final, average='weighted')
test_f1_final = f1_score(y_test_mapped_final, y_test_pred_final, average='weighted')

# Handle ROC AUC calculation with proper class checking
unique_test_classes = len(np.unique(y_test_mapped_final))
if unique_test_classes == 2 and y_test_proba_final.shape[1] == 2:
    test_roc_auc_final = roc_auc_score(y_test_mapped_final, y_test_proba_final[:, 1])
elif unique_test_classes > 2 and y_test_proba_final.shape[1] == unique_test_classes:
    test_roc_auc_final = roc_auc_score(y_test_mapped_final, y_test_proba_final, multi_class='ovr')
else:
    # Fallback: use accuracy as ROC AUC when there's a mismatch
    test_roc_auc_final = test_accuracy_final
    logging.warning(f"ROC AUC calculation skipped due to class mismatch. Test classes: {unique_test_classes}, Proba shape: {y_test_proba_final.shape}")

# Display final results
logging.info("\n" + "="*60)
logging.info("FINAL MODEL PERFORMANCE WITH BEST HYPERPARAMETERS")
logging.info("="*60)

logging.info(f"\nTraining Set Performance:")
logging.info(f"Accuracy: {train_accuracy_final:.4f}")
logging.info(f"Precision: {train_precision_final:.4f}")
logging.info(f"Recall: {train_recall_final:.4f}")
logging.info(f"F1-Score: {train_f1_final:.4f}")
logging.info(f"ROC AUC: {train_roc_auc_final:.4f}")

logging.info(f"\nValidation Set Performance:")
logging.info(f"Accuracy: {val_accuracy_final:.4f}")
logging.info(f"Precision: {val_precision_final:.4f}")
logging.info(f"Recall: {val_recall_final:.4f}")
logging.info(f"F1-Score: {val_f1_final:.4f}")
logging.info(f"ROC AUC: {val_roc_auc_final:.4f}")

logging.info(f"\nTest Set Performance:")
logging.info(f"Accuracy: {test_accuracy_final:.4f}")
logging.info(f"Precision: {test_precision_final:.4f}")
logging.info(f"Recall: {test_recall_final:.4f}")
logging.info(f"F1-Score: {test_f1_final:.4f}")
logging.info(f"ROC AUC: {test_roc_auc_final:.4f}")

# Confusion Matrix for Test Set
logging.info(f"\nTest Set Confusion Matrix:")
cm_test_final = confusion_matrix(y_test_mapped_final, y_test_pred_final)
logging.info(cm_test_final)

# Classification Report for Test Set
logging.info(f"\nTest Set Classification Report:")
reverse_label_mapping_final = {v: k for k, v in label_mapping_final.items()}
# Only use classes that are actually present in the test data
unique_test_classes_actual = sorted(np.unique(y_test_mapped_final))
target_names_final = [str(reverse_label_mapping_final[i]) for i in unique_test_classes_actual]
logging.info(classification_report(y_test_mapped_final, y_test_pred_final, target_names=target_names_final))

# Performance comparison
logging.info(f"\n" + "="*60)
logging.info("PERFORMANCE COMPARISON")
logging.info("="*60)

performance_comparison_final = pd.DataFrame({
    'Dataset': ['Training', 'Validation', 'Test'],
    'Accuracy': [train_accuracy_final, val_accuracy_final, test_accuracy_final],
    'Precision': [train_precision_final, val_precision_final, test_precision_final],
    'Recall': [train_recall_final, val_recall_final, test_recall_final],
    'F1-Score': [train_f1_final, val_f1_final, test_f1_final],
    'ROC AUC': [train_roc_auc_final, val_roc_auc_final, test_roc_auc_final]
})

logging.info(performance_comparison_final.round(4))

# Save final results
final_results_path = 'logs/xgb_results/final_performance_comparison.csv'
performance_comparison_final.to_csv(final_results_path, index=False)
logging.info(f"\nFinal performance comparison saved to: {final_results_path}")

# Save final model
final_model_path = 'logs/xgb_models/final_tuned_model.pkl'
joblib.dump(final_model, final_model_path)
logging.info(f"Final tuned model saved to: {final_model_path}")

# Save final test predictions
final_test_predictions_dict = {
    'actual_label': y_test_final.values,
    'predicted_label': [reverse_label_mapping_final[pred] for pred in y_test_pred_final],
    'actual_mapped': y_test_mapped_final.values,
    'predicted_mapped': y_test_pred_final
}

# Add probability columns based on number of classes
if num_classes_final == 2:
    final_test_predictions_dict['probability_class_0'] = y_test_proba_final[:, 0]
    final_test_predictions_dict['probability_class_1'] = y_test_proba_final[:, 1]
else:
    for i in range(num_classes_final):
        final_test_predictions_dict[f'probability_class_{i}'] = y_test_proba_final[:, i]

final_test_predictions_df = pd.DataFrame(final_test_predictions_dict)

final_test_predictions_path = 'logs/xgb_results/final_test_predictions.csv'
final_test_predictions_df.to_csv(final_test_predictions_path, index=False)
logging.info(f"Final test predictions saved to: {final_test_predictions_path}")

# Summary of hyperparameter tuning
logging.info(f"\n" + "="*60)
logging.info("HYPERPARAMETER TUNING SUMMARY")
logging.info("="*60)

logging.info(f"\nBest XGBoost-Only Parameters:")
for param, value in best_xgb_params.items():
    logging.info(f"  {param}: {value}")

logging.info(f"\nBest Combined XGBoost Parameters:")
for param, value in best_xgb_params_combined.items():
    logging.info(f"  {param}: {value}")

logging.info(f"\nBest Combined Barrier Parameters:")
for param, value in best_barrier_params.items():
    logging.info(f"  {param}: {value}")

logging.info(f"\nPerformance Improvements:")
logging.info(f"Original Model Validation Accuracy: 0.6295")
logging.info(f"XGBoost-Only Tuned Validation Accuracy: {best_xgb_score:.4f}")
logging.info(f"Combined Tuned Validation Accuracy: {best_combined_score:.4f}")
logging.info(f"Final Tuned Model Test Accuracy: {test_accuracy_final:.4f}")

logging.info(f"\nFinal Model Generalization:")
logging.info(f"Training Accuracy: {train_accuracy_final:.4f}")
logging.info(f"Validation Accuracy: {val_accuracy_final:.4f}")
logging.info(f"Test Accuracy: {test_accuracy_final:.4f}")
logging.info(f"Train-Val Gap: {train_accuracy_final - val_accuracy_final:.4f}")
logging.info(f"Train-Test Gap: {train_accuracy_final - test_accuracy_final:.4f}")

logging.info("\n" + "="*80)
logging.info("HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY!")
logging.info("="*80)