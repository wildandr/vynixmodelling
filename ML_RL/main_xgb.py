# Import Section
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

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

# Dataframe Init
original_df = pd.read_csv('/root/vynixmodelling/dataset/TSLA_original.csv')

# Logging Init
# logging location: /root/vynixmodelling/ML_RL/logs
# log name format: {log_name}_{time}.log
logging.basicConfig(filename='/root/vynixmodelling/ML_RL/logs/main.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Preprocessing
df = original_df.copy()
logging.info(f"Original dataframe shape: {df.shape}")

# Exclude rows with NaN or null values
non_null_df = df.dropna()

# Print total rows in the dataframe without NaN or null values
logging.info(f"Total rows in df without NaN or null values: {len(non_null_df)}")

# Convert 'time' column to datetime format and print the first few rows
non_null_df['time_converted'] = pd.to_datetime(non_null_df['time'], unit='s').dt.strftime('%d%m%Y')
# print(non_null_df[['time', 'time_converted']].head(1))
print(non_null_df.head)
print(non_null_df.tail)
print(non_null_df.info())
print(non_null_df.describe())
print(non_null_df.columns)

# Data Gathering with Fundamental data
# 1. Process Fundamental Data from Local Files
# Menggunakan fungsi baru untuk membaca data lokal berdasarkan CIK
pivoted_df = process_fundamental_data_local("TSLA")

# Jika gagal memuat dari lokal, fallback ke API
if pivoted_df is None:
    print("Fallback to API method...")
    fundamental_data = get_fundamental_data("TSLA")
    pivoted_df = process_fundamental_data(fundamental_data, "TSLA")
    print("Data loaded from API successfully")
else:
    # Baca kembali CSV yang sudah diproses untuk mendapatkan DataFrame
    pivoted_df = pd.read_csv('/root/vynixmodelling/dataset/data_fundamental/TSLA_time.csv', index_col=0)
    print(f"Data loaded from local files successfully: {pivoted_df.shape}")

# 3. Fundamental Data Financial Feature Engineering
from fundamental_feature_engineering import apply_feature_engineering
enhanced_fundamental_df = apply_feature_engineering(pivoted_df, "TSLA")
# print(enhanced_fundamental_df.head(1))

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
print(f"\nFinal processed data shape: {filtered_df.shape}")
print(f"Date range: {filtered_df['date'].min()} to {filtered_df['date'].max()}")
print(f"Technical columns: {len([col for col in filtered_df.columns if any(tech in col.lower() for tech in ['open', 'high', 'low', 'close', 'volume'])])}")
print(f"Fundamental columns: {len([col for col in filtered_df.columns if col not in ['time', 'date', 'datetime'] and not any(tech in col.lower() for tech in ['open', 'high', 'low', 'close', 'volume'])])}")
print("\nPreprocessing completed successfully!")
print(f'Data completeness: {(filtered_df.count().sum() / (len(filtered_df) * len(filtered_df.columns)) * 100):.2f}%')

# 5. Implementasi Labelling Triple Barrier Method ke filtered_df
# Semua parameter di init di file ini. main.py.
from triple_barrier.triplebarrier import apply_triple_barrier_labeling
from triple_barrier.visualizebarrier import generate_triple_barrier_visualizations

# Triple Barrier Method Parameters - dapat dikonfigurasi sesuai kebutuhan
TRIPLE_BARRIER_PARAMS = {
    'volatility_window': 20,           # Window untuk menghitung volatilitas
    'upper_barrier_multiplier': 1.0,   # Multiplier untuk upper barrier
    'lower_barrier_multiplier': 1.0,   # Multiplier untuk lower barrier
    'time_barrier_days': 15,            # Maksimum periode untuk menunggu barrier touch
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

print("\n=== Applying Triple Barrier Method ===")
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
print("\n=== Generating Triple Barrier Visualizations ===")
logging.info("Generating Triple Barrier visualizations...")

visualization_files = generate_triple_barrier_visualizations(
    data=filtered_df,
    triple_barrier_df=triple_barrier_df,
    **VISUALIZATION_PARAMS
)

logging.info(f"Visualizations generated: {len(visualization_files)} files")
for file_type, file_path in visualization_files.items():
    logging.info(f"{file_type}: {file_path}")

print(f"\n=== Triple Barrier Implementation Complete ===")
print(f"Labels generated: {len(triple_barrier_df)} samples")
print(f"Results saved to: {triple_barrier_output_path}")
# print(f"Visualizations saved to: {VISUALIZATION_PARAMS['output_dir']}")
# print(f"Total visualization files: {len(visualization_files)}")

# Summary statistik final
# 6. Merge Triple Barrier labels dengan filtered_df berdasarkan date
# merged_label_df = triple_barrier_df.copy() + filtered_df.copy(), gunakan triple_barrier_df sebagai
# data utama yang sudah ber-label, lalu tarik data filtered_df sesuai dengan 'date' dari 'decision_date' triple_barrier_df

print("\n=== Merging Triple Barrier Labels with Features ===")
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

print(f"Triple Barrier data shape: {triple_barrier_df.shape}")
print(f"Filtered data shape: {filtered_df.shape}")
print(f"Merged data shape: {merged_label_df.shape}")
print(f"Successfully merged: {len(merged_label_df[merged_label_df.notna().all(axis=1)])} complete rows")

# Simpan hasil merge
merged_output_path = '/root/vynixmodelling/ML_RL/merged_labeled_data.csv'
merged_label_df.to_csv(merged_output_path, index=False)
logging.info(f"Merged labeled data saved to: {merged_output_path}")
print(f"Merged labeled data saved to: {merged_output_path}")

print(f"\n=== Final Summary ===")
print(f"Original processed data shape: {filtered_df.shape}")
print(f"Triple Barrier labels: {len(triple_barrier_df)} samples")
print(f"Merged labeled data shape: {merged_label_df.shape}")
print(f"Date range: {filtered_df['date'].min()} to {filtered_df['date'].max()}")
print(f"Date range: {triple_barrier_df['decision_date'].min()} to {triple_barrier_df['decision_date'].max()}")
print(f"Technical + Fundamental features: {len(filtered_df.columns)} columns")
print(f"Total features in merged data: {len(merged_label_df.columns)} columns")
print(f"Triple Barrier parameters used: {TRIPLE_BARRIER_PARAMS}")
print("\nAll preprocessing, labeling, and merging completed successfully!")

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
print(f"Training DataFrame shape after dropping columns: {training_df.shape}")
print(f"Training DataFrame columns: {training_df.columns.tolist()}")

# 6.2 Train Val Test Split. 70:20:10
print("\n=== Splitting Data into Train, Validation, and Test Sets ===")
logging.info("Splitting data into train, validation, and test sets...")

X = training_df.drop('label', axis=1)
y = training_df['label']

# Split 70% for training, 30% for temp (test + validation)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

# Split X_temp and y_temp into test (20%) and validation (10%)
# 20% of original is 2/3 of X_temp, and 10% of original is 1/3 of X_temp
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.333333, random_state=42, stratify=y_temp)

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
print(f"Train, test, and validation data saved to {output_dir}")

# 7. Training
# 7.1 XGBoost. Hanya gunakan data Train dan Val dulu.
print("\n=== Training XGBoost Model ===")
logging.info("Initializing and training XGBoost Classifier...")

# Check unique labels
print(f"Unique labels in training data: {sorted(y_train.unique())}")
print(f"Label distribution in training data: {y_train.value_counts().sort_index()}")
logging.info(f"Unique labels: {sorted(y_train.unique())}")

# Check unique labels in training data
unique_labels = sorted(y_train.unique())
print(f"Unique labels found in training data: {unique_labels}")

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

print(f"Label mapping: {label_mapping}")
print(f"Number of classes: {num_classes}")

# Apply mapping to training and validation labels
y_train_mapped = y_train.map(label_mapping)
y_val_mapped = y_val.map(label_mapping)

print(f"Mapped labels in training data: {sorted(y_train_mapped.unique())}")
print(f"Mapped label distribution: {y_train_mapped.value_counts().sort_index()}")

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
print("Training XGBoost model...")
xgb_model.fit(X_train, y_train_mapped,
              eval_set=[(X_train, y_train_mapped), (X_val, y_val_mapped)],
              verbose=True)

logging.info("XGBoost Model training completed.")
print("XGBoost Model training completed.")

# Evaluate on Validation Set
print("\n=== Evaluating XGBoost Model on Validation Set ===")
logging.info("Evaluating XGBoost Model on validation set...")

# Make predictions using mapped labels
y_pred_val_mapped = xgb_model.predict(X_val)
y_proba_val = xgb_model.predict_proba(X_val)

# Convert predictions back to original labels
y_pred_val = pd.Series(y_pred_val_mapped).map(reverse_label_mapping)

print(f"Predicted labels (mapped): {sorted(y_pred_val_mapped)}")
print(f"Predicted labels (original): {sorted(y_pred_val.unique())}")
print(f"Validation labels (original): {sorted(y_val.unique())}")

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

print(f"Validation Accuracy: {accuracy_val:.4f}")
print(f"Validation Precision: {precision_val:.4f}")
print(f"Validation Recall: {recall_val:.4f}")
print(f"Validation F1-Score: {f1_val:.4f}")
print(f"Validation ROC AUC: {roc_auc_val:.4f}")

# Display confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_val, y_pred_val)
print(f"Confusion Matrix:\n{cm}")
print(f"\nClassification Report:\n{classification_report(y_val, y_pred_val)}")

print("XGBoost Model evaluation on validation set completed.")

# 7.2 Evaluate on Training Set for comparison
print("\n=== Evaluating XGBoost Model on Training Set ===")
logging.info("Evaluating XGBoost Model on training set...")

# Make predictions on training set
y_pred_train_mapped = xgb_model.predict(X_train)
y_pred_train = pd.Series(y_pred_train_mapped).map(reverse_label_mapping)

# Calculate training metrics
accuracy_train = accuracy_score(y_train, y_pred_train)
precision_train = precision_score(y_train, y_pred_train, average='weighted')
recall_train = recall_score(y_train, y_pred_train, average='weighted')
f1_train = f1_score(y_train, y_pred_train, average='weighted')

print(f"Training Accuracy: {accuracy_train:.4f}")
print(f"Training Precision: {precision_train:.4f}")
print(f"Training Recall: {recall_train:.4f}")
print(f"Training F1-Score: {f1_train:.4f}")

logging.info(f"Training Accuracy: {accuracy_train:.4f}")
logging.info(f"Training Precision: {precision_train:.4f}")
logging.info(f"Training Recall: {recall_train:.4f}")
logging.info(f"Training F1-Score: {f1_train:.4f}")

# 7.3 Save the trained model
print("\n=== Saving XGBoost Model ===")
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
print(f"Model saved to: {model_path}")
print(f"Label mappings saved to: {mapping_path}")

# 7.4 Feature Importance Analysis
print("\n=== Feature Importance Analysis ===")
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
print("Top 20 Most Important Features:")
print(feature_importance_df.head(20))

# Save feature importance
feature_importance_path = os.path.join(model_dir, 'feature_importance.csv')
feature_importance_df.to_csv(feature_importance_path, index=False)
logging.info(f"Feature importance saved to: {feature_importance_path}")
print(f"Feature importance saved to: {feature_importance_path}")

print("\n=== XGBoost Training and Evaluation Complete ===")
logging.info("XGBoost training and evaluation completed successfully.")

# 7.5 Testing model menggunakan data test dengan inferensi menggunakan model yang sudah dibuat sebelumnya
print("\n=== Testing XGBoost Model on Test Set ===")
logging.info("Testing XGBoost Model on test set...")

# Make predictions on test set using the trained model
y_pred_test_mapped = xgb_model.predict(X_test)
y_proba_test = xgb_model.predict_proba(X_test)

# Convert predictions back to original labels
y_pred_test = pd.Series(y_pred_test_mapped).map(reverse_label_mapping)

print(f"Test set size: {len(X_test)} samples")
print(f"Predicted labels (mapped): {sorted(set(y_pred_test_mapped))}")
print(f"Predicted labels (original): {sorted(y_pred_test.unique())}")
print(f"Test labels (original): {sorted(y_test.unique())}")

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
print("\n=== Test Set Performance ===")
print(f"Test Accuracy: {accuracy_test:.4f}")
print(f"Test Precision: {precision_test:.4f}")
print(f"Test Recall: {recall_test:.4f}")
print(f"Test F1-Score: {f1_test:.4f}")
print(f"Test ROC AUC: {roc_auc_test:.4f}")

logging.info(f"Test Accuracy: {accuracy_test:.4f}")
logging.info(f"Test Precision: {precision_test:.4f}")
logging.info(f"Test Recall: {recall_test:.4f}")
logging.info(f"Test F1-Score: {f1_test:.4f}")
logging.info(f"Test ROC AUC: {roc_auc_test:.4f}")

# Display test confusion matrix
print("\n=== Test Set Confusion Matrix ===")
cm_test = confusion_matrix(y_test, y_pred_test)
print(f"Test Confusion Matrix:\n{cm_test}")
print(f"\nTest Classification Report:\n{classification_report(y_test, y_pred_test)}")

# Compare performance across all sets
print("\n=== Performance Comparison Across All Sets ===")
performance_comparison = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC'],
    'Training': [accuracy_train, precision_train, recall_train, f1_train, 0.0],  # ROC AUC not calculated for training
    'Validation': [accuracy_val, precision_val, recall_val, f1_val, roc_auc_val],
    'Test': [accuracy_test, precision_test, recall_test, f1_test, roc_auc_test]
})

print(performance_comparison.round(4))

# Save performance comparison
performance_path = os.path.join(model_dir, 'performance_comparison.csv')
performance_comparison.to_csv(performance_path, index=False)
logging.info(f"Performance comparison saved to: {performance_path}")
print(f"Performance comparison saved to: {performance_path}")

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
print(f"Test predictions saved to: {test_predictions_path}")

# Analyze prediction confidence
print("\n=== Prediction Confidence Analysis ===")
print(f"Average prediction confidence: {test_predictions_df['prediction_confidence'].mean():.4f}")
print(f"Confidence for correct predictions: {test_predictions_df[test_predictions_df['correct_prediction']]['prediction_confidence'].mean():.4f}")
print(f"Confidence for incorrect predictions: {test_predictions_df[~test_predictions_df['correct_prediction']]['prediction_confidence'].mean():.4f}")

# Show prediction distribution
print("\n=== Test Set Prediction Distribution ===")
print("Actual vs Predicted Label Distribution:")
print(pd.crosstab(y_test, y_pred_test, margins=True))

print("\n=== XGBoost Model Testing Complete ===")
logging.info("XGBoost model testing on test set completed successfully.")

# 8. Hyperparameter Tuning. Hanya gunakan data Train dan Val terlebih dahulu saja.
print("\n" + "="*80)
print("8. HYPERPARAMETER TUNING")
print("="*80)

# Import libraries for hyperparameter tuning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
import optuna
import time
import warnings
warnings.filterwarnings('ignore')

# 8.1 Hyperparameter pada parameter-parameter Model XGboost saja.
print("\n8.1 XGBoost Hyperparameter Tuning")
print("-" * 50)

# Define XGBoost parameter grid
if num_classes == 2:
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }
    objective = 'binary:logistic'
    eval_metric = 'logloss'
else:
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }
    objective = 'multi:softmax'
    eval_metric = 'mlogloss'

# Create base model for tuning
base_xgb = XGBClassifier(
    objective=objective,
    num_class=num_classes if num_classes > 2 else None,
    eval_metric=eval_metric,
    random_state=42,
    n_jobs=-1
)

# Use RandomizedSearchCV for efficiency
print("Starting XGBoost hyperparameter tuning...")
start_time = time.time()

xgb_random_search = RandomizedSearchCV(
    estimator=base_xgb,
    param_distributions=xgb_param_grid,
    n_iter=50,  # Number of parameter settings sampled
    scoring='accuracy',
    cv=3,  # 3-fold cross validation
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# Fit the random search
xgb_random_search.fit(X_train, y_train_mapped)

xgb_tuning_time = time.time() - start_time
print(f"XGBoost tuning completed in {xgb_tuning_time:.2f} seconds")

# Get best parameters and score
best_xgb_params = xgb_random_search.best_params_
best_xgb_score = xgb_random_search.best_score_

print(f"\nBest XGBoost Parameters:")
for param, value in best_xgb_params.items():
    print(f"  {param}: {value}")
print(f"Best Cross-Validation Score: {best_xgb_score:.4f}")

# Train best XGBoost model
best_xgb_model = xgb_random_search.best_estimator_

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

print(f"\nTuned XGBoost Validation Performance:")
print(f"Accuracy: {val_accuracy_tuned:.4f}")
print(f"Precision: {val_precision_tuned:.4f}")
print(f"Recall: {val_recall_tuned:.4f}")
print(f"F1-Score: {val_f1_tuned:.4f}")
print(f"ROC AUC: {val_roc_auc_tuned:.4f}")

# Compare with original model (use validation accuracy from section 7.4)
print(f"\nComparison with Original Model:")
original_val_accuracy = 0.6295  # From previous validation results
print(f"Original Validation Accuracy: {original_val_accuracy:.4f}")
print(f"Tuned Validation Accuracy: {val_accuracy_tuned:.4f}")
print(f"Improvement: {val_accuracy_tuned - original_val_accuracy:.4f}")

# Save tuned XGBoost model
tuned_xgb_model_path = 'logs/xgb_models/tuned_xgb_model.pkl'
joblib.dump(best_xgb_model, tuned_xgb_model_path)
print(f"\nTuned XGBoost model saved to: {tuned_xgb_model_path}")

# 8.2 Melakukan tuning juga pada volatility_window, upper_barrier_multiplier, lower_barrier_multiplier, time_barrier_days.
print("\n8.2 Triple Barrier Parameters Tuning")
print("-" * 50)

# Define triple barrier parameter grid
barrier_param_grid = {
    'volatility_window': [10, 20, 30],
    'upper_barrier_multiplier': [1.0, 1.5, 2.0],
    'lower_barrier_multiplier': [1.0, 1.5, 2.0],
    'time_barrier_days': [5, 10, 15]
}

print("Starting Triple Barrier parameters tuning...")
start_time = time.time()

# Store original parameters from TRIPLE_BARRIER_PARAMS
original_volatility_window = TRIPLE_BARRIER_PARAMS['volatility_window']
original_upper_barrier_multiplier = TRIPLE_BARRIER_PARAMS['upper_barrier_multiplier']
original_lower_barrier_multiplier = TRIPLE_BARRIER_PARAMS['lower_barrier_multiplier']
original_time_barrier_days = TRIPLE_BARRIER_PARAMS['time_barrier_days']

best_barrier_score = 0
best_barrier_params = {}
best_barrier_model = None

# Grid search for barrier parameters
total_combinations = len(barrier_param_grid['volatility_window']) * len(barrier_param_grid['upper_barrier_multiplier']) * len(barrier_param_grid['lower_barrier_multiplier']) * len(barrier_param_grid['time_barrier_days'])
current_combination = 0

for vol_window in barrier_param_grid['volatility_window']:
    for upper_mult in barrier_param_grid['upper_barrier_multiplier']:
        for lower_mult in barrier_param_grid['lower_barrier_multiplier']:
            for time_days in barrier_param_grid['time_barrier_days']:
                current_combination += 1
                print(f"\nTesting combination {current_combination}/{total_combinations}:")
                print(f"  volatility_window: {vol_window}")
                print(f"  upper_barrier_multiplier: {upper_mult}")
                print(f"  lower_barrier_multiplier: {lower_mult}")
                print(f"  time_barrier_days: {time_days}")
                
                try:
                    # Create temporary parameters dictionary
                    temp_params = {
                        'volatility_window': vol_window,
                        'upper_barrier_multiplier': upper_mult,
                        'lower_barrier_multiplier': lower_mult,
                        'time_barrier_days': time_days,
                        'verbose': False
                    }
                    
                    # Apply triple barrier labeling with current parameters
                    temp_triple_barrier_df = apply_triple_barrier_labeling(
                        data=filtered_df,
                        **temp_params
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
                        print(f"  Skipped: Too few samples ({len(df_with_labels_new)})")
                        continue
                    
                    # Prepare features and labels
                    feature_columns_new = [col for col in df_with_labels_new.columns if col != 'label']
                    X_new = df_with_labels_new[feature_columns_new]
                    y_new = df_with_labels_new['label']
                    
                    # Split data
                    X_train_new, X_temp_new, y_train_new, y_temp_new = train_test_split(
                        X_new, y_new, test_size=0.4, random_state=42, stratify=y_new
                    )
                    X_val_new, X_test_new, y_val_new, y_test_new = train_test_split(
                        X_temp_new, y_temp_new, test_size=0.5, random_state=42, stratify=y_temp_new
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
                    
                    # Train model with best XGBoost parameters
                    if num_classes_new == 2:
                        temp_model = XGBClassifier(
                            objective='binary:logistic',
                            eval_metric='logloss',
                            random_state=42,
                            **best_xgb_params
                        )
                    else:
                        temp_model = XGBClassifier(
                            objective='multi:softmax',
                            num_class=num_classes_new,
                            eval_metric='mlogloss',
                            random_state=42,
                            **best_xgb_params
                        )
                    
                    temp_model.fit(X_train_new, y_train_mapped_new)
                    
                    # Evaluate on validation set
                    y_val_pred_new = temp_model.predict(X_val_new)
                    val_accuracy_new = accuracy_score(y_val_mapped_new, y_val_pred_new)
                    
                    print(f"  Validation Accuracy: {val_accuracy_new:.4f}")
                    
                    # Update best parameters if this is better
                    if val_accuracy_new > best_barrier_score:
                        best_barrier_score = val_accuracy_new
                        best_barrier_params = {
                            'volatility_window': vol_window,
                            'upper_barrier_multiplier': upper_mult,
                            'lower_barrier_multiplier': lower_mult,
                            'time_barrier_days': time_days
                        }
                        best_barrier_model = temp_model
                        print(f"  *** New best score: {val_accuracy_new:.4f} ***")
                    
                except Exception as e:
                    print(f"  Error: {str(e)}")
                    continue

barrier_tuning_time = time.time() - start_time
print(f"\nTriple Barrier tuning completed in {barrier_tuning_time:.2f} seconds")

print(f"\nBest Triple Barrier Parameters:")
for param, value in best_barrier_params.items():
    print(f"  {param}: {value}")
print(f"Best Validation Score: {best_barrier_score:.4f}")

# Compare with original barrier parameters
print(f"\nComparison with Original Barrier Parameters:")
print(f"Original Validation Accuracy: {val_accuracy_tuned:.4f}")
print(f"Best Barrier Validation Accuracy: {best_barrier_score:.4f}")
print(f"Improvement: {best_barrier_score - val_accuracy_tuned:.4f}")

# Save best barrier model
best_barrier_model_path = 'logs/xgb_models/best_barrier_model.pkl'
joblib.dump(best_barrier_model, best_barrier_model_path)
print(f"\nBest barrier model saved to: {best_barrier_model_path}")

# Save best parameters
best_params_combined = {
    'xgb_params': best_xgb_params,
    'barrier_params': best_barrier_params,
    'best_xgb_score': best_xgb_score,
    'best_barrier_score': best_barrier_score
}

best_params_path = 'logs/xgb_results/best_hyperparameters.pkl'
joblib.dump(best_params_combined, best_params_path)
print(f"Best hyperparameters saved to: {best_params_path}")

# 8.3 Melakukan testing dataset test dengan model yang disimpan dari proses 8.1 dan 8.2
print("\n8.3 Final Testing with Best Hyperparameters")
print("-" * 50)

# Recreate the dataset with best barrier parameters
print("Recreating dataset with best barrier parameters...")
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

print(f"Final dataset shape: {df_with_labels_final.shape}")
print(f"Label distribution:")
print(df_with_labels_final['label'].value_counts().sort_index())

# Prepare features and labels
feature_columns_final = [col for col in df_with_labels_final.columns if col != 'label']
X_final = df_with_labels_final[feature_columns_final]
y_final = df_with_labels_final['label']

# Split data with best parameters
X_train_final, X_temp_final, y_train_final, y_temp_final = train_test_split(
    X_final, y_final, test_size=0.4, random_state=42, stratify=y_final
)
X_val_final, X_test_final, y_val_final, y_test_final = train_test_split(
    X_temp_final, y_temp_final, test_size=0.5, random_state=42, stratify=y_temp_final
)

print(f"\nFinal data splits:")
print(f"Training set: {X_train_final.shape[0]} samples")
print(f"Validation set: {X_val_final.shape[0]} samples")
print(f"Test set: {X_test_final.shape[0]} samples")

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

print(f"\nFinal label mapping: {label_mapping_final}")
print(f"Number of classes: {num_classes_final}")

# Train final model with best hyperparameters
print("\nTraining final model with best hyperparameters...")
if num_classes_final == 2:
    final_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        **best_xgb_params
    )
else:
    final_model = XGBClassifier(
        objective='multi:softmax',
        num_class=num_classes_final,
        eval_metric='mlogloss',
        random_state=42,
        **best_xgb_params
    )

# Train the final model
final_model.fit(X_train_final, y_train_mapped_final)

# Evaluate on all sets
print("\nEvaluating final model on all datasets...")

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

if num_classes_final == 2:
    test_roc_auc_final = roc_auc_score(y_test_mapped_final, y_test_proba_final[:, 1])
else:
    test_roc_auc_final = roc_auc_score(y_test_mapped_final, y_test_proba_final, multi_class='ovr')

# Display final results
print("\n" + "="*60)
print("FINAL MODEL PERFORMANCE WITH BEST HYPERPARAMETERS")
print("="*60)

print(f"\nTraining Set Performance:")
print(f"Accuracy: {train_accuracy_final:.4f}")
print(f"Precision: {train_precision_final:.4f}")
print(f"Recall: {train_recall_final:.4f}")
print(f"F1-Score: {train_f1_final:.4f}")
print(f"ROC AUC: {train_roc_auc_final:.4f}")

print(f"\nValidation Set Performance:")
print(f"Accuracy: {val_accuracy_final:.4f}")
print(f"Precision: {val_precision_final:.4f}")
print(f"Recall: {val_recall_final:.4f}")
print(f"F1-Score: {val_f1_final:.4f}")
print(f"ROC AUC: {val_roc_auc_final:.4f}")

print(f"\nTest Set Performance:")
print(f"Accuracy: {test_accuracy_final:.4f}")
print(f"Precision: {test_precision_final:.4f}")
print(f"Recall: {test_recall_final:.4f}")
print(f"F1-Score: {test_f1_final:.4f}")
print(f"ROC AUC: {test_roc_auc_final:.4f}")

# Confusion Matrix for Test Set
print(f"\nTest Set Confusion Matrix:")
cm_test_final = confusion_matrix(y_test_mapped_final, y_test_pred_final)
print(cm_test_final)

# Classification Report for Test Set
print(f"\nTest Set Classification Report:")
reverse_label_mapping_final = {v: k for k, v in label_mapping_final.items()}
target_names_final = [str(reverse_label_mapping_final[i]) for i in range(num_classes_final)]
print(classification_report(y_test_mapped_final, y_test_pred_final, target_names=target_names_final))

# Performance comparison
print(f"\n" + "="*60)
print("PERFORMANCE COMPARISON")
print("="*60)

performance_comparison_final = pd.DataFrame({
    'Dataset': ['Training', 'Validation', 'Test'],
    'Accuracy': [train_accuracy_final, val_accuracy_final, test_accuracy_final],
    'Precision': [train_precision_final, val_precision_final, test_precision_final],
    'Recall': [train_recall_final, val_recall_final, test_recall_final],
    'F1-Score': [train_f1_final, val_f1_final, test_f1_final],
    'ROC AUC': [train_roc_auc_final, val_roc_auc_final, test_roc_auc_final]
})

print(performance_comparison_final.round(4))

# Save final results
final_results_path = 'logs/xgb_results/final_performance_comparison.csv'
performance_comparison_final.to_csv(final_results_path, index=False)
print(f"\nFinal performance comparison saved to: {final_results_path}")

# Save final model
final_model_path = 'logs/xgb_models/final_tuned_model.pkl'
joblib.dump(final_model, final_model_path)
print(f"Final tuned model saved to: {final_model_path}")

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
print(f"Final test predictions saved to: {final_test_predictions_path}")

# Summary of hyperparameter tuning
print(f"\n" + "="*60)
print("HYPERPARAMETER TUNING SUMMARY")
print("="*60)

print(f"\nBest XGBoost Parameters:")
for param, value in best_xgb_params.items():
    print(f"  {param}: {value}")

print(f"\nBest Triple Barrier Parameters:")
for param, value in best_barrier_params.items():
    print(f"  {param}: {value}")

print(f"\nPerformance Improvements:")
print(f"Original Model Validation Accuracy: {original_val_accuracy:.4f}")
print(f"XGBoost Tuned Validation Accuracy: {best_xgb_score:.4f}")
print(f"Final Tuned Model Test Accuracy: {test_accuracy_final:.4f}")

print(f"\nFinal Model Generalization:")
print(f"Training Accuracy: {train_accuracy_final:.4f}")
print(f"Validation Accuracy: {val_accuracy_final:.4f}")
print(f"Test Accuracy: {test_accuracy_final:.4f}")
print(f"Train-Val Gap: {train_accuracy_final - val_accuracy_final:.4f}")
print(f"Train-Test Gap: {train_accuracy_final - test_accuracy_final:.4f}")

print("\n" + "="*80)
print("HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY!")
print("="*80)