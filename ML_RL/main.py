# Import Section
import pandas as pd
import warnings
import logging
from utils.get_ticker import *
from utils.load_data import *
from fundamental_feature_engineering import *

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

# 6. Train Test Split










