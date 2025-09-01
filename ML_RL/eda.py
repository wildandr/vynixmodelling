# %% [markdown]
# ## Library Installation

# %%
# Install all required libraries in one command
# !pip install tensorflow numpy pandas scikit-learn matplotlib scipy optuna joblib plotly nbformat>=4.2.0 -q

# %% [markdown]
# ## Preprocessing

# %%
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/root/vynixmodelling/dataset/TSLA_original.csv')

# %%
# df.head()
df.tail()

# %%
# Convert time column to datetime
df['date'] = pd.to_datetime(df['time'], unit='s')

# Filter data from Jan 1, 2012 onwards
jan_1_2012_timestamp = pd.Timestamp('2012-01-01').timestamp()
filtered_df = df[df['time'] >= jan_1_2012_timestamp]

filtered_df.head()

# %%
import os

# Buat direktori target jika belum ada
target_dir = "/root/vynixmodelling/dataset"
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    print(f"Directory created: {target_dir}")

# Buat symbolic link
symlink_path = "/dataset"
if os.path.exists(symlink_path):
    os.remove(symlink_path)
    print(f"Removed existing symlink: {symlink_path}")

os.symlink(target_dir, symlink_path)
print(f"Symbolic link created: {symlink_path} -> {target_dir}")

# %%
# Save the filtered dataframe to CSV
filtered_df.to_csv('/root/vynixmodelling/dataset/TSLA_from_2012.csv', index=False)
print("Data saved successfully to 'TSLA_from_2012.csv'")

# %% [markdown]
# ## Data Engineering

# %% [markdown]
# ### implement Triple Barrier Method

# %%
print(filtered_df["time"].dtype)

# %%
# Perbaikan kode untuk memproses DataFrame dengan kolom 'time' (int64) bukan 'date'

# 1. Periksa apakah ada kolom 'time' di DataFrame
if 'time' in filtered_df.columns and not 'date' in filtered_df.columns:
    # 2. Konversi kolom 'time' ke format datetime
    # Jika 'time' adalah timestamp UNIX (dalam detik atau milidetik)
    if filtered_df['time'].iloc[0] > 10000000000:  # Kemungkinan dalam milidetik
        filtered_df['date'] = pd.to_datetime(filtered_df['time'], unit='ms')
    else:  # Kemungkinan dalam detik
        filtered_df['date'] = pd.to_datetime(filtered_df['time'], unit='s')
# )

# %%
filtered_df.columns

# %%
def print_dataframe(df):
    """
    Print all rows of a DataFrame with their index, column names, and values.

    Parameters:
    df (pd.DataFrame): The DataFrame to print.
    """
    # Print column names
    print(" | ".join(df.columns))
    
    # Print each row
    for idx, row in df.iterrows():
        print(f"Row {idx}: {', '.join(str(val) for val in row.values)}")

# %%
# Contoh penggunaan fungsi
print_dataframe(filtered_df)

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

def triple_barrier_method(data, volatility_window=20, upper_barrier_multiplier=1.0, 
                          lower_barrier_multiplier=1.0, time_barrier_days=5):
    """
    Implement Triple Barrier Method using OHLC data with proper entry point calculation.
    
    Parameters:
    - data: DataFrame with 'open', 'high', 'low', 'close' columns and datetime index
    - volatility_window: Window for calculating volatility
    - upper_barrier_multiplier: Multiplier for upper barrier (as multiple of volatility)
    - lower_barrier_multiplier: Multiplier for lower barrier (as multiple of volatility)
    - time_barrier_days: Maximum number of periods to wait for a barrier touch
    
    Returns:
    - DataFrame with labels and barrier information
    """
    # Ensure data is properly indexed
    if 'date' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
        data['date'] = pd.to_datetime(data['date'])
        data = data.set_index('date')
    
    result = []
    dates = data.index
    
    # Calculate daily returns based on close prices
    returns = data['close'].pct_change().fillna(0)
    
    # Calculate rolling volatility
    volatility = returns.rolling(window=volatility_window).std().fillna(method='bfill')
    
    for i in range(len(data) - time_barrier_days - 1):  # -1 because we need room for D-1
        # D-1 is the decision point, D is the entry point
        decision_date = dates[i]
        entry_date = dates[i+1]  # D day is the actual entry
        entry_price = data['close'].iloc[i]  # Use close price of D-1 as entry price
        
        # Set dynamic barriers based on volatility
        upper_barrier = entry_price * (1 + volatility.iloc[i] * upper_barrier_multiplier)
        lower_barrier = entry_price * (1 - volatility.iloc[i] * lower_barrier_multiplier)
        
        # Define the window to look for barrier touches, starting from D (not D-1)
        data_window = data.iloc[i+1:i+1+time_barrier_days]
        date_window = dates[i+1:i+1+time_barrier_days]
        
        if len(data_window) == 0:
            continue  # Skip if we don't have enough data for the time window
        
        # Check if and when barriers are touched using HIGH and LOW prices
        upper_touch_idx = None
        lower_touch_idx = None
        
        for j in range(len(data_window)):
            # Check if HIGH price touches upper barrier
            if data_window['high'].iloc[j] >= upper_barrier:
                upper_touch_idx = j
                break
                
        for j in range(len(data_window)):
            # Check if LOW price touches lower barrier
            if data_window['low'].iloc[j] <= lower_barrier:
                lower_touch_idx = j
                break
        
        # Determine which barrier was touched first
        if upper_touch_idx is not None and (lower_touch_idx is None or upper_touch_idx < lower_touch_idx):
            label = 1  # Up - upper barrier touched first
            barrier_type = "upper"
            touch_date = date_window[upper_touch_idx]
            value_at_barrier = data_window['high'].iloc[upper_touch_idx]
        elif lower_touch_idx is not None:
            label = -1  # Down - lower barrier touched first
            barrier_type = "lower" 
            touch_date = date_window[lower_touch_idx]
            value_at_barrier = data_window['low'].iloc[lower_touch_idx]
        else:
            # Time barrier touched - ALWAYS label 0
            label = 0  # Neutral - vertical barrier touched first
            barrier_type = "time"
            touch_date = date_window[-1] if len(date_window) > 0 else entry_date
            value_at_barrier = data_window['close'].iloc[-1] if len(data_window) > 0 else entry_price
        
        # Calculate actual return (for information only, not used for labeling)
        end_price = data.loc[touch_date, 'close']
        actual_return = (end_price - entry_price) / entry_price
        
        result.append({
            'decision_date': decision_date,  # D-1 (when decision is made)
            'entry_date': entry_date,       # D (when position is entered)
            'end_date': touch_date,         # When barrier is touched
            'entry_price': entry_price,     # Close price of D-1
            'end_price': end_price,         # Close price at barrier touch
            'return': actual_return,        # Actual return
            'upper_barrier': upper_barrier,
            'lower_barrier': lower_barrier,
            'barrier_touched': barrier_type,
            'value_at_barrier_touched': value_at_barrier,
            'label': label
        })
    
    return pd.DataFrame(result)

# Penggunaan kode:

# Jika date belum diset sebagai indeks
# filtered_df = filtered_df.set_index('date')

# Aplikasikan Triple Barrier Method
triple_barrier_df = triple_barrier_method(
    filtered_df,
    volatility_window=20,
    upper_barrier_multiplier=1.0,
    lower_barrier_multiplier=1.0,
    time_barrier_days=5  # Perhatikan: ini bisa berarti 5 periode (bukan hanya hari)
)

# Tampilkan hasil
# print(triple_barrier_df.head())

# Hitung distribusi label
label_counts = triple_barrier_df['label'].value_counts()
print("\nLabel Distribution:")
print(label_counts)
print(f"Percentage UP: {label_counts.get(1, 0)/len(triple_barrier_df)*100:.2f}%")
print(f"Percentage DOWN: {label_counts.get(-1, 0)/len(triple_barrier_df)*100:.2f}%")
print(f"Percentage NEUTRAL: {label_counts.get(0, 0)/len(triple_barrier_df)*100:.2f}%")

# %%
triple_barrier_df.to_csv("triple_barrier_results.csv", index=False)

# %%
triple_barrier_df.columns

# %% [markdown]
# ### Triple Barrier Visualization

# %%
filtered_df

# %%
# Print unique values in the "barrier_touched" column
unique_values = triple_barrier_df['barrier_touched'].unique()
print("Unique values in 'barrier_touched' column:", unique_values)

# %%
# Filter rows where barrier_touched is 'time'
time_barrier_df = triple_barrier_df[triple_barrier_df['barrier_touched'] == 'time']
lower_barrier_df = triple_barrier_df[triple_barrier_df['barrier_touched'] == 'lower']
upper_barrier_df = triple_barrier_df[triple_barrier_df['barrier_touched'] == 'upper']

# %%
lower_barrier_df

# %%
time_barrier_df

# %%
import plotly.graph_objects as go
import pandas as pd
import random

def visualize_triple_barrier_sample(data, triple_barrier_df, label_value, window_size=50):
    """
    Visualisasi satu sampel acak dengan label tertentu dari hasil Triple Barrier Method
    
    Parameters:
    - data: DataFrame OHLC original
    - triple_barrier_df: Hasil dari fungsi triple_barrier_method
    - label_value: Nilai label yang ingin divisualisasikan (1, -1, atau 0)
    - window_size: Jumlah data sebelum dan sesudah untuk ditampilkan
    
    Returns:
    - Objek figure Plotly untuk visualisasi
    """
    # Filter sampel dengan label yang ditentukan dan pilih secara acak
    label_samples = triple_barrier_df[triple_barrier_df['label'] == label_value]
    
    if len(label_samples) == 0:
        print(f"Tidak ada sampel dengan label {label_value}")
        return None
    
    # Pilih satu sampel secara acak
    random_sample = label_samples.sample(1).iloc[0]
    
    # Pastikan data memiliki indeks datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        if 'date' in data.columns:
            data = data.copy()
            data['date'] = pd.to_datetime(data['date'])
            data = data.set_index('date')
    
    # Periksa nama kolom yang tersedia untuk tanggal entry
    if 'entry_date' in random_sample:
        entry_date = pd.to_datetime(random_sample['entry_date'])
    elif 'decision_date' in random_sample:
        entry_date = pd.to_datetime(random_sample['decision_date'])
    else:
        # Fallback jika nama kolom berbeda
        date_cols = [col for col in random_sample.index if 'date' in col.lower() and col != 'end_date']
        if date_cols:
            entry_date = pd.to_datetime(random_sample[date_cols[0]])
        else:
            raise KeyError("Tidak dapat menemukan kolom tanggal entry")
    
    # Dapatkan indeks untuk tanggal entry
    entry_idx = data.index.get_indexer([entry_date], method='nearest')[0]
    
    # Tentukan range data untuk visualisasi dengan 50 data sebelum dan sesudah
    start_window = max(0, entry_idx - window_size)
    end_window = min(len(data), entry_idx + window_size)
    sample_data = data.iloc[start_window:end_window]
    
    # Dapatkan indeks untuk tanggal barrier tersentuh
    end_date = pd.to_datetime(random_sample['end_date'])
    end_idx_rel = sample_data.index.get_indexer([end_date], method='nearest')[0]
    
    # Buat plot candlestick
    fig = go.Figure()
    
    # Tambahkan candlestick
    fig.add_trace(go.Candlestick(
        x=sample_data.index,
        open=sample_data['open'],
        high=sample_data['high'],
        low=sample_data['low'],
        close=sample_data['close'],
        name='Price',
        increasing_line_color='green',
        decreasing_line_color='red'
    ))
    
    # Tambahkan barrier lines
    # Upper barrier
    fig.add_trace(go.Scatter(
        x=[sample_data.index[0], sample_data.index[-1]],
        y=[random_sample['upper_barrier'], random_sample['upper_barrier']],
        mode='lines',
        line=dict(color='green', width=2, dash='dash'),
        name='Upper Barrier'
    ))
    
    # Lower barrier
    fig.add_trace(go.Scatter(
        x=[sample_data.index[0], sample_data.index[-1]],
        y=[random_sample['lower_barrier'], random_sample['lower_barrier']],
        mode='lines',
        line=dict(color='red', width=2, dash='dash'),
        name='Lower Barrier'
    ))
    
    # Titik entry
    entry_idx_rel = sample_data.index.get_indexer([entry_date], method='nearest')[0]
    fig.add_trace(go.Scatter(
        x=[sample_data.index[entry_idx_rel]],
        y=[random_sample['entry_price']],
        mode='markers',
        marker=dict(color='blue', size=10, symbol='circle'),
        name='Entry Point'
    ))
    
    # Titik barrier tersentuh
    barrier_colors = {'upper': 'green', 'lower': 'red', 'time': 'purple'}
    touch_value = random_sample.get('value_at_barrier_touched', random_sample['end_price'])
    
    fig.add_trace(go.Scatter(
        x=[sample_data.index[end_idx_rel]],
        y=[touch_value],
        mode='markers',
        marker=dict(
            color=barrier_colors[random_sample['barrier_touched']], 
            size=10, 
            symbol='star'
        ),
        name=f"{random_sample['barrier_touched'].capitalize()} Barrier Touch"
    ))
    
    # Tambahkan anotasi
    fig.add_annotation(
        x=sample_data.index[entry_idx_rel],
        y=random_sample['entry_price'],
        text=f"Entry: {random_sample['entry_price']:.4f}",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-40
    )
    
    fig.add_annotation(
        x=sample_data.index[end_idx_rel],
        y=touch_value,
        text=f"Touch: {touch_value:.4f}",
        showarrow=True,
        arrowhead=2,
        ax=-40,
        ay=-40
    )
    
    # Highlight periode trading
    fig.add_shape(
        type="rect",
        x0=sample_data.index[entry_idx_rel],
        x1=sample_data.index[end_idx_rel],
        y0=0,
        y1=1,
        yref="paper",
        fillcolor="lightblue",
        opacity=0.2,
        line_width=0
    )
    
    # Definisi nama label
    label_names = {1: "Positif (1)", -1: "Negatif (-1)", 0: "Netral (0)"}
    
    # Update layout
    fig.update_layout(
        title=f"Triple Barrier Example - Label: {label_names.get(label_value)} (Barrier: {random_sample['barrier_touched']})",
        height=500,
        width=1000,
        template="plotly_white"
    )
    
    fig.update_xaxes(
        rangeslider_visible=False
    )
    
    return fig

# %%
# Visualisasi sampel dengan label 1 (positif)
try:
    fig_label_1 = visualize_triple_barrier_sample(filtered_df, triple_barrier_df, label_value=1, window_size=50)
    if fig_label_1:
        fig_label_1.show()
except Exception as e:
    print(f"Error saat memvisualisasikan label 1: {e}")
    # Menampilkan kolom yang tersedia jika terjadi error
    if not triple_barrier_df.empty:
        print("\nKolom yang tersedia dalam triple_barrier_df:")
        print(triple_barrier_df.columns.tolist())

# %%
# Visualisasi sampel dengan label -1 (negatif)
try:
    fig_label_minus_1 = visualize_triple_barrier_sample(filtered_df, triple_barrier_df, label_value=-1, window_size=50)
    if fig_label_minus_1:
        fig_label_minus_1.show()
except Exception as e:
    print(f"Error saat memvisualisasikan label -1: {e}")

# %%
# Visualisasi sampel dengan label 0 (netral)
try:
    fig_label_0 = visualize_triple_barrier_sample(filtered_df, triple_barrier_df, label_value=0, window_size=50)
    if fig_label_0:
        fig_label_0.show()
except Exception as e:
    print(f"Error saat memvisualisasikan label 0: {e}")


