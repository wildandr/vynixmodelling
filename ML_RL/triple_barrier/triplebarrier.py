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
    # Create a copy to avoid modifying original data
    data_copy = data.copy()

    # Ensure data has proper date column and index
    if 'date' in data_copy.columns and not isinstance(data_copy.index, pd.DatetimeIndex):
        data_copy['date'] = pd.to_datetime(data_copy['date'])
        data_copy = data_copy.set_index('date')
    
    result = []
    
    # Calculate daily returns based on close prices
    returns = data_copy['close'].pct_change().fillna(0)
    
    # Calculate rolling volatility
    volatility = returns.rolling(window=volatility_window).std().fillna(method='bfill')
    
    for i in range(len(data_copy) - time_barrier_days - 1):  # -1 because we need room for D-1
        # D-1 is the decision point, D is the entry point
        # Get actual datetime values from the index, not just the position
        decision_date = data_copy.index[i]  # This is now the actual datetime
        entry_date = data_copy.index[i+1]   # This is now the actual datetime
        entry_price = data_copy['close'].iloc[i]  # Use close price of D-1 as entry price
        
        # Set dynamic barriers based on volatility
        upper_barrier = entry_price * (1 + volatility.iloc[i] * upper_barrier_multiplier)
        lower_barrier = entry_price * (1 - volatility.iloc[i] * lower_barrier_multiplier)
        
        # Define the window to look for barrier touches, starting from D (not D-1)
        data_window = data_copy.iloc[i+1:i+1+time_barrier_days]
        
        if len(data_window) == 0:
            continue  # Skip if we don't have enough data for the time window
        
        # Check if and when barriers are touched using HIGH and LOW prices
        upper_touch_idx = None
        lower_touch_idx = None
        upper_touch_value = None
        lower_touch_value = None
        
        for j in range(len(data_window)):
            # Check if HIGH price touches upper barrier
            if data_window['high'].iloc[j] >= upper_barrier:
                upper_touch_idx = j
                upper_touch_value = upper_barrier  # Store the barrier value that was touched
                break
                
        for j in range(len(data_window)):
            # Check if LOW price touches lower barrier
            if data_window['low'].iloc[j] <= lower_barrier:
                lower_touch_idx = j
                lower_touch_value = lower_barrier  # Store the barrier value that was touched
                break
        
        # Determine which barrier was touched first
        if upper_touch_idx is not None and (lower_touch_idx is None or upper_touch_idx < lower_touch_idx):
            label = 1  # Up - upper barrier touched first
            barrier_type = "upper"
            touch_date = data_window.index[upper_touch_idx]  # Actual datetime from index
            value_at_barrier = upper_touch_value  # Use the barrier value that was touched
        elif lower_touch_idx is not None:
            label = -1  # Down - lower barrier touched first
            barrier_type = "lower" 
            touch_date = data_window.index[lower_touch_idx]  # Actual datetime from index
            value_at_barrier = lower_touch_value  # Use the barrier value that was touched
        else:
            # Time barrier touched - ALWAYS label 0
            label = 0  # Neutral - vertical barrier touched first
            barrier_type = "time"
            touch_date = data_window.index[-1] if len(data_window) > 0 else entry_date  # Actual datetime from index
            value_at_barrier = data_window['close'].iloc[-1] if len(data_window) > 0 else entry_price
        
        # Calculate actual return (for information only, not used for labeling)
        end_price = data_copy.loc[touch_date, 'close']
        actual_return = (end_price - entry_price) / entry_price
        
        result.append({
            'decision_date': decision_date,  # D-1 (when decision is made) - actual datetime
            'entry_date': entry_date,       # D (when position is entered) - actual datetime
            'end_date': touch_date,         # When barrier is touched - actual datetime
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

def apply_triple_barrier_labeling(data, 
                                 volatility_window=20, 
                                 upper_barrier_multiplier=1.0, 
                                 lower_barrier_multiplier=1.0, 
                                 time_barrier_days=5,
                                 verbose=True):
    """
    Fungsi wrapper untuk menerapkan Triple Barrier Method dengan logging.
    
    Parameters:
    - data: DataFrame dengan kolom OHLC dan datetime index atau kolom date
    - volatility_window: Window untuk menghitung volatilitas (default: 20)
    - upper_barrier_multiplier: Multiplier untuk upper barrier (default: 1.0)
    - lower_barrier_multiplier: Multiplier untuk lower barrier (default: 1.0)
    - time_barrier_days: Maksimum periode untuk menunggu barrier touch (default: 5)
    - verbose: Apakah menampilkan hasil statistik (default: True)
    
    Returns:
    - DataFrame hasil Triple Barrier Method dengan kolom label
    """
    # Aplikasikan Triple Barrier Method
    triple_barrier_df = triple_barrier_method(
        data,
        volatility_window=volatility_window,
        upper_barrier_multiplier=upper_barrier_multiplier,
        lower_barrier_multiplier=lower_barrier_multiplier,
        time_barrier_days=time_barrier_days
    )
    
    if verbose:
        print("\n=== Triple Barrier Method Results ===")
        print(f"Total samples generated: {len(triple_barrier_df)}")
        print("\nFirst few rows:")
        print(triple_barrier_df.head())
        
        # Hitung distribusi label
        label_counts = triple_barrier_df['label'].value_counts()
        print("\nLabel Distribution:")
        print(label_counts)
        print(f"Percentage UP (1): {label_counts.get(1, 0)/len(triple_barrier_df)*100:.2f}%")
        print(f"Percentage DOWN (-1): {label_counts.get(-1, 0)/len(triple_barrier_df)*100:.2f}%")
        print(f"Percentage NEUTRAL (0): {label_counts.get(0, 0)/len(triple_barrier_df)*100:.2f}%")
        
        # Statistik tambahan
        print(f"\nBarrier Touch Statistics:")
        barrier_counts = triple_barrier_df['barrier_touched'].value_counts()
        for barrier_type, count in barrier_counts.items():
            print(f"{barrier_type.capitalize()}: {count} ({count/len(triple_barrier_df)*100:.2f}%)")
    
    return triple_barrier_df