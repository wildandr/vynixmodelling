import plotly.graph_objects as go
import pandas as pd
import random
import os
from datetime import datetime

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

def generate_triple_barrier_visualizations(data, triple_barrier_df, 
                                         output_dir='/root/vynixmodelling/ML_RL/logs/visualization',
                                         window_size=50, 
                                         save_html=True,
                                         save_png=True,
                                         verbose=True):
    """
    Generate dan simpan visualisasi Triple Barrier Method untuk semua label.
    
    Parameters:
    - data: DataFrame OHLC original
    - triple_barrier_df: Hasil dari fungsi triple_barrier_method
    - output_dir: Directory untuk menyimpan hasil visualisasi
    - window_size: Jumlah data sebelum dan sesudah untuk ditampilkan
    - save_html: Apakah menyimpan dalam format HTML (default: True)
    - save_png: Apakah menyimpan dalam format PNG (default: False)
    - verbose: Apakah menampilkan log proses (default: True)
    
    Returns:
    - Dictionary dengan path file yang disimpan
    """
    # Buat directory jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp untuk nama file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    saved_files = {}
    labels_to_visualize = [1, -1, 0]
    label_names = {1: "positif", -1: "negatif", 0: "netral"}
    
    if verbose:
        print("\n=== Generating Triple Barrier Visualizations ===")
    
    for label_value in labels_to_visualize:
        try:
            fig = visualize_triple_barrier_sample(data, triple_barrier_df, 
                                                 label_value=label_value, 
                                                 window_size=window_size)
            
            if fig:
                label_name = label_names[label_value]
                base_filename = f"visualisasi_{timestamp}_label_{label_name}"
                
                # Simpan sebagai HTML
                if save_html:
                    html_path = os.path.join(output_dir, f"{base_filename}.html")
                    fig.write_html(html_path)
                    saved_files[f'label_{label_value}_html'] = html_path
                    if verbose:
                        print(f"Saved HTML: {html_path}")
                
                # Simpan sebagai PNG (memerlukan kaleido)
                if save_png:
                    try:
                        png_path = os.path.join(output_dir, f"{base_filename}.png")
                        fig.write_image(png_path, width=1200, height=600)
                        saved_files[f'label_{label_value}_png'] = png_path
                        if verbose:
                            print(f"Saved PNG: {png_path}")
                    except Exception as png_error:
                        if verbose:
                            print(f"Warning: Could not save PNG for label {label_value}: {png_error}")
                            print("Note: PNG export requires 'kaleido' package: pip install kaleido")
            
        except Exception as e:
            if verbose:
                print(f"Error saat memvisualisasikan label {label_value}: {e}")
                # Menampilkan kolom yang tersedia jika terjadi error
                if not triple_barrier_df.empty:
                    print(f"Kolom yang tersedia dalam triple_barrier_df: {triple_barrier_df.columns.tolist()}")
    
    # Simpan summary file
    summary_path = os.path.join(output_dir, f"visualisasi_{timestamp}_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Triple Barrier Visualization Summary\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Total samples: {len(triple_barrier_df)}\n")
        f.write(f"Window size: {window_size}\n")
        f.write(f"\nLabel Distribution:\n")
        
        label_counts = triple_barrier_df['label'].value_counts()
        for label, count in label_counts.items():
            percentage = count / len(triple_barrier_df) * 100
            f.write(f"Label {label}: {count} samples ({percentage:.2f}%)\n")
        
        f.write(f"\nGenerated Files:\n")
        for key, path in saved_files.items():
            f.write(f"{key}: {path}\n")
    
    saved_files['summary'] = summary_path
    
    if verbose:
        print(f"\nVisualization complete! Summary saved to: {summary_path}")
        print(f"Total files generated: {len(saved_files)}")
    
    return saved_files