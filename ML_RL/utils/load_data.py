import json
import pandas as pd
import os
from datetime import datetime
import glob

def process_fundamental_data(data, ticker):
    if data is None:
        print(f"No fundamental data available for {ticker}. Skipping processing.")
        return
    # Function to extract quarterly data for a given metric
    def extract_quarterly_data(metric_name):
        quarterly_data = []
        
        if metric_name in data["facts"]["us-gaap"] and "units" in data["facts"]["us-gaap"][metric_name]:
            metric_data = data["facts"]["us-gaap"][metric_name]
            
            # Most financial data is in USD
            if "USD" in metric_data["units"]:
                # Track used period keys to avoid duplicates
                used_periods = set()
                
                # Sort entries by filed date (latest first) to get most recent values
                sorted_entries = sorted(
                    metric_data["units"]["USD"],
                    key=lambda x: x.get("filed", "0000-00-00"),
                    reverse=True
                )
                
                for entry in sorted_entries:
                    # Check if fp exists, is not None, and starts with Q or is FY with form 10-K
                    if ("fp" in entry and 
                        entry["fp"] is not None and 
                        isinstance(entry["fp"], str) and 
                        (entry["fp"].startswith("Q") or 
                         (entry["fp"] == "FY" and entry.get("form") == "10-K")) and 
                        "fy" in entry and 
                        "val" in entry):
                        
                        # Treat FY as Q4 if form is 10-K
                        period_fp = "Q4" if entry["fp"] == "FY" else entry["fp"]
                        period_key = f"{entry['fy']}-{period_fp}"
                        
                        # Skip if we already have this period for this metric
                        if period_key in used_periods:
                            continue
                            
                        used_periods.add(period_key)
                        
                        quarterly_data.append({
                            "date": period_key,
                            "value": entry["val"],
                            "metric": metric_name,
                            "filed_date": entry.get("filed", "")
                        })
        
        return quarterly_data

    # Get all available metrics
    all_available_metrics = list(data["facts"]["us-gaap"].keys())
    print(f"Found {len(all_available_metrics)} total metrics in the data")

    # Collect all quarterly data
    all_quarterly_data = []
    metrics_with_data = 0

    # Process each metric and collect data
    for i, metric in enumerate(all_available_metrics):
        metric_data = extract_quarterly_data(metric)
        
        if metric_data:
            all_quarterly_data.extend(metric_data)
            metrics_with_data += 1
        
        # Print progress every 100 metrics
        # if (i + 1) % 100 == 0 or i == len(all_available_metrics) - 1:
        #     print(f"Processed {i+1}/{len(all_available_metrics)} metrics")

    print(f"Found data for {metrics_with_data} metrics out of {len(all_available_metrics)} total metrics")

    # Create DataFrame
    if all_quarterly_data:
        df = pd.DataFrame(all_quarterly_data)
        
        # Verify no duplicates in date-metric combinations
        duplicate_check = df.duplicated(subset=['date', 'metric'])
        if duplicate_check.any():
            print(f"Warning: Found {duplicate_check.sum()} duplicate entries. Keeping only the first occurrence.")
            df = df.drop_duplicates(subset=['date', 'metric'])
        
        # Pivot the data to get metrics as columns
        pivoted_df = df.pivot(index='date', columns='metric', values='value')
        
        # Sort by date (assuming format YYYY-QX)
        pivoted_df = pivoted_df.sort_index()
        
        # Create filename with ticker
        filename = f"/root/vynixmodelling/dataset/data_fundamental/{ticker}_time.csv"
        
        # Save to CSV
        pivoted_df.to_csv(filename)
        print(f"Data saved to {filename}")
        print(f"Saved {len(pivoted_df)} quarters of data for {len(pivoted_df.columns)} metrics")
        
        # Show statistics
        print(f"\nDataFrame statistics:")
        print(f"Number of quarters: {len(pivoted_df)}")
        print(f"Number of metrics: {len(pivoted_df.columns)}")
        print(f"Number of data points: {pivoted_df.count().sum()}")
        print(f"Data completeness: {(pivoted_df.count().sum() / (len(pivoted_df) * len(pivoted_df.columns)) * 100):.2f}%")
        
        
        # Display the first few rows
        return pivoted_df
    return None
        # print("\nSample of saved data (first 5 rows, first 5 columns):")
        # print(pivoted_df.iloc[:5, :5])
    # else:
    #     print("No quarterly data found for any metrics")

def get_fundamental_data_local(ticker, cik=None):
    """
    Membaca data fundamental dari file lokal berdasarkan CIK.
    
    Parameters:
    - ticker: Symbol ticker (untuk penamaan file output)
    - cik: CIK number (jika tidak diberikan, akan dicari berdasarkan ticker)
    
    Returns:
    - Dictionary data fundamental atau None jika tidak ditemukan
    """
    # Jika CIK tidak diberikan, coba cari berdasarkan ticker
    if cik is None:
        # Untuk Tesla, CIK-nya adalah 1318605
        if ticker.upper() == "TSLA":
            cik = "1318605"
        else:
            print(f"CIK untuk ticker {ticker} tidak diketahui. Silakan berikan CIK secara manual.")
            return None
    
    # Format CIK dengan padding zeros
    if isinstance(cik, str) and not cik.startswith("CIK"):
        cik_formatted = f"CIK{cik.zfill(10)}"
    elif isinstance(cik, int):
        cik_formatted = f"CIK{str(cik).zfill(10)}"
    else:
        cik_formatted = cik
    
    # Path ke direktori sec_data
    sec_data_dir = "/root/vynixmodelling/dataset/data_fundamental/sec_data"
    
    # Cari file berdasarkan CIK
    json_file_path = os.path.join(sec_data_dir, f"{cik_formatted}.json")
    
    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            print(f"Data fundamental untuk {ticker} (CIK: {cik_formatted}) berhasil dimuat dari file lokal.")
            print(f"File: {json_file_path}")
            return data
        except Exception as e:
            print(f"Error membaca file {json_file_path}: {e}")
            return None
    else:
        print(f"File tidak ditemukan: {json_file_path}")
        # Coba cari file dengan pattern yang mirip
        pattern = os.path.join(sec_data_dir, f"*{cik}*.json")
        matching_files = glob.glob(pattern)
        if matching_files:
            print(f"File yang mungkin cocok ditemukan: {matching_files}")
        return None

def list_available_cik_files():
    """
    Menampilkan daftar file CIK yang tersedia di direktori sec_data.
    
    Returns:
    - List of available CIK files
    """
    sec_data_dir = "/root/vynixmodelling/dataset/data_fundamental/sec_data"
    
    if not os.path.exists(sec_data_dir):
        print(f"Direktori {sec_data_dir} tidak ditemukan.")
        return []
    
    json_files = glob.glob(os.path.join(sec_data_dir, "CIK*.json"))
    cik_list = []
    
    for file_path in json_files:
        filename = os.path.basename(file_path)
        cik = filename.replace("CIK", "").replace(".json", "")
        cik_list.append({
            'cik': cik,
            'filename': filename,
            'path': file_path
        })
    
    print(f"Ditemukan {len(cik_list)} file CIK di {sec_data_dir}")
    return cik_list

def get_cik_from_local_file(file_path):
    """
    Mendapatkan informasi CIK dan entity name dari file lokal.
    
    Parameters:
    - file_path: Path ke file JSON
    
    Returns:
    - Dictionary dengan informasi CIK dan entity name
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return {
            'cik': data.get('cik'),
            'entityName': data.get('entityName'),
            'file_path': file_path
        }
    except Exception as e:
        print(f"Error membaca file {file_path}: {e}")
        return None

def process_fundamental_data_local(ticker, cik=None, output_dir="/root/vynixmodelling/dataset/data_fundamental"):
    """
    Memproses data fundamental dari file lokal berdasarkan CIK dan menyimpannya ke CSV.
    
    Parameters:
    - ticker: Symbol ticker
    - cik: CIK number (opsional)
    - output_dir: Direktori output untuk menyimpan CSV
    
    Returns:
    - Path ke file CSV yang dihasilkan atau None jika gagal
    """
    # Ambil data dari file lokal
    data = get_fundamental_data_local(ticker, cik)
    
    if data is None:
        print(f"Gagal memuat data untuk ticker {ticker}")
        return None
    
    # Proses data menggunakan fungsi yang sudah ada
    try:
        result_path = process_fundamental_data(data, ticker)
        print(f"Data fundamental untuk {ticker} berhasil diproses dari file lokal.")
        return result_path
    except Exception as e:
        print(f"Error memproses data fundamental untuk {ticker}: {e}")
        return None