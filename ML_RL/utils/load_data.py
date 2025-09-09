import json
import pandas as pd
import os
from datetime import datetime

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