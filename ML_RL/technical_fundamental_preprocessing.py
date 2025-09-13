import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalFundamentalPreprocessor:
    """
    Modul untuk preprocessing dan penggabungan data teknikal dan fundamental.
    
    Kelas ini menyediakan fungsi-fungsi untuk:
    - Konversi data quarterly ke format harian
    - Penggabungan data teknikal (harian) dengan data fundamental (quarterly)
    - Filter data berdasarkan periode waktu
    - Penyimpanan data yang telah diproses
    """
    
    def __init__(self):
        self.technical_data = None
        self.fundamental_data = None
        self.combined_data = None
        
    def prepare_technical_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mempersiapkan data teknikal dari DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame dengan data teknikal
            
        Returns:
            pd.DataFrame: DataFrame dengan data teknikal yang telah dipersiapkan
            
        Raises:
            ValueError: Jika format data tidak sesuai
        """
        try:
            logger.info("Preparing technical data")
            df_clean = df.copy()
            
            # Validasi kolom yang diperlukan
            required_columns = ['time', 'open', 'high', 'low', 'close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df_clean.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Konversi timestamp ke datetime jika belum ada
            if 'datetime' not in df_clean.columns:
                df_clean['datetime'] = pd.to_datetime(df_clean['time'], unit='s')
            if 'date' not in df_clean.columns:
                df_clean['date'] = df_clean['datetime'].dt.date
            
            logger.info(f"Technical data prepared: {len(df_clean)} rows, {len(df_clean.columns)} columns")
            
            self.technical_data = df_clean
            return df_clean
            
        except Exception as e:
            logger.error(f"Error preparing technical data: {str(e)}")
            raise ValueError(f"Failed to prepare technical data: {str(e)}")
    
    def prepare_fundamental_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mempersiapkan data fundamental dari DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame dengan data fundamental
            
        Returns:
            pd.DataFrame: DataFrame dengan data fundamental yang telah dipersiapkan
            
        Raises:
            ValueError: Jika format data tidak sesuai
        """
        try:
            logger.info("Preparing fundamental data")
            df_clean = df.copy()
            
            # Validasi bahwa index adalah quarter format
            if not all(isinstance(idx, str) and '-Q' in str(idx) for idx in df_clean.index[:5]):
                logger.warning("Index might not be in quarter format (YYYY-QX)")
            
            logger.info(f"Fundamental data prepared: {len(df_clean)} quarters, {len(df_clean.columns)} features")
            
            self.fundamental_data = df_clean
            return df_clean
            
        except Exception as e:
            logger.error(f"Error preparing fundamental data: {str(e)}")
            raise ValueError(f"Failed to prepare fundamental data: {str(e)}")
    
    def quarter_to_daily_conversion(self, fundamental_df: pd.DataFrame, 
                                  start_date: str = '2012-04-01', 
                                  end_date: str = '2025-06-30') -> pd.DataFrame:
        """
        Mengkonversi data fundamental quarterly menjadi format harian.
        
        Args:
            fundamental_df (pd.DataFrame): DataFrame dengan data fundamental quarterly
            start_date (str): Tanggal mulai dalam format 'YYYY-MM-DD'
            end_date (str): Tanggal akhir dalam format 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: DataFrame dengan data fundamental dalam format harian
            
        Raises:
            ValueError: Jika format data tidak sesuai
        """
        try:
            logger.info("Converting quarterly data to daily format")
            
            # Create date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Create mapping from quarter to date ranges
            quarter_mapping = {}
            for quarter in fundamental_df.index:
                try:
                    year, q = quarter.split('-Q')
                    year = int(year)
                    q = int(q)
                    
                    # Define quarter start and end dates
                    if q == 1:
                        q_start = pd.Timestamp(year, 1, 1)
                        q_end = pd.Timestamp(year, 3, 31)
                    elif q == 2:
                        q_start = pd.Timestamp(year, 4, 1)
                        q_end = pd.Timestamp(year, 6, 30)
                    elif q == 3:
                        q_start = pd.Timestamp(year, 7, 1)
                        q_end = pd.Timestamp(year, 9, 30)
                    else:  # q == 4
                        q_start = pd.Timestamp(year, 10, 1)
                        q_end = pd.Timestamp(year, 12, 31)
                    
                    quarter_mapping[quarter] = (q_start, q_end)
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping invalid quarter format: {quarter}")
                    continue
            
            # Create daily DataFrame with optimized approach
            daily_data_dict = {}
            
            # Fill daily data with quarterly values using optimized approach
            for col in fundamental_df.columns:
                daily_values = pd.Series(index=date_range, dtype=float)
                
                for quarter, (q_start, q_end) in quarter_mapping.items():
                    if quarter in fundamental_df.index:
                        # Get dates within this quarter that are in our date range
                        quarter_dates = date_range[(date_range >= q_start) & (date_range <= q_end)]
                        
                        # Fill with quarterly value
                        if len(quarter_dates) > 0:
                            daily_values.loc[quarter_dates] = fundamental_df.loc[quarter, col]
                
                daily_data_dict[col] = daily_values
            
            # Create DataFrame from dictionary (more efficient)
            daily_data = pd.DataFrame(daily_data_dict)
            
            # Forward fill missing values (carry forward quarterly data)
            daily_data = daily_data.ffill()
            
            logger.info(f"Daily conversion completed: {len(daily_data)} days, {len(daily_data.columns)} features")
            
            return daily_data
            
        except Exception as e:
            logger.error(f"Error in quarter to daily conversion: {str(e)}")
            raise ValueError(f"Failed to convert quarterly to daily data: {str(e)}")
    
    def combine_technical_fundamental(self, technical_df: pd.DataFrame, 
                                   fundamental_daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Menggabungkan data teknikal dan fundamental.
        
        Args:
            technical_df (pd.DataFrame): DataFrame dengan data teknikal harian
            fundamental_daily_df (pd.DataFrame): DataFrame dengan data fundamental harian
            
        Returns:
            pd.DataFrame: DataFrame gabungan teknikal dan fundamental
            
        Raises:
            ValueError: Jika penggabungan gagal
        """
        try:
            logger.info("Combining technical and fundamental data")
            
            # Ensure technical data has date column
            if 'date' not in technical_df.columns:
                if 'datetime' in technical_df.columns:
                    technical_df['date'] = technical_df['datetime'].dt.date
                elif 'time' in technical_df.columns:
                    technical_df['datetime'] = pd.to_datetime(technical_df['time'], unit='s')
                    technical_df['date'] = technical_df['datetime'].dt.date
                else:
                    raise ValueError("No date/time column found in technical data")
            
            # Convert fundamental index to date for merging
            fundamental_daily_df = fundamental_daily_df.copy()
            fundamental_daily_df['date'] = fundamental_daily_df.index.date
            
            # Merge on date
            combined_df = pd.merge(technical_df, fundamental_daily_df, 
                                 on='date', how='inner', suffixes=('_tech', '_fund'))
            
            # Remove duplicate date columns and reorganize
            if 'date_fund' in combined_df.columns:
                combined_df = combined_df.drop('date_fund', axis=1)
            
            logger.info(f"Data combination completed: {len(combined_df)} rows, {len(combined_df.columns)} columns")
            logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
            
            self.combined_data = combined_df
            return combined_df
            
        except Exception as e:
            logger.error(f"Error combining data: {str(e)}")
            raise ValueError(f"Failed to combine technical and fundamental data: {str(e)}")
    
    def filter_by_period(self, df: pd.DataFrame, 
                        start_period: str = '2012-Q2', 
                        end_period: str = '2025-Q2') -> pd.DataFrame:
        """
        Memfilter data berdasarkan periode waktu.
        
        Args:
            df (pd.DataFrame): DataFrame yang akan difilter
            start_period (str): Periode mulai dalam format 'YYYY-QX'
            end_period (str): Periode akhir dalam format 'YYYY-QX'
            
        Returns:
            pd.DataFrame: DataFrame yang telah difilter
            
        Raises:
            ValueError: Jika format periode tidak sesuai
        """
        try:
            logger.info(f"Filtering data from {start_period} to {end_period}")
            
            # Convert period to dates
            def period_to_date(period: str, is_start: bool = True) -> pd.Timestamp:
                year, q = period.split('-Q')
                year = int(year)
                q = int(q)
                
                if q == 1:
                    return pd.Timestamp(year, 1, 1) if is_start else pd.Timestamp(year, 3, 31)
                elif q == 2:
                    return pd.Timestamp(year, 4, 1) if is_start else pd.Timestamp(year, 6, 30)
                elif q == 3:
                    return pd.Timestamp(year, 7, 1) if is_start else pd.Timestamp(year, 9, 30)
                else:  # q == 4
                    return pd.Timestamp(year, 10, 1) if is_start else pd.Timestamp(year, 12, 31)
            
            start_date = period_to_date(start_period, True)
            end_date = period_to_date(end_period, False)
            
            # Ensure date column exists and is datetime
            if 'date' in df.columns:
                df_filtered = df.copy()
                df_filtered['date'] = pd.to_datetime(df_filtered['date'])
                
                # Filter by date range
                mask = (df_filtered['date'] >= start_date) & (df_filtered['date'] <= end_date)
                df_filtered = df_filtered[mask]
                
            elif 'datetime' in df.columns:
                df_filtered = df.copy()
                mask = (df_filtered['datetime'] >= start_date) & (df_filtered['datetime'] <= end_date)
                df_filtered = df_filtered[mask]
                
            else:
                raise ValueError("No date or datetime column found for filtering")
            
            logger.info(f"Filtering completed: {len(df_filtered)} rows remaining")
            
            return df_filtered
            
        except Exception as e:
            logger.error(f"Error filtering data: {str(e)}")
            raise ValueError(f"Failed to filter data by period: {str(e)}")
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Menyimpan data yang telah diproses ke file CSV.
        
        Args:
            df (pd.DataFrame): DataFrame yang akan disimpan
            output_path (str): Path output file
            
        Raises:
            IOError: Jika gagal menyimpan file
        """
        try:
            logger.info(f"Saving processed data to: {output_path}")
            
            # Save main data
            df.to_csv(output_path, index=False)
            
            logger.info(f"Data saved successfully: {len(df)} rows, {len(df.columns)} columns")
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise IOError(f"Failed to save processed data: {str(e)}")
    
    def process_complete_pipeline(self, 
                                technical_df: pd.DataFrame,
                                fundamental_df: pd.DataFrame,
                                output_path: str = None,
                                start_period: str = '2012-Q2',
                                end_period: str = '2025-Q2') -> pd.DataFrame:
        """
        Menjalankan pipeline lengkap preprocessing data.
        
        Args:
            technical_df (pd.DataFrame): DataFrame data teknikal
            fundamental_df (pd.DataFrame): DataFrame data fundamental
            output_path (str, optional): Path output file (opsional)
            start_period (str): Periode mulai
            end_period (str): Periode akhir
            
        Returns:
            pd.DataFrame: DataFrame hasil preprocessing
        """
        try:
            logger.info("Starting complete preprocessing pipeline")
            
            # Step 1: Prepare data
            technical_prepared = self.prepare_technical_data(technical_df)
            fundamental_prepared = self.prepare_fundamental_data(fundamental_df)
            
            # Step 2: Convert quarterly to daily
            start_date = '2012-04-01' if start_period == '2012-Q2' else '2012-01-01'
            end_date = '2025-06-30' if end_period == '2025-Q2' else '2025-12-31'
            
            fundamental_daily = self.quarter_to_daily_conversion(
                fundamental_prepared, start_date, end_date
            )
            
            # Step 3: Combine data
            combined_df = self.combine_technical_fundamental(technical_prepared, fundamental_daily)
            
            # Step 4: Filter by period
            filtered_df = self.filter_by_period(combined_df, start_period, end_period)
            
            # Step 5: Save processed data (optional)
            if output_path:
                self.save_processed_data(filtered_df, output_path)
            
            logger.info("Complete preprocessing pipeline finished successfully")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


# Convenience functions for main.py integration
def preprocess_technical_fundamental_data(technical_df: pd.DataFrame,
                                        fundamental_df: pd.DataFrame,
                                        output_path: str = None,
                                        start_period: str = '2012-Q2',
                                        end_period: str = '2025-Q2') -> pd.DataFrame:
    """
    Fungsi utama untuk preprocessing data teknikal dan fundamental.
    
    Args:
        technical_df (pd.DataFrame): DataFrame data teknikal
        fundamental_df (pd.DataFrame): DataFrame data fundamental
        output_path (str, optional): Path output file (opsional)
        start_period (str): Periode mulai dalam format 'YYYY-QX'
        end_period (str): Periode akhir dalam format 'YYYY-QX'
        
    Returns:
        pd.DataFrame: DataFrame hasil preprocessing
    """
    preprocessor = TechnicalFundamentalPreprocessor()
    return preprocessor.process_complete_pipeline(
        technical_df, fundamental_df, output_path, start_period, end_period
    )


def convert_quarterly_to_daily(fundamental_df: pd.DataFrame,
                             start_date: str = '2012-04-01',
                             end_date: str = '2025-06-30') -> pd.DataFrame:
    """
    Fungsi untuk konversi data quarterly ke harian.
    
    Args:
        fundamental_df (pd.DataFrame): DataFrame dengan data fundamental quarterly
        start_date (str): Tanggal mulai
        end_date (str): Tanggal akhir
        
    Returns:
        pd.DataFrame: DataFrame dengan data fundamental harian
    """
    preprocessor = TechnicalFundamentalPreprocessor()
    return preprocessor.quarter_to_daily_conversion(fundamental_df, start_date, end_date)


def combine_data(technical_df: pd.DataFrame, 
               fundamental_daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fungsi untuk menggabungkan data teknikal dan fundamental.
    
    Args:
        technical_df (pd.DataFrame): DataFrame data teknikal
        fundamental_daily_df (pd.DataFrame): DataFrame data fundamental harian
        
    Returns:
        pd.DataFrame: DataFrame gabungan
    """
    preprocessor = TechnicalFundamentalPreprocessor()
    return preprocessor.combine_technical_fundamental(technical_df, fundamental_daily_df)


def filter_data_by_period(df: pd.DataFrame,
                         start_period: str = '2012-Q2',
                         end_period: str = '2025-Q2') -> pd.DataFrame:
    """
    Fungsi untuk memfilter data berdasarkan periode.
    
    Args:
        df (pd.DataFrame): DataFrame yang akan difilter
        start_period (str): Periode mulai
        end_period (str): Periode akhir
        
    Returns:
        pd.DataFrame: DataFrame yang telah difilter
    """
    preprocessor = TechnicalFundamentalPreprocessor()
    return preprocessor.filter_by_period(df, start_period, end_period)


if __name__ == "__main__":
    # Example usage - load sample data for testing
    try:
        # Load sample data for testing
        technical_df = pd.read_csv("/root/vynixmodelling/dataset/TSLA_original.csv")
        fundamental_df = pd.read_csv("/root/vynixmodelling/dataset/data_fundamental/TSLA_enhanced_features.csv", index_col=0)
        
        result_df = preprocess_technical_fundamental_data(
            technical_df, fundamental_df
        )
        print(f"Preprocessing completed successfully!")
        print(f"Result shape: {result_df.shape}")
        print(f"Date range: {result_df['date'].min()} to {result_df['date'].max()}")
        
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")