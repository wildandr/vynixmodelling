# XGBoost Model Inference Script
import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
import sys
import os
import logging

# Add the ML_RL directory to the path to import feature engineering functions
sys.path.append('/root/vynixmodelling/ML_RL')
# from fundamental_feature_engineering import apply_feature_engineering

warnings.filterwarnings('ignore')

class XGBoostInference:
    def __init__(self, model_path=None, label_mapping_path=None):
        """
        Initialize XGBoost inference class
        
        Args:
            model_path (str): Path to the trained XGBoost model (.pkl file)
            label_mapping_path (str): Path to the label mapping file (.pkl file)
        """
        # Default paths based on the training script structure
        if model_path is None:
            model_path = '/root/vynixmodelling/ML_RL/logs/xgb_models/final_tuned_model.pkl'
        if label_mapping_path is None:
            label_mapping_path = '/root/vynixmodelling/ML_RL/logs/xgb_models/label_mappings.pkl'
            
        self.model_path = model_path
        self.label_mapping_path = label_mapping_path
        self.model = None
        self.label_mapping = None
        self.reverse_label_mapping = None
        self.feature_columns = None
        self.raw_feature_columns = None
        self.technical_columns = None
        
        # Load model and mappings
        self._load_model()
        self._load_label_mappings()
        self._load_feature_columns()
        self._load_raw_feature_columns()
        self._load_technical_columns()
    
    def _load_model(self):
        """Load the trained XGBoost model"""
        try:
            self.model = joblib.load(self.model_path)
            print(f"Model loaded successfully from: {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _load_label_mappings(self):
        """Load label mappings for converting predictions back to original labels"""
        try:
            mappings = joblib.load(self.label_mapping_path)
            self.label_mapping = mappings['label_mapping']
            self.reverse_label_mapping = mappings['reverse_label_mapping']
            print(f"Label mappings loaded successfully from: {self.label_mapping_path}")
            print(f"Label mapping: {self.label_mapping}")
        except Exception as e:
            print(f"Error loading label mappings: {e}")
            # Default mapping based on training logs
            self.label_mapping = {-1: 0, 0: 1, 1: 2}
            self.reverse_label_mapping = {0: -1, 1: 0, 2: 1}
            print("Using default label mappings")
    
    def _load_feature_columns(self):
        """Load feature column names from training data"""
        try:
            # Load feature columns from training data
            train_data_path = '/root/vynixmodelling/dataset/training_data/X_train.csv'
            if Path(train_data_path).exists():
                sample_df = pd.read_csv(train_data_path, nrows=1)
                self.feature_columns = sample_df.columns.tolist()
                print(f"Feature columns loaded: {len(self.feature_columns)} features")
            else:
                print("Training data not found, using default feature columns")
                self._set_default_feature_columns()
        except Exception as e:
            print(f"Error loading feature columns: {e}")
            self._set_default_feature_columns()
    
    def _set_default_feature_columns(self):
        """Set default feature columns when training data is not available"""
        # Default feature columns based on common financial features
        self.feature_columns = [
            'current_ratio', 'quick_ratio', 'cash_ratio', 'debt_to_equity',
            'debt_to_assets', 'interest_coverage', 'asset_turnover', 'inventory_turnover',
            'receivables_turnover', 'roa', 'roe', 'gross_margin', 'operating_margin',
            'net_margin', 'revenue_growth', 'earnings_growth', 'book_value_per_share',
            'price_to_book', 'working_capital', 'operating_cash_flow_ratio',
            'free_cash_flow', 'cash_conversion_cycle', 'days_sales_outstanding',
            'days_inventory_outstanding', 'days_payable_outstanding', 'financial_leverage',
            'equity_multiplier', 'times_interest_earned', 'cash_coverage_ratio',
            'operating_cash_flow_to_sales', 'capex_to_sales', 'dividend_payout_ratio',
            'retention_ratio', 'sustainable_growth_rate', 'altman_z_score',
            'piotroski_score', 'beneish_m_score', 'revenue_per_employee',
            'asset_quality_ratio', 'accruals_ratio', 'operating_leverage',
            'financial_leverage_index', 'asset_coverage_ratio', 'tangible_book_value_per_share',
            'working_capital_turnover', 'fixed_asset_turnover', 'total_asset_turnover',
            'inventory_to_sales', 'sales_to_working_capital', 'long_term_debt_to_equity',
            'short_term_debt_to_equity', 'total_debt_to_total_capital', 'capitalization_ratio',
            'long_term_debt_to_total_assets', 'debt_service_coverage', 'ebitda_coverage_ratio',
            'cash_debt_coverage', 'operating_cash_flow_coverage', 'reinvestment_rate',
            'earnings_retention_rate', 'plowback_ratio', 'internal_growth_rate',
            'dupont_roe', 'net_profit_margin', 'total_asset_turnover_dupont',
            'equity_multiplier_dupont', 'return_on_invested_capital', 'economic_value_added',
            'market_value_added', 'tobin_q_ratio', 'enterprise_value_to_ebitda',
            'price_to_sales', 'price_to_cash_flow', 'ev_to_sales', 'ev_to_ebit',
            'peg_ratio', 'dividend_yield', 'earnings_yield', 'free_cash_flow_yield',
            'book_to_market', 'sales_per_share', 'cash_per_share', 'operating_cash_flow_per_share'
        ]
        print(f"Using default feature columns: {len(self.feature_columns)} features")
    
    def _apply_basic_feature_engineering(self, raw_df, ticker="TSLA"):
        """
        Apply basic feature engineering to raw fundamental data
        
        Args:
            raw_df (pd.DataFrame): Raw fundamental data
            ticker (str): Stock ticker symbol
            
        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        try:
            # Create a copy of the raw data
            df = raw_df.copy()
            
            # Basic financial ratios
            # Current Ratio
            df['current_ratio'] = df.get('AssetsCurrent', 0) / df.get('LiabilitiesCurrent', 1)
            
            # Quick Ratio
            df['quick_ratio'] = (df.get('AssetsCurrent', 0) - df.get('InventoryNet', 0)) / df.get('LiabilitiesCurrent', 1)
            
            # Cash Ratio
            df['cash_ratio'] = df.get('CashAndCashEquivalentsAtCarryingValue', 0) / df.get('LiabilitiesCurrent', 1)
            
            # Debt to Equity
            df['debt_to_equity'] = df.get('Liabilities', 0) / df.get('StockholdersEquity', 1)
            
            # Debt to Assets
            df['debt_to_assets'] = df.get('Liabilities', 0) / df.get('Assets', 1)
            
            # Asset Turnover
            df['asset_turnover'] = df.get('Revenues', 0) / df.get('Assets', 1)
            
            # ROA (Return on Assets)
            df['roa'] = df.get('NetIncomeLoss', 0) / df.get('Assets', 1)
            
            # ROE (Return on Equity)
            df['roe'] = df.get('NetIncomeLoss', 0) / df.get('StockholdersEquity', 1)
            
            # Gross Margin
            df['gross_margin'] = df.get('GrossProfit', 0) / df.get('Revenues', 1)
            
            # Operating Margin
            df['operating_margin'] = df.get('OperatingIncomeLoss', 0) / df.get('Revenues', 1)
            
            # Net Margin
            df['net_margin'] = df.get('NetIncomeLoss', 0) / df.get('Revenues', 1)
            
            # Working Capital
            df['working_capital'] = df.get('AssetsCurrent', 0) - df.get('LiabilitiesCurrent', 0)
            
            # Interest Coverage
            df['interest_coverage'] = df.get('OperatingIncomeLoss', 0) / df.get('InterestExpense', 1)
            
            # Inventory Turnover
            df['inventory_turnover'] = df.get('CostOfRevenue', 0) / df.get('InventoryNet', 1)
            
            # Receivables Turnover
            df['receivables_turnover'] = df.get('Revenues', 0) / df.get('AccountsReceivableNetCurrent', 1)
            
            # Operating Cash Flow Ratio
            df['operating_cash_flow_ratio'] = df.get('NetCashProvidedByUsedInOperatingActivities', 0) / df.get('LiabilitiesCurrent', 1)
            
            # Free Cash Flow
            df['free_cash_flow'] = df.get('NetCashProvidedByUsedInOperatingActivities', 0) - df.get('PaymentsToAcquirePropertyPlantAndEquipment', 0)
            
            # Financial Leverage
            df['financial_leverage'] = df.get('Assets', 0) / df.get('StockholdersEquity', 1)
            
            # Equity Multiplier
            df['equity_multiplier'] = df.get('Assets', 0) / df.get('StockholdersEquity', 1)
            
            # Times Interest Earned
            df['times_interest_earned'] = df.get('OperatingIncomeLoss', 0) / df.get('InterestExpense', 1)
            
            # Cash Coverage Ratio
            df['cash_coverage_ratio'] = (df.get('OperatingIncomeLoss', 0) + df.get('AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment', 0)) / df.get('InterestExpense', 1)
            
            # Operating Cash Flow to Sales
            df['operating_cash_flow_to_sales'] = df.get('NetCashProvidedByUsedInOperatingActivities', 0) / df.get('Revenues', 1)
            
            # CapEx to Sales
            df['capex_to_sales'] = df.get('PaymentsToAcquirePropertyPlantAndEquipment', 0) / df.get('Revenues', 1)
            
            # Fill remaining features with zeros for features not calculated
            for feature in self.feature_columns:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Replace infinite values with NaN, then fill with 0
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            print(f"Feature engineering completed. Generated {len(df.columns)} features.")
            return df
            
        except Exception as e:
            print(f"Error in basic feature engineering: {e}")
            # Return a DataFrame with all features set to 0
            result_df = pd.DataFrame(index=raw_df.index)
            for feature in self.feature_columns:
                result_df[feature] = 0
            return result_df
     
    def _load_raw_feature_columns(self):
        """Load raw fundamental data column names for feature engineering"""
        self.raw_feature_columns = [
            'AccountsPayableCurrent', 'AccountsReceivableNetCurrent', 
            'AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment',
            'AccumulatedOtherComprehensiveIncomeLossNetOfTax', 'AdditionalPaidInCapitalCommonStock',
            'AllocatedShareBasedCompensationExpense', 'Assets', 'AssetsCurrent', 
            'CashAndCashEquivalentsAtCarryingValue', 'CommonStockValue', 'ComprehensiveIncomeNetOfTax',
            'CostOfRevenue', 'EmployeeRelatedLiabilitiesCurrent', 'GrossProfit', 
            'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
            'IncomeTaxExpenseBenefit', 'IncreaseDecreaseInAccountsPayableAndAccruedLiabilities',
            'IncreaseDecreaseInAccountsReceivable', 'IncreaseDecreaseInOtherNoncurrentLiabilities',
            'IncreaseDecreaseInPrepaidDeferredExpenseAndOtherAssets', 'InterestCostsCapitalized',
            'InterestExpense', 'InventoryNet', 'InventoryWriteDown', 'InvestmentIncomeInterest',
            'Liabilities', 'LiabilitiesAndStockholdersEquity', 'LiabilitiesCurrent',
            'NetCashProvidedByUsedInFinancingActivities', 'NetCashProvidedByUsedInInvestingActivities',
            'NetCashProvidedByUsedInOperatingActivities', 'NetIncomeLoss',
            'NoncashOrPartNoncashAcquisitionValueOfAssetsAcquired1', 'NoncurrentAssets',
            'OperatingExpenses', 'OperatingIncomeLoss', 'OtherAssetsNoncurrent',
            'OtherComprehensiveIncomeLossForeignCurrencyTransactionAndTranslationAdjustmentNetOfTax',
            'OtherLiabilitiesNoncurrent', 'OtherNonoperatingIncomeExpense',
            'PaymentsToAcquirePropertyPlantAndEquipment', 'PrepaidExpenseAndOtherAssetsCurrent',
            'ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlansIncludingStockOptions',
            'ProductWarrantyAccrualPreexistingIncreaseDecrease', 'PropertyPlantAndEquipmentGross',
            'PropertyPlantAndEquipmentNet', 'ResearchAndDevelopmentExpense', 'RetainedEarningsAccumulatedDeficit',
            'Revenues', 'SellingGeneralAndAdministrativeExpense', 'ShareBasedCompensation',
            'StandardProductWarrantyAccrual', 'StandardProductWarrantyAccrualPayments',
            'StandardProductWarrantyAccrualWarrantiesIssued', 'StockholdersEquity', 'TaxesPayableCurrent',
            'UnrecognizedTaxBenefits'
        ]
        print(f"Raw feature columns loaded: {len(self.raw_feature_columns)} fundamental features")
    
    def _load_technical_columns(self):
        """Load technical data columns for technical analysis"""
        try:
            # Define technical data columns based on TSLA_original.csv structure
            self.technical_columns = [
                'time', 'open', 'high', 'low', 'close', 'Volume', 'Histogram', 
                'MACD', 'Signal', 'K', 'D', 'Turnover (Cr)', '10 MA Turnover', 
                'Turnover / 10MA (X)'
            ]
            print(f"Loaded {len(self.technical_columns)} technical feature columns")
        except Exception as e:
            print(f"Error loading technical columns: {str(e)}")
    

    
    def preprocess_raw_data(self, raw_data, ticker="TSLA"):
        """
        Preprocess raw fundamental data using feature engineering
        
        Args:
            raw_data (dict or pd.DataFrame): Raw fundamental data
            ticker (str): Stock ticker symbol
            
        Returns:
            pd.DataFrame: Processed features ready for model prediction
        """
        try:
            # Convert raw_data to DataFrame if it's a dictionary
            if isinstance(raw_data, dict):
                # Create a DataFrame with quarterly index (simulating time series)
                raw_df = pd.DataFrame([raw_data])
                raw_df.index = ['2024-Q1']  # Default quarter
            elif isinstance(raw_data, list):
                if len(raw_data) != len(self.raw_feature_columns):
                    raise ValueError(f"Raw data length ({len(raw_data)}) doesn't match expected raw features ({len(self.raw_feature_columns)})")
                raw_df = pd.DataFrame([raw_data], columns=self.raw_feature_columns)
                raw_df.index = ['2024-Q1']  # Default quarter
            elif isinstance(raw_data, pd.DataFrame):
                raw_df = raw_data.copy()
            else:
                raise ValueError("Raw data must be a dict, list, or pandas DataFrame")
            
            # Ensure all required raw columns are present
            missing_cols = set(self.raw_feature_columns) - set(raw_df.columns)
            if missing_cols:
                print(f"Warning: Missing raw columns will be filled with 0: {missing_cols}")
                for col in missing_cols:
                    raw_df[col] = 0
            
            # Apply feature engineering
            print("Applying feature engineering preprocessing...")
            processed_df = self._apply_basic_feature_engineering(raw_df, ticker)
            
            # Ensure all model features are present
            missing_model_cols = set(self.feature_columns) - set(processed_df.columns)
            if missing_model_cols:
                print(f"Warning: Missing model features will be filled with 0: {len(missing_model_cols)} features")
                for col in missing_model_cols:
                    processed_df[col] = 0
            
            # Select only the features used by the model
            processed_df = processed_df[self.feature_columns]
            
            return processed_df
            
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            raise
    
    def predict(self, raw_fundamental_data=None, technical_data=None, ticker="TSLA"):
        """
        Make predictions using raw fundamental data and/or technical data
        
        Args:
            raw_fundamental_data (dict or pd.DataFrame): Raw fundamental data
            technical_data (dict or pd.DataFrame): Technical data
            ticker (str): Stock ticker symbol
            
        Returns:
            dict: Dictionary containing predictions and probabilities
        """
        try:
            processed_df = pd.DataFrame()
            
            # Process fundamental data if provided
            if raw_fundamental_data is not None:
                print("Processing raw fundamental data...")
                fundamental_df = self.preprocess_raw_data(raw_fundamental_data, ticker)
                
                if isinstance(fundamental_df, pd.DataFrame) and not fundamental_df.empty:
                    if processed_df.empty:
                        processed_df = fundamental_df.copy()
                    else:
                        processed_df = pd.concat([processed_df, fundamental_df], axis=1)
                else:
                    print("Warning: fundamental_df is not a valid DataFrame or is empty")
            
            # Process technical data if provided
            if technical_data is not None:
                print("Processing technical data...")
                technical_df = self._preprocess_technical_data(technical_data)
                
                if isinstance(technical_df, pd.DataFrame) and not technical_df.empty:
                    if processed_df.empty:
                        processed_df = technical_df.copy()
                    else:
                        # Ensure both DataFrames have the same index and reset to avoid duplication
                        processed_df = processed_df.reset_index(drop=True)
                        technical_df = technical_df.reset_index(drop=True)
                        
                        # Check for overlapping columns and handle them
                        overlapping_cols = set(processed_df.columns) & set(technical_df.columns)
                        if overlapping_cols:
                            print(f"Warning: Overlapping columns found: {overlapping_cols}. Keeping values from fundamental data.")
                            technical_df = technical_df.drop(columns=list(overlapping_cols))
                        
                        processed_df = pd.concat([processed_df, technical_df], axis=1)
                else:
                    print("Warning: technical_df is not a valid DataFrame or is empty")
            
            if processed_df.empty or not isinstance(processed_df, pd.DataFrame):
                raise ValueError("Failed to create valid processed DataFrame. Either raw_fundamental_data or technical_data must be provided and valid.")
            
            # Ensure all required features are present
            missing_model_cols = set(self.feature_columns) - set(processed_df.columns)
            if missing_model_cols:
                print(f"Warning: Missing model features will be filled with 0: {len(missing_model_cols)} features")
                for col in missing_model_cols:
                    processed_df[col] = 0
            
            # Select only the features used by the model
            processed_df = processed_df[self.feature_columns]
            
            # Validate processed_df before prediction
            if not isinstance(processed_df, pd.DataFrame):
                raise ValueError(f"processed_df is not a DataFrame, it's {type(processed_df)}")
            
            # Make predictions
            predictions_mapped = self.model.predict(processed_df)
            probabilities = self.model.predict_proba(processed_df)
            
            # Convert predictions back to original labels
            predictions_original = [self.reverse_label_mapping[pred] for pred in predictions_mapped]
            
            # Prepare results
            results = {
                'predicted_label': predictions_original[0] if len(predictions_original) == 1 else predictions_original,
                'predicted_label_mapped': predictions_mapped[0] if len(predictions_mapped) == 1 else predictions_mapped,
                'probabilities': probabilities[0] if len(probabilities) == 1 else probabilities,
                'class_probabilities': {}
            }
            
            # Add class-specific probabilities
            if len(probabilities) == 1:
                for mapped_label, original_label in self.reverse_label_mapping.items():
                    results['class_probabilities'][f'class_{original_label}'] = probabilities[0][mapped_label]
            
            return results
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            raise
    
    def _preprocess_technical_data(self, technical_data):
        """
        Preprocess technical data for model prediction
        
        Args:
            technical_data (dict or pd.DataFrame): Technical data
            
        Returns:
            pd.DataFrame: Processed technical features
        """
        try:
            # Convert technical_data to DataFrame if it's a dictionary
            if isinstance(technical_data, dict):
                tech_df = pd.DataFrame([technical_data])
            elif isinstance(technical_data, pd.DataFrame):
                tech_df = technical_data.copy()
            else:
                raise ValueError("Technical data must be a dict or pandas DataFrame")
            
            # Ensure all required technical columns are present
            missing_tech_cols = set(self.technical_columns) - set(tech_df.columns)
            if missing_tech_cols:
                print(f"Warning: Missing technical columns will be filled with 0: {missing_tech_cols}")
                for col in missing_tech_cols:
                    tech_df[col] = 0
            
            # Select only the technical columns we need
            tech_df = tech_df[self.technical_columns]
            
            return tech_df
            
        except Exception as e:
            print(f"Error during technical data preprocessing: {e}")
            raise
    
    def get_feature_importance(self, top_n=20):
        """
        Get feature importance from the trained model
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        try:
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df.head(top_n)
            
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return None

# Sample raw fundamental data
raw_fundamental_data = {
    'AccountsPayableCurrent': 303969000.0,
    'AccountsReceivableNetCurrent': 49109000.0,
    'AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment': 140142000.0,
    'AccumulatedOtherComprehensiveIncomeLossNetOfTax': -3000.0,
    'AdditionalPaidInCapitalCommonStock': 1806617000.0,
    'AllocatedShareBasedCompensationExpense': 55566000.0,
    'Assets': 2416930000.0,
    'AssetsCurrent': 1265939000.0,
    'CashAndCashEquivalentsAtCarryingValue': 201890000.0,
    'CommonStockValue': 123000.0,
    'ComprehensiveIncomeNetOfTax': -254414000.0,
    'CostOfRevenue': 1098604000.0,
    'EmployeeRelatedLiabilitiesCurrent': 26535000.0,
    'GrossProfit': 299673000.0,
    'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest': -56520000.0,
    'IncomeTaxExpenseBenefit': 1230000.0,
    'IncreaseDecreaseInAccountsPayableAndAccruedLiabilities': 5862000.0,
    'IncreaseDecreaseInAccountsReceivable': 20716000.0,
    'IncreaseDecreaseInOtherNoncurrentLiabilities': 28669000.0,
    'IncreaseDecreaseInPrepaidDeferredExpenseAndOtherAssets': 9115000.0,
    'InterestCostsCapitalized': 2300000.0,
    'InterestExpense': 26705000.0,
    'InventoryNet': 340355000.0,
    'InventoryWriteDown': 6788000.0,
    'InvestmentIncomeInterest': 97000.0,
    'Liabilities': 1749810000.0,
    'LiabilitiesAndStockholdersEquity': 2416930000.0,
    'LiabilitiesCurrent': 675160000.0,
    'NetCashProvidedByUsedInFinancingActivities': 3684000.0,
    'NetCashProvidedByUsedInInvestingActivities': -55236000.0,
    'NetCashProvidedByUsedInOperatingActivities': 64079000.0,
    'NetIncomeLoss': -57750000.0,
    'NoncashOrPartNoncashAcquisitionValueOfAssetsAcquired1': 24708000.0,
    'NoncurrentAssets': 1120919000.0,
    'OperatingExpenses': 347603000.0,
    'OperatingIncomeLoss': -47930000.0,
    'OtherAssetsNoncurrent': 23637000.0,
    'OtherComprehensiveIncomeLossForeignCurrencyTransactionAndTranslationAdjustmentNetOfTax': -16147000.0,
    'OtherLiabilitiesNoncurrent': 58197000.0,
    'OtherNonoperatingIncomeExpense': 18018000.0,
    'PaymentsToAcquirePropertyPlantAndEquipment': 174790000.0,
    'PrepaidExpenseAndOtherAssetsCurrent': 27574000.0,
    'ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlansIncludingStockOptions': 82219000.0,
    'ProductWarrantyAccrualPreexistingIncreaseDecrease': 8052000.0,
    'PropertyPlantAndEquipmentGross': 878636000.0,
    'PropertyPlantAndEquipmentNet': 738494000.0,
    'ResearchAndDevelopmentExpense': 163523000.0,
    'RetainedEarningsAccumulatedDeficit': -1139620000.0,
    'Revenues': 1398277000.0,
    'SellingGeneralAndAdministrativeExpense': 184080000.0,
    'ShareBasedCompensation': 55566000.0,
    'StandardProductWarrantyAccrual': 13012000.0,
    'StandardProductWarrantyAccrualPayments': 11100000.0,
    'StandardProductWarrantyAccrualWarrantiesIssued': 43758000.0,
    'StockholdersEquity': 667120000.0,
    'TaxesPayableCurrent': 38067000.0,
    'UnrecognizedTaxBenefits': 13400000.0
}

# Sample technical data
technical_data = {
    'entry_price': 350.00,
    'upper_barrier': 370.00,
    'lower_barrier': 330.00,
    'open': 352.26,
    'high': 357.00,
    'low': 351.28,
    'close': 354.58,
    'Volume': 15693400,
    'Histogram': 2.45,
    'MACD': 8.32,
    'Signal': 5.87,
    'K': 65.23,
    'D': 58.91,
    'Turnover (Cr)': 55.67,
    '10 MA Turnover': 48.32,
    'Turnover / 10MA (X)': 1.15
}

if __name__ == "__main__":
    # Example usage with different data combinations
    
    # Example 1: Using only fundamental data
    print("\n=== Example 1: Fundamental Data Only ===")
    try:
        inference = XGBoostInference()
        results = inference.predict(raw_fundamental_data=raw_fundamental_data, ticker="TSLA")
        
        print(f"Predicted Label: {results['predicted_label']}")
        print(f"Predicted Label (Mapped): {results['predicted_label_mapped']}")
        print(f"\nClass Probabilities:")
        for class_name, prob in results['class_probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
        
    except Exception as e:
        print(f"Error in fundamental prediction: {e}")
    
    # Example 2: Using only technical data
    print("\n=== Example 2: Technical Data Only ===")
    try:
        inference = XGBoostInference()
        results = inference.predict(technical_data=technical_data, ticker="TSLA")
        
        print(f"Predicted Label: {results['predicted_label']}")
        print(f"Predicted Label (Mapped): {results['predicted_label_mapped']}")
        print(f"\nClass Probabilities:")
        for class_name, prob in results['class_probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
        
    except Exception as e:
        print(f"Error in technical prediction: {e}")
    
    # Example 3: Using both fundamental and technical data
    print("\n=== Example 3: Combined Fundamental + Technical Data ===")
    try:
        inference = XGBoostInference()
        results = inference.predict(
            raw_fundamental_data=raw_fundamental_data, 
            technical_data=technical_data, 
            ticker="TSLA"
        )
        
        print(f"Predicted Label: {results['predicted_label']}")
        print(f"Predicted Label (Mapped): {results['predicted_label_mapped']}")
        print(f"\nClass Probabilities:")
        for class_name, prob in results['class_probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
        
        # Get feature importance
        print("\n=== Top 10 Feature Importance ===")
        importance_df = inference.get_feature_importance(top_n=10)
        if importance_df is not None:
            print(importance_df.to_string(index=False))
            
    except Exception as e:
        print(f"Error in combined prediction: {e}")
    
    # Label interpretation
    print("\n=== Label Interpretation ===")
    print("Label -1: Negative/Bearish signal (price expected to go down)")
    print("Label  0: Neutral signal (price expected to stay relatively stable)")
    print("Label  1: Positive/Bullish signal (price expected to go up)")
    
    print("\nNote: This model can process:")
    print("- Raw fundamental data through feature engineering")
    print("- Technical indicators and price data")
    print("- Combined fundamental + technical data for enhanced predictions")
    print("Please ensure the model files exist and are accessible.")