# XGBoost Model Inference Script
import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
import sys
import os
import logging

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
sys.path.append('/root/vynixmodelling/ML_RL')

def current_ratio(df):
    return df['AssetsCurrent'] / df['LiabilitiesCurrent']
def quick_ratio(df):
    return (df['AssetsCurrent'] - df.get('InventoryNet', 0)) / df['LiabilitiesCurrent']
def cash_ratio(df):
    return df['CashAndCashEquivalentsAtCarryingValue'] / df['LiabilitiesCurrent']
def working_capital(df):
    return df['AssetsCurrent'] - df['LiabilitiesCurrent']
def accounts_receivable_turnover(df):
    return df['Revenues'] / df['AccountsReceivableNetCurrent']
def days_sales_outstanding(df):
    return (df['AccountsReceivableNetCurrent'] / df['Revenues']) * 365
def inventory_turnover(df):
    return df['CostOfRevenue'] / df.get('InventoryNet', 1)
def days_inventory_outstanding(df):
    return (df.get('InventoryNet', 0) / df['CostOfRevenue']) * 365
def accounts_payable_turnover(df):
    return df['CostOfRevenue'] / df['AccountsPayableCurrent']
def days_payable_outstanding(df):
    return (df['AccountsPayableCurrent'] / df['CostOfRevenue']) * 365
def cash_conversion_cycle(df):
    dso = days_sales_outstanding(df)
    dio = days_inventory_outstanding(df)
    dpo = days_payable_outstanding(df)
    return dso + dio - dpo
def gross_profit_margin(df):
    return (df.get('GrossProfit', df['Revenues'] - df['CostOfRevenue']) / df['Revenues']) * 100
def operating_profit_margin(df):
    return (df['OperatingIncomeLoss'] / df['Revenues']) * 100
def net_profit_margin(df):
    return (df['NetIncomeLoss'] / df['Revenues']) * 100
def return_on_assets(df):
    return (df['NetIncomeLoss'] / df['Assets']) * 100
def return_on_equity(df):
    return (df['NetIncomeLoss'] / df['StockholdersEquity']) * 100
def debt_to_equity_ratio(df):
    return df['Liabilities'] / df['StockholdersEquity']
def debt_to_assets_ratio(df):
    return df['Liabilities'] / df['Assets']
def interest_coverage_ratio(df):
    return df['OperatingIncomeLoss'] / df.get('InvestmentIncomeInterest', 1)
def operating_expense_ratio(df):
    return (df.get('OperatingExpenses', 0) / df['Revenues']) * 100
def rd_to_revenue_ratio(df):
    return (df.get('ResearchAndDevelopmentExpense', 0) / df['Revenues']) * 100
def sga_to_revenue_ratio(df):
    return (df.get('SellingGeneralAndAdministrativeExpense', 0) / df['Revenues']) * 100
def fixed_asset_turnover(df):
    return df['Revenues'] / df['PropertyPlantAndEquipmentNet']
def total_asset_turnover(df):
    return df['Revenues'] / df['Assets']
def capital_expenditure_ratio(df):
    return df.get('PaymentsToAcquirePropertyPlantAndEquipment', 0) / df['NetIncomeLoss']
def compensation_efficiency(df):
    return (df.get('AllocatedShareBasedCompensationExpense', 0) / df['Revenues']) * 100
def warranty_reserve_ratio(df):
    return (df.get('StandardProductWarrantyAccrual', 0) / df['Revenues']) * 100
def depreciation_rate(df):
    return (df.get('AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment', 0) / df.get('PropertyPlantAndEquipmentGross', 1)) * 100
def operating_cash_flow_to_net_income_ratio(df):
    cash_flow = df.get('IncreaseDecreaseInAccountsReceivable', 0) + df.get('IncreaseDecreaseInPrepaidDeferredExpenseAndOtherAssets', 0)
    return cash_flow / df['NetIncomeLoss']
def effective_tax_rate(df):
    return (df.get('IncomeTaxExpenseBenefit', 0) / df.get('IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest', 1)) * 100
def revenue_growth_rate(df):
    return df['Revenues'].pct_change() * 100
def net_income_growth_rate(df):
    return df['NetIncomeLoss'].pct_change() * 100
def asset_growth_rate(df):
    return df['Assets'].pct_change() * 100
def accrual_ratio(df):
    cash_flow = df.get('IncreaseDecreaseInAccountsReceivable', 0) + df.get('IncreaseDecreaseInPrepaidDeferredExpenseAndOtherAssets', 0)
    return (df['NetIncomeLoss'] - cash_flow) / df['Assets']
def cash_flow_to_revenue_ratio(df):
    cash_flow = df.get('IncreaseDecreaseInAccountsReceivable', 0) + df.get('IncreaseDecreaseInPrepaidDeferredExpenseAndOtherAssets', 0)
    return cash_flow / df['Revenues']
def qoq_growth(df, column):
    return df[column].pct_change() * 100
def yoy_quarterly_growth(df, column):
    return df[column].pct_change(periods=4) * 100
def trailing_twelve_months(df, column):
    return df[column].rolling(window=4).sum()
def quarterly_acceleration(df, column):
    qoq = qoq_growth(df, column)
    return qoq.diff()
def seasonal_index(df, column):
    avg_4q = df[column].rolling(window=4).mean()
    return df[column] / avg_4q
def seasonal_growth_rate(df, column):
    return yoy_quarterly_growth(df, column)
def moving_average_4q(df, column):
    return df[column].rolling(window=4).mean()
def quarter_run_rate(df, column):
    return df[column] * 4
def quarterly_operating_leverage(df):
    revenue_growth = qoq_growth(df, 'Revenues')
    operating_growth = qoq_growth(df, 'OperatingIncomeLoss')
    return operating_growth / revenue_growth
def quarterly_cash_burn_rate(df):
    cash_change = df['CashAndCashEquivalentsAtCarryingValue'].diff()
    return -cash_change / 3
def ytd_performance(df, column):
    return df[column].expanding().sum()
def quarterly_volatility(df, column):
    qoq = qoq_growth(df, column)
    return qoq.rolling(window=4).std()
def seasonal_dependency_index(df, column):
    quarterly_avg = df[column].rolling(window=4).mean()
    max_var = df[column].rolling(window=4).max() - df[column].rolling(window=4).min()
    return (max_var / quarterly_avg) * 100
def return_on_invested_capital(df):
    tax_rate = 0.25  # Assumed tax rate
    nopat = df['OperatingIncomeLoss'] * (1 - tax_rate)
    invested_capital = df['Assets'] - df['LiabilitiesCurrent']
    return (nopat / invested_capital) * 100
def cash_return_on_capital_invested(df):
    cash_flow = df.get('IncreaseDecreaseInAccountsReceivable', 0) + df.get('IncreaseDecreaseInPrepaidDeferredExpenseAndOtherAssets', 0)
    invested_capital = df['Assets'] - df['LiabilitiesCurrent']
    return cash_flow / invested_capital
def fixed_assets_to_long_term_debt_ratio(df):
    long_term_debt = df['Liabilities'] - df['LiabilitiesCurrent']
    return df['PropertyPlantAndEquipmentNet'] / long_term_debt
def non_current_asset_turnover(df):
    non_current_assets = df['Assets'] - df['AssetsCurrent']
    return df['Revenues'] / non_current_assets
def quarterly_gross_profit_stability(df):
    gross_margin = gross_profit_margin(df)
    return gross_margin.rolling(window=4).std()
def revenue_to_expense_growth_differential(df):
    revenue_growth = qoq_growth(df, 'Revenues')
    expense_growth = qoq_growth(df, 'OperatingExpenses') if 'OperatingExpenses' in df.columns else 0
    return revenue_growth - expense_growth
def quarterly_cash_flow_quality(df):
    cash_flow = df.get('IncreaseDecreaseInAccountsReceivable', 0) + df.get('IncreaseDecreaseInPrepaidDeferredExpenseAndOtherAssets', 0)
    quality_ratio = cash_flow / df['NetIncomeLoss']
    return quality_ratio.rolling(window=4).std()
def dividend_payout_ratio(df):
    dividends = df.get('Dividends_Declared', 0)
    return (dividends / df['NetIncomeLoss']) * 100
def stock_based_compensation_to_operating_expense_ratio(df):
    sbc = df.get('ShareBasedCompensation', 0)
    operating_expenses = df.get('OperatingExpenses', 1)
    return (sbc / operating_expenses) * 100
def long_term_revenue_cagr(df, periods=12):
    if len(df) < periods:
        return np.nan
    current_revenue = df['Revenues'].iloc[-1]
    past_revenue = df['Revenues'].iloc[-periods]
    return ((current_revenue / past_revenue) ** (1/periods) - 1) * 100
def operating_margin_trend(df):
    operating_margin = operating_profit_margin(df)
    x = np.arange(len(operating_margin))
    slope, _, _, _, _ = stats.linregress(x, operating_margin)
    return slope
def financial_leverage_index(df):
    leverage_ratio = debt_to_assets_ratio(df)
    return leverage_ratio / leverage_ratio.shift(1)
def asset_coverage_ratio(df):
    tangible_assets = df['Assets'] - df['LiabilitiesCurrent']
    long_term_debt = df['Liabilities'] - df['LiabilitiesCurrent']
    return tangible_assets / long_term_debt
def quarterly_margin_expansion(df):
    net_margin = net_profit_margin(df)
    return net_margin.diff()
def asset_utilization_ratio(df):
    avg_assets = (df['Assets'] + df['Assets'].shift(1)) / 2
    return df['Revenues'] / avg_assets
def capacity_utilization_proxy(df):
    industry_capacity_factor = 0.8  # Assumed
    return df['CostOfRevenue'] / (df['PropertyPlantAndEquipmentNet'] * industry_capacity_factor)
def altman_z_score(df):
    working_capital = df['working_capital']
    retained_earnings = df.get('RetainedEarningsAccumulatedDeficit', 0)
    z_score = (1.2 * (working_capital / df['Assets']) + 
               1.4 * (retained_earnings / df['Assets']) + 
               3.3 * (df['OperatingIncomeLoss'] / df['Assets']) + 
               0.6 * (df['StockholdersEquity'] / df['Liabilities']) + 
               0.999 * (df['Revenues'] / df['Assets']))
    return z_score
def dupont_analysis_roe(df):
    net_margin = net_profit_margin(df) / 100
    asset_turnover = total_asset_turnover(df)
    equity_multiplier = df['Assets'] / df['StockholdersEquity']
    return net_margin * asset_turnover * equity_multiplier * 100
def economic_value_added(df):
    wacc = 0.10  # Assumed weighted average cost of capital
    tax_rate = 0.25  # Assumed tax rate
    nopat = df['OperatingIncomeLoss'] * (1 - tax_rate)
    invested_capital = df['Assets'] - df['LiabilitiesCurrent']
    return nopat - (wacc * invested_capital)
def rd_efficiency_ratio(df):
    gross_profit = df.get('GrossProfit', df['Revenues'] - df['CostOfRevenue'])
    rd_expense = df.get('ResearchAndDevelopmentExpense', 1)
    return gross_profit / rd_expense
def innovation_investment_ratio(df):
    rd_expense = df.get('ResearchAndDevelopmentExpense', 0)
    capex = df.get('PaymentsToAcquirePropertyPlantAndEquipment', 0)
    return ((rd_expense + capex) / df['Revenues']) * 100
def revenue_momentum(df):
    if len(df) < 6:
        return np.nan
    q2_q1_change = df['Revenues'].iloc[-1] - df['Revenues'].iloc[-2]
    q1_q4_change = df['Revenues'].iloc[-2] - df['Revenues'].iloc[-5]
    return q2_q1_change / q1_q4_change if q1_q4_change != 0 else np.nan
def earnings_momentum(df):
    if len(df) < 6:
        return np.nan
    q2_q1_change = df['NetIncomeLoss'].iloc[-1] - df['NetIncomeLoss'].iloc[-2]
    q1_q4_change = df['NetIncomeLoss'].iloc[-2] - df['NetIncomeLoss'].iloc[-5]
    return q2_q1_change / q1_q4_change if q1_q4_change != 0 else np.nan
def quarterly_earnings_quality_index(df):
    operating_net_diff = (df['OperatingIncomeLoss'] - df['NetIncomeLoss']) / df['Revenues']
    return operating_net_diff / operating_net_diff.shift(1)
def non_operating_items_ratio(df):
    non_operating = df['NetIncomeLoss'] - df['OperatingIncomeLoss']
    return (non_operating / df['NetIncomeLoss']) * 100

# Daftar fungsi yang akan diterapkan
feature_functions = [
    ('current_ratio', current_ratio),
    ('quick_ratio', quick_ratio),
    ('cash_ratio', cash_ratio),
    ('working_capital', working_capital),
    ('accounts_receivable_turnover', accounts_receivable_turnover),
    ('days_sales_outstanding', days_sales_outstanding),
    ('inventory_turnover', inventory_turnover),
    ('days_inventory_outstanding', days_inventory_outstanding),
    ('accounts_payable_turnover', accounts_payable_turnover),
    ('days_payable_outstanding', days_payable_outstanding),
    ('cash_conversion_cycle', cash_conversion_cycle),
    ('gross_profit_margin', gross_profit_margin),
    ('operating_profit_margin', operating_profit_margin),
    ('net_profit_margin', net_profit_margin),
    ('return_on_assets', return_on_assets),
    ('return_on_equity', return_on_equity),
    ('debt_to_equity_ratio', debt_to_equity_ratio),
    ('debt_to_assets_ratio', debt_to_assets_ratio),
    ('interest_coverage_ratio', interest_coverage_ratio),
    ('operating_expense_ratio', operating_expense_ratio),
    ('rd_to_revenue_ratio', rd_to_revenue_ratio),
    ('sga_to_revenue_ratio', sga_to_revenue_ratio),
    ('fixed_asset_turnover', fixed_asset_turnover),
    ('total_asset_turnover', total_asset_turnover),
    ('capital_expenditure_ratio', capital_expenditure_ratio),
    ('compensation_efficiency', compensation_efficiency),
    ('warranty_reserve_ratio', warranty_reserve_ratio),
    ('depreciation_rate', depreciation_rate),
    ('operating_cash_flow_to_net_income_ratio', operating_cash_flow_to_net_income_ratio),
    ('effective_tax_rate', effective_tax_rate),
    ('revenue_growth_rate', revenue_growth_rate),
    ('net_income_growth_rate', net_income_growth_rate),
    ('asset_growth_rate', asset_growth_rate),
    ('accrual_ratio', accrual_ratio),
    ('cash_flow_to_revenue_ratio', cash_flow_to_revenue_ratio),
    ('return_on_invested_capital', return_on_invested_capital),
    ('cash_return_on_capital_invested', cash_return_on_capital_invested),
    ('fixed_assets_to_long_term_debt_ratio', fixed_assets_to_long_term_debt_ratio),
    ('non_current_asset_turnover', non_current_asset_turnover),
    ('quarterly_gross_profit_stability', quarterly_gross_profit_stability),
    ('revenue_to_expense_growth_differential', revenue_to_expense_growth_differential),
    ('quarterly_cash_flow_quality', quarterly_cash_flow_quality),
    ('dividend_payout_ratio', dividend_payout_ratio),
    ('stock_based_compensation_to_operating_expense_ratio', stock_based_compensation_to_operating_expense_ratio),
    ('financial_leverage_index', financial_leverage_index),
    ('asset_coverage_ratio', asset_coverage_ratio),
    ('quarterly_margin_expansion', quarterly_margin_expansion),
    ('asset_utilization_ratio', asset_utilization_ratio),
    ('capacity_utilization_proxy', capacity_utilization_proxy),
    ('altman_z_score', altman_z_score),
    ('dupont_analysis_roe', dupont_analysis_roe),
    ('economic_value_added', economic_value_added),
    ('rd_efficiency_ratio', rd_efficiency_ratio),
    ('innovation_investment_ratio', innovation_investment_ratio),
    ('revenue_momentum', revenue_momentum),
    ('earnings_momentum', earnings_momentum),
    ('quarterly_earnings_quality_index', quarterly_earnings_quality_index),
    ('non_operating_items_ratio', non_operating_items_ratio)
]

# Fungsi yang memerlukan parameter kolom
column_based_functions = [
    ('revenues_qoq_growth', 'Revenues', qoq_growth),
    ('revenues_yoy_growth', 'Revenues', yoy_quarterly_growth),
    ('revenues_ttm', 'Revenues', trailing_twelve_months),
    ('revenues_acceleration', 'Revenues', quarterly_acceleration),
    ('revenues_seasonal_index', 'Revenues', seasonal_index),
    ('revenues_moving_avg', 'Revenues', moving_average_4q),
    ('revenues_run_rate', 'Revenues', quarter_run_rate),
    ('revenues_volatility', 'Revenues', quarterly_volatility),
    ('revenues_seasonal_dependency', 'Revenues', seasonal_dependency_index),
    ('net_income_qoq_growth', 'NetIncomeLoss', qoq_growth),
    ('net_income_yoy_growth', 'NetIncomeLoss', yoy_quarterly_growth),
    ('net_income_ttm', 'NetIncomeLoss', trailing_twelve_months),
    ('net_income_acceleration', 'NetIncomeLoss', quarterly_acceleration),
    ('net_income_seasonal_index', 'NetIncomeLoss', seasonal_index),
    ('net_income_moving_avg', 'NetIncomeLoss', moving_average_4q),
    ('net_income_run_rate', 'NetIncomeLoss', quarter_run_rate),
    ('net_income_volatility', 'NetIncomeLoss', quarterly_volatility),
    ('assets_qoq_growth', 'Assets', qoq_growth),
    ('assets_yoy_growth', 'Assets', yoy_quarterly_growth),
    ('assets_ttm', 'Assets', trailing_twelve_months),
    ('operating_income_qoq_growth', 'OperatingIncomeLoss', qoq_growth),
    ('operating_income_yoy_growth', 'OperatingIncomeLoss', yoy_quarterly_growth),
    ('operating_income_ttm', 'OperatingIncomeLoss', trailing_twelve_months),
    ('cash_qoq_growth', 'CashAndCashEquivalentsAtCarryingValue', qoq_growth),
    ('cash_yoy_growth', 'CashAndCashEquivalentsAtCarryingValue', yoy_quarterly_growth),
    ('equity_qoq_growth', 'StockholdersEquity', qoq_growth),
    ('equity_yoy_growth', 'StockholdersEquity', yoy_quarterly_growth),
    ('liabilities_qoq_growth', 'Liabilities', qoq_growth),
    ('liabilities_yoy_growth', 'Liabilities', yoy_quarterly_growth)
]

class XGBoostInference:
    def __init__(self, model_path=None, label_mapping_path=None):
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
        
        # Load model and mappings
        self._load_model()
        self._load_label_mappings()
        self._load_feature_columns()
    
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
        # Load feature columns from training data
        train_data_path = '/root/vynixmodelling/dataset/training_data/X_train.csv'
        if Path(train_data_path).exists():
            sample_df = pd.read_csv(train_data_path, nrows=1)
            self.feature_columns = sample_df.columns.tolist()
            print(f"Feature columns loaded: {len(self.feature_columns)} features")
        else:
            raise FileNotFoundError(f"Training data not found at {train_data_path}")

    
    def _apply_basic_feature_engineering(self, raw_df, ticker="TSLA"):
        try:
            # Create a copy of the raw data
            enhanced_df = raw_df.copy()
            
            # Apply all feature engineering functions
            for name, func in feature_functions:
                try:
                    result = func(enhanced_df)
                    enhanced_df[name] = result
                    
                    # Debug: Check if result is NaN or 0 and why
                    if isinstance(result, pd.Series):
                        final_val = result.iloc[0] if len(result) > 0 else 0
                    else:
                        final_val = result
                    
                    if pd.isna(final_val) or final_val == 0:
                        print(f"DEBUG - {name}: result={final_val} (likely needs historical data for growth/trend calculation)")
                        
                except Exception as e:
                    enhanced_df[name] = 0
                    pass
            
            # Apply column-based functions
            for name, column, func in column_based_functions:
                try:
                    if column in enhanced_df.columns:
                        result = func(enhanced_df, column)
                        enhanced_df[name] = result
                        
                        # Debug: Check if result is NaN or 0 and why
                        if isinstance(result, pd.Series):
                            final_val = result.iloc[0] if len(result) > 0 else 0
                        else:
                            final_val = result
                        
                        if pd.isna(final_val) or final_val == 0:
                            print(f"DEBUG - {name}: result={final_val} (column-based function needs historical data)")
                    else:
                        enhanced_df[name] = 0
                except Exception as e:
                    enhanced_df[name] = 0
                    pass
            
            # Apply special functions
            try:
                enhanced_df['quarterly_operating_leverage'] = quarterly_operating_leverage(enhanced_df)
            except Exception as e:
                enhanced_df['quarterly_operating_leverage'] = 0
                pass
            
            try:
                enhanced_df['quarterly_cash_burn_rate'] = quarterly_cash_burn_rate(enhanced_df)
            except Exception as e:
                enhanced_df['quarterly_cash_burn_rate'] = 0
                pass
            
            # Apply YTD functions for key columns
            ytd_columns = ['Revenues', 'NetIncomeLoss', 'OperatingIncomeLoss']
            for col in ytd_columns:
                try:
                    if col in enhanced_df.columns:
                        enhanced_df[f'{col.lower()}_ytd'] = ytd_performance(enhanced_df, col)
                    else:
                        enhanced_df[f'{col.lower()}_ytd'] = 0
                except Exception as e:
                    enhanced_df[f'{col.lower()}_ytd'] = 0
                    pass
            
            # Apply CAGR function
            try:
                enhanced_df['long_term_revenue_cagr'] = long_term_revenue_cagr(enhanced_df)
            except Exception as e:
                enhanced_df['long_term_revenue_cagr'] = 0
                pass
            
            # Apply trend function
            try:
                enhanced_df['operating_margin_trend'] = operating_margin_trend(enhanced_df)
            except Exception as e:
                enhanced_df['operating_margin_trend'] = 0
                pass
            
            # Fill remaining features with zeros for features not calculated
            for feature in self.feature_columns:
                if feature not in enhanced_df.columns:
                    enhanced_df[feature] = 0
            
            # Replace infinite values with NaN, then fill with 0
            enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            print(f"Feature engineering completed. Generated {len(enhanced_df.columns)} features.")
            return enhanced_df
            
        except Exception as e:
            print(f"Error in feature engineering: {e}")
            # Return a DataFrame with all features set to 0
            result_df = pd.DataFrame(index=raw_df.index)
            for feature in self.feature_columns:
                result_df[feature] = 0
            return result_df

    def preprocess_raw_data(self, raw_data, ticker="TSLA"):
        try:
            # Convert raw_data to DataFrame if it's a dictionary
            if isinstance(raw_data, dict):
                # Create a DataFrame with quarterly index (simulating time series)
                raw_df = pd.DataFrame([raw_data])
                raw_df.index = ['2024-Q1']  # Default quarter
            elif isinstance(raw_data, pd.DataFrame):
                raw_df = raw_data.copy()
            else:
                raise ValueError("Raw data must be a dict or pandas DataFrame")
            
            # Apply feature engineering
            print("Applying feature engineering preprocessing...")
            processed_df = self._apply_basic_feature_engineering(raw_df, ticker)
            print("Fundamental data after preprocessing (one row at a time):")
            for index, row in processed_df.iterrows():
                print(f"Index: {index}, Data: {row.to_dict()}")
            
            # Debug: Check for zero values in processed data
            zero_features = [col for col in processed_df.columns if processed_df[col].iloc[0] == 0.0]
            non_zero_features = [col for col in processed_df.columns if processed_df[col].iloc[0] != 0.0]
            print(f"\nDEBUG - Zero features count: {len(zero_features)}")
            print(f"DEBUG - Non-zero features count: {len(non_zero_features)}")
            if zero_features:
                print(f"DEBUG - Some zero features: {zero_features[:10]}")
            if non_zero_features:
                print(f"DEBUG - Some non-zero features: {non_zero_features[:10]}")
            
            # Select only the features used by the model
            processed_df = processed_df[self.feature_columns]
            
            return processed_df
            
        except Exception as e:
            print(f"Error during preprocessing: {e}")
            raise
    
    def predict(self, raw_fundamental_data=None, technical_data=None, ticker="TSLA"):
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
                
                # Debug: Print technical data that will be used
                print(f"\nDEBUG - Technical data used as input:")
                for col, val in technical_df.iloc[0].items():
                    print(f"  {col}: {val}")
                
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
                            print(f"Warning: Overlapping columns found: {overlapping_cols}. Using technical data values for overlapping columns.")
                            # Debug: Show values being replaced
                            print(f"DEBUG - Overlapping column values:")
                            for col in list(overlapping_cols)[:5]:  # Show first 5
                                print(f"  {col}: fundamental={processed_df[col].iloc[0]}, technical={technical_df[col].iloc[0]}")
                            # Keep technical data values for overlapping columns by dropping them from processed_df first
                            processed_df = processed_df.drop(columns=list(overlapping_cols))
                        
                        processed_df = pd.concat([processed_df, technical_df], axis=1)
                        
                        # Debug: Final combined data zero analysis
                        final_zero_features = [col for col in processed_df.columns if processed_df[col].iloc[0] == 0.0]
                        final_non_zero_features = [col for col in processed_df.columns if processed_df[col].iloc[0] != 0.0]
                        print(f"\nDEBUG - Final combined data:")
                        print(f"  Zero features: {len(final_zero_features)} out of {len(processed_df.columns)}")
                        print(f"  Non-zero features: {len(final_non_zero_features)}")
                        if final_zero_features:
                            print(f"  Some zero features: {final_zero_features[:10]}")
                else:
                    print("Warning: technical_df is not a valid DataFrame or is empty")
            
            if processed_df.empty or not isinstance(processed_df, pd.DataFrame):
                raise ValueError("Failed to create valid processed DataFrame. Either raw_fundamental_data or technical_data must be provided and valid.")
            
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

        try:
            # Convert technical_data to DataFrame if it's a dictionary
            if isinstance(technical_data, dict):
                tech_df = pd.DataFrame([technical_data])
            elif isinstance(technical_data, pd.DataFrame):
                tech_df = technical_data.copy()
            else:
                raise ValueError("Technical data must be a dict or pandas DataFrame")
            
            return tech_df
            
        except Exception as e:
            print(f"Error during technical data preprocessing: {e}")
            raise
    
    def get_feature_importance(self, top_n=20):

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
    # Example usage with combined fundamental and technical data
    print("\n=== Combined Fundamental + Technical Data Prediction ===")
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
        print(f"Error in prediction: {e}")