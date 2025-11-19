
from common.modules import os, pd, np, List, Dict, Tuple
from common.modules import TA
from common.modules import USFederalHolidayCalendar
from src.utils.logging import log_function_call, log_execution_time, get_logger
import time
from datetime import datetime

try:
    from alpha_vantage.timeseries import TimeSeries
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False
    print("Warning: alpha_vantage library not installed. Install with: pip install alpha_vantage")

REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
DATA_DIR = ".\\dataset-cache"

# Alpha Vantage rate limiting: 5 calls per minute for free tier
RATE_LIMIT_CALLS = 5
RATE_LIMIT_WINDOW = 60  # seconds


class RateLimiter:
    """Simple rate limiter for Alpha Vantage API calls"""
    
    def __init__(self, max_calls: int = RATE_LIMIT_CALLS, window: int = RATE_LIMIT_WINDOW):
        self.max_calls = max_calls
        self.window = window
        self.call_times = []
    
    def wait_if_needed(self):
        """Wait if we've exceeded the rate limit"""
        now = time.time()
        # Remove calls outside the current window
        self.call_times = [t for t in self.call_times if now - t < self.window]
        
        if len(self.call_times) >= self.max_calls:
            # Calculate how long to wait
            oldest_call = min(self.call_times)
            wait_time = self.window - (now - oldest_call) + 1  # Add 1 second buffer
            if wait_time > 0:
                time.sleep(wait_time)
                # Update call times after waiting
                now = time.time()
                self.call_times = [t for t in self.call_times if now - t < self.window]
        
        # Record this call
        self.call_times.append(time.time())


class DataPipeline: 
    
    """
    Pipeline for managing all tabular timeseries data preparation using Alpha Vantage API.
    Using an experiment config file, data is retrieved, preprocessed, validated, and saved.
    """
    
    def __init__(
        self, 
        experiment_name, 
        run_id = None,
        api_key: str = None
    ) -> None:
        
        if not ALPHA_VANTAGE_AVAILABLE:
            raise ImportError("alpha_vantage library is required. Install with: pip install alpha_vantage")
        
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.logger = get_logger(experiment_name, run_id=run_id)
        self.data_dir = DATA_DIR
        os.makedirs(DATA_DIR, exist_ok = True)
        
        # Initialize Alpha Vantage API
        if api_key is None:
            # Try to get from environment variable
            api_key = os.environ.get('ALPHA_VANTAGE_API_KEY')
            if api_key is None:
                raise ValueError("Alpha Vantage API key required. Provide via api_key parameter or ALPHA_VANTAGE_API_KEY environment variable")
        
        self.api_key = api_key
        self.ts = TimeSeries(key=api_key, output_format='pandas')
        self.rate_limiter = RateLimiter()
        
        self.logger.info(f"Data manager initialized with directory: {DATA_DIR}")
        self.logger.info("Using Alpha Vantage API for data fetching")
    
    def _map_interval(self, interval: str) -> str:
        """Map interval from config format to Alpha Vantage format"""
        interval_map = {
            '1d': 'daily',
            '1day': 'daily',
            'daily': 'daily',
            '1m': '1min',
            '1min': '1min',
            '5m': '5min',
            '5min': '5min',
            '15m': '15min',
            '15min': '15min',
            '30m': '30min',
            '30min': '30min',
            '60m': '60min',
            '60min': '60min'
        }
        return interval_map.get(interval.lower(), 'daily')
    
    @log_function_call
    def fetch_price_data(
        self, 
        ticker: str, 
        start_date: str, 
        end_date: str, 
        interval: str
    ) -> pd.DataFrame:
        """Fetch a single stock's timeseries data from Alpha Vantage API"""
        
        try:
            # Apply rate limiting
            self.rate_limiter.wait_if_needed()
            
            # Map interval to Alpha Vantage format
            av_interval = self._map_interval(interval)
            
            # fetch data from alpha vantage api
            self.logger.info(f"Fetching {ticker} data [{start_date} - {end_date}] from Alpha Vantage")
            
            if av_interval == 'daily':
                # Use daily endpoint
                data, meta_data = self.ts.get_daily(symbol=ticker, outputsize='full')
            else:
                # Use intraday endpoint
                data, meta_data = self.ts.get_intraday(symbol=ticker, interval=av_interval, outputsize='full')
            
            if data.empty:
                self.logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
            
            # Alpha Vantage returns columns as: '1. open', '2. high', '3. low', '4. close', '5. volume'
            # Rename to standard format
            column_mapping = {
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            }
            
            # Rename columns
            for old_col, new_col in column_mapping.items():
                if old_col in data.columns:
                    data = data.rename(columns={old_col: new_col})
            
            # Ensure we have the required columns
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in data.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns for {ticker}: {missing_cols}")
                return pd.DataFrame()
            
            # Filter data by date range
            data.index = pd.to_datetime(data.index)
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Filter to requested date range
            data = data[(data.index >= start_dt) & (data.index <= end_dt)]
            
            # Sort by date (ascending)
            data = data.sort_index()
            
            # Ensure index is datetime
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # Validate that we have the required price columns
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in data.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns for {ticker}: {missing_cols}")
                return pd.DataFrame()
            
            self.logger.info(f"Successfully fetched {len(data)} records for {ticker}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    @log_function_call
    def fetch_technical_data(
        self, 
        data: pd.DataFrame, 
        config: Dict
    ) -> pd.DataFrame:
        """Fetch technical indicators for stocks in the dataset"""
        
        try:
            if data.empty:
                self.logger.warning("Empty data provided for technical indicators")
                return data
            
            df = data.copy() # create a copy to avoid modifying original
            
            # validate required columns exist
            missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Missing required columns for technical indicators: {missing_cols}")
                return data
            
            # Current technical features configured for fetching 
            # (1) SMA: simple moving average
            # (2) RSI: relative strength index 
            # (3) OBV: on-balance volume
            
            gaf_features = config.get('GAF features', [])
            
            # calculate SMA if requested
            if 'SMA' in gaf_features:
                sma_periods = config.get('SMA periods')
                df['SMA'] = TA.SMA(df, period=sma_periods, column='close')
            
            # calculate RSI if requested
            if 'RSI' in gaf_features:
                rsi_periods = config.get('RSI periods')
                df['RSI'] = TA.RSI(df, period=rsi_periods, column='close')

            # calculate OBV if requested
            if 'OBV' in gaf_features:
                df['OBV'] = TA.OBV(df, column='close')
            
            # fill NaN values
            df = df.ffill().bfill()
            
            self.logger.info(f"Calculated technical indicators for {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return data
    
    @log_function_call
    def save_data(
        self, 
        data: pd.DataFrame, 
        filename: str
    ) -> bool:
        """Save a stock's data to a csv file in the data directory"""
        
        try:
            filepath = os.path.join(self.data_dir, filename)
            data.to_csv(filepath, index=True)
            self.logger.info(f"Data saved to {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            return False
    
    @log_function_call
    def load_data(
        self, 
        filename: str
    ) -> pd.DataFrame:
        """Load a stock's data from a csv file in the data directory"""
        
        try:
            filepath = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(filepath):
                self.logger.error(f"File not found: {filepath}")
                return pd.DataFrame()
            
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)            
            self.logger.info(f"Data loaded from {filepath}: {len(data)} records")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    @log_function_call
    def check_cache(
        self, 
        ticker: str, 
        required_features: List[str] = None
    ) -> Dict:
        """Check what date range and features are covered in the existing data 
        file for a given stock. Return status message about file contents"""
        
        bad_status = {
            'exists': False, 
            'file_path': None, 
            'data': None,
            'date_range': None, 
            'features': [], 
            'gaps': []
        }
        
        try: 
            # fetch the existing data if available in cache
            csv_path = os.path.join(self.data_dir, f"{ticker}_data.csv")
            existing_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            
            # retrieve the date range
            date_range = {
                'start': existing_data.index.min().strftime('%Y-%m-%d'),
                'end': existing_data.index.max().strftime('%Y-%m-%d')
            }
            
            # get available features and check for missing ones 
            existing_features = list(existing_data.columns)
            missing_features = []
            if required_features: missing_features = [feat for feat in \
                    required_features if feat not in existing_features]
            
            # return status about data cache's contents
            return {
                'exists': True,
                'file_path': csv_path,
                'data': existing_data,
                'date_range': date_range,
                'features': existing_features,
                'missing_features': missing_features,
                'record_count': len(existing_data)
            }
            
        except Exception as e:
            self.logger.error(f"Error checking existing data for {ticker}: {e}")
            bad_status['file_path'] = csv_path
            return bad_status
            
    @log_function_call
    def check_data_gaps(
        self, 
        ticker: str, 
        required_start: str, 
        required_end: str, 
        required_features: List[str]
    ) -> Dict:
        """Determine if gaps exist in a stock's existing data: check if start/end 
        dates match required range, check if required features are present"""
        
        try:
            existing_info = self.check_cache(ticker, required_features)
            
            # return this status if the data requires a full fetch
            if not existing_info['exists']:
                return {
                'needs_full_fetch': True,
                'missing_dates': {'start': required_start, 'end': required_end},
                'missing_features': required_features,
                'reason': 'No existing data'
            }
            
            # check date gaps
            existing_start = existing_info['date_range']['start']
            existing_end = existing_info['date_range']['end']
            
            required_start_dt = pd.to_datetime(required_start)
            existing_start_dt = pd.to_datetime(existing_start)
            required_end_dt = pd.to_datetime(required_end)
            existing_end_dt = pd.to_datetime(existing_end)
            
            missing_dates = {}
            
            # check if earlier data is needed
            if required_start_dt < existing_start_dt:
                missing_dates['start'] = required_start
                missing_dates['existing_start'] = existing_start
            
            # check if later data is needed
            if required_end_dt > existing_end_dt:
                missing_dates['end'] = required_end
                missing_dates['existing_end'] = existing_end
            
            # check for missing features
            missing_features = existing_info['missing_features']
            
            # determine if ANY data needs to be fetched
            needs_fetch = bool(missing_dates) or bool(missing_features)
            
            return {
                'needs_full_fetch': not existing_info['exists'],
                'needs_incremental_fetch': needs_fetch and existing_info['exists'],
                'existing_info': existing_info,
                'missing_dates': missing_dates,
                'missing_features': missing_features,
                'existing_date_range': existing_info['date_range'],
                'required_date_range': {'start': required_start, 'end': required_end},
            }
            
        except Exception as e:
            self.logger.error(f"Error identifying data gaps for {ticker}: {e}")
            # return status for full fetch if error encountered
            return {
                'needs_full_fetch': True,
                'missing_dates': {'start': required_start, 'end': required_end},
                'missing_features': required_features,
                'reason': f'Error: {str(e)}'
            }
    
    @log_function_call
    def fetch_incremental_data(
        self, 
        config: Dict, 
        ticker: str, 
        missing_dates: Dict, 
        missing_features: List[str]
    ) -> pd.DataFrame:
        """Fetch missing data values for a single stock"""

        try:
            # early return if no incremental fetch is needed
            if not missing_dates and not missing_features:
                self.logger.info(f"No missing dates or features for {ticker}")
                return pd.DataFrame()

            # determine what date ranges need to be fetched
            fetch_periods = []
            if missing_dates:
                if 'start' in missing_dates and 'end' in missing_dates:
                    # data is needed before and after existing range
                    fetch_periods.extend([
                        {
                            'start': missing_dates['start'],
                            'end': missing_dates['existing_start'],
                            'type': 'pre_period'
                        },
                        {
                            'start': missing_dates['existing_end'],
                            'end': missing_dates['end'],
                            'type': 'post_period'
                        }
                    ])
                    self.logger.info(f"""Fetching incremental data for {ticker}:
                              ({missing_dates['start']} to 
                               {missing_dates['existing_start']}) and 
                              ({missing_dates['existing_end']} to 
                               {missing_dates['end']})""")
                
                elif 'start' in missing_dates:
                    # data is needed only before existing range
                    fetch_periods.append({
                        'start': missing_dates['start'],
                        'end': missing_dates['existing_start'],
                        'type': 'pre_period'
                    })
                    self.logger.info(f"""Fetching incremental data for {ticker}:
                               ({missing_dates['start']} to 
                                {missing_dates['existing_start']})""")
                
                elif 'end' in missing_dates:
                    # data is needed only after existing range
                    fetch_periods.append({
                        'start': missing_dates['existing_end'],
                        'end': missing_dates['end'],
                        'type': 'post_period'
                    })
                    self.logger.info(f"""Fetching incremental data for {ticker}: 
                               ({missing_dates['existing_end']} to 
                                {missing_dates['end']})""")
            
            # fetch data for each required period
            fetched_dataframes = []
            for period in fetch_periods:
                price_data = self.fetch_price_data(ticker, period['start'], \
                            period['end'], interval = config['Update frequency'])
                
                if not price_data.empty:
                    # calculate technical indicators if we have price data
                    price_data = self.fetch_technical_data(price_data, config)
                    fetched_dataframes.append(price_data)
                    self.logger.info(f"""Fetched {len(price_data)} records for 
                                {ticker}{period['type']} ({period['start']} to 
                                {period['end']})""")
                else:
                    self.logger.warning(f"""No price data fetched for {ticker} 
                                   {period['type']} ({period['start']} 
                                   to {period['end']})""")

            # merge all fetched data
            if fetched_dataframes:
                new_data = pd.concat(fetched_dataframes, ignore_index=False)
                new_data = new_data[~new_data.index.duplicated(keep='last')].sort_index()
            else:
                new_data = pd.DataFrame()

            # if we have missing features but no new price data, handle features
            if missing_features and new_data.empty:
                self.logger.info(f"""No new price data for {ticker}, but missing 
                            features need to be calculated""")
                return pd.DataFrame()  # features will be calculated on existing data
            
            # if we have new data but missing features, recalculate indicators
            if missing_features and not new_data.empty:
                self.logger.info(f"""Recalculating technical indicators for {ticker}
                            to include missing features""")
                new_data = self.fetch_technical_data(new_data, config)
            
            if new_data.empty:
                self.logger.warning(f"No new data fetched for {ticker}")
            else:
                self.logger.info(f"Successfully fetched {len(new_data)} records for {ticker}")
            
            return new_data
            
        except Exception as e:
            self.logger.error(f"Error fetching incremental data for {ticker}: {e}")
            return pd.DataFrame()
    
    @log_function_call
    def merge_incremental_data(
        self, 
        existing_data: pd.DataFrame, 
        new_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge newly fetched data with existing data"""
        
        try:
            # no merging if either is empty
            if existing_data.empty: return new_data
            if new_data.empty: return existing_data
            
            # combine data; remove duplicates based on index (date); sort
            combined = pd.concat([existing_data, new_data])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()

            self.logger.info(f"Merged data: {len(combined)} total records")
            return combined
            
        except Exception as e:
            self.logger.error(f"Error merging data: {e}")
            return existing_data
    
    @log_function_call
    def exe_data_pipeline(
        self, 
        config: Dict
    ) -> Dict[str, pd.DataFrame]:
        """Prepare and return all tabular timeseries data needed for a given
        experiment. Retrieve price and technical data; preprocess price and 
        technical data; and load/save datasets in project data directory"""
        
        tickers = config['Tickers']
        start_date = config['Start date']
        end_date = config['End date']
        start_date_dt = pd.to_datetime(start_date)
        end_date_dt = pd.to_datetime(end_date)
        
        try:
            
            # calculate fetch start date with buffer for technical indicators
            max_adj = 0
            if config.get('SMA periods'): 
                max_adj = max(max_adj, config['SMA periods'])
            if config.get('RSI periods'): 
                max_adj = max(max_adj, config['RSI periods'])
            adjust_days = config.get('GAF timeseries periods', 0) + max_adj
            
            fetch_start = self.business_date_adjust(start_date, adjust_days)
            self.logger.info(f"Adjusted fetch date to {fetch_start}")
            
            required_features = config.get('GAF features', [])
            
            # Process each ticker
            stock_data = {}
            for ticker in tickers:
                self.logger.info(f"Processing {ticker} with smart caching...")
                
                # check what data we already have
                gap_info = self.check_data_gaps(ticker, fetch_start, end_date, required_features)
                
                # Process based on what's needed
                if gap_info['needs_full_fetch']:
                    # fetch all data from scratch
                    self.logger.info(f"Fetching all data for {ticker} (none existing)")
                    data = self.fetch_price_data(ticker, fetch_start, end_date, \
                                            interval = config['Update frequency'])
                    
                    if not data.empty:
                        # calculate technical indicators
                        data = self.fetch_technical_data(data, config)
                        
                        # validate data
                        is_valid, issues = self.validate_data(data, config)
                        if is_valid: # save
                            self.save_data(data, f"{ticker}_data.csv")
                            self.logger.info(f"Fetched and saved data for {ticker}")
                        else:
                            self.logger.warning(f"Data validation failed for {ticker}: {issues}")
                            data = pd.DataFrame()
                    else:
                        self.logger.error(f"Failed to fetch data for {ticker}")
                        
                elif gap_info['needs_incremental_fetch']:
                    # fetch only missing data
                    self.logger.info(f"Fetching incremental data for {ticker}")
                    
                    existing_data = gap_info['existing_info']['data']
                    
                    new_data = self.fetch_incremental_data(
                        config,
                        ticker,
                        gap_info['missing_dates'],
                        gap_info['missing_features']
                    )
                    
                    if not new_data.empty:
                        self.logger.info(f"Fetched {len(new_data)} new records for {ticker}")
                        # merge with existing data
                        data = self.merge_incremental_data(existing_data, new_data)
                    else:
                        # use existing data 
                        data = existing_data
                    
                    # save updated data
                    if not data.empty:
                        self.save_data(data, f"{ticker}_data.csv")
                        self.logger.info(f"Updated and saved data for {ticker}")
                        
                else:
                    # use existing data
                    self.logger.info(f"Using existing data for {ticker}")
                    data = gap_info['existing_info']['data']
                    self.logger.info(f"Loaded existing data for {ticker}")
                
                # trim data to experiment date range and add to results
                if not data.empty:
                    data = self.trim_dates(data, start_date_dt, end_date_dt)
                    
                    # trim excess features if config has fewer features than cached data
                    if required_features:
                        # Find features in data that are not in the required features list
                        current_features = set(data.columns)
                        required_features_set = set(required_features)
                        excess_features = list(current_features - required_features_set)
                        
                        data = self.trim_features(data, excess_features)
                    
                    stock_data[ticker] = data

                else:
                    self.logger.error(f"Failed to prepare data for {ticker}")
            
            self.logger.info(f"Successfully prepared data for {len(stock_data)} tickers")
            
            return stock_data
            
        except Exception as e:
            self.logger.error(f"Error preparing data for experiment: {e}")
            return {}

    @log_function_call
    def validate_data(
        self, 
        data: pd.DataFrame, 
        config: Dict
    ) -> Tuple[bool, List[str]]:
        """Validate data quality. Return list of issues encountered if needed"""
        
        issues = []
        
        # check if data is empty
        if data.empty:
            issues.append("Data is empty")
            return False, issues
        
        # check required features
        required_features = config.get('GAF features', [])
        if required_features:
            missing_cols = [col for col in required_features if col not in data.columns]
            if missing_cols:
                issues.append(f"Missing required columns: {missing_cols}")
            
            # check for infinite values
            inf_counts = np.isinf(data[required_features]).sum()
            if inf_counts.sum() > 0:
                issues.append(f"Infinite values found: {inf_counts.to_dict()}")
        
        # check for negative prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        if all(col in data.columns for col in price_cols):
            negative_prices = (data[price_cols] < 0).any().any()
            if negative_prices:
                issues.append("Negative prices found")
        
        is_valid = len(issues) == 0
        
        if is_valid: self.logger.info("Data validation passed")
        else: self.logger.warning(f"Data validation failed: {issues}")
        
        return is_valid, issues
    
    def business_date_adjust(
        self, 
        date: str, 
        n_days: int
    ) -> str:
        """Adjust date by subtracting business days"""
        
        current_date = pd.to_datetime(date)
        
        # fetch holidays within one year of the current date
        calendar = USFederalHolidayCalendar()
        holidays = calendar.holidays(current_date - pd.Timedelta(days=365), \
                    current_date).date 
        
        # iteratively subtract business days; account for holidays
        business_days_subtracted = 0
        while business_days_subtracted < n_days:
            current_date -= pd.Timedelta(days=1)
            if current_date.weekday() < 5 and current_date not in holidays:
                business_days_subtracted += 1
            
        return current_date.strftime('%Y-%m-%d')
    
    def trim_dates(
        self, 
        data: pd.DataFrame, 
        start_date_dt: pd.Timestamp, 
        end_date_dt: pd.Timestamp
    ) -> pd.DataFrame:
        """Trim data to experiment date range with proper timezone handling"""
        if data.empty:
            return data
        
        # handle timezone consistently
        timezone = data.index[0].tz if hasattr(data.index[0], 'tz') else None
        
        if timezone is not None:
            start_dt = start_date_dt.tz_localize(timezone) if start_date_dt.tz \
                is None else start_date_dt
            end_dt = end_date_dt.tz_localize(timezone) if end_date_dt.tz \
                is None else end_date_dt
        else:
            start_dt = start_date_dt.tz_localize('UTC') if start_date_dt.tz is \
                None else start_date_dt
            end_dt = end_date_dt.tz_localize('UTC') if end_date_dt.tz is \
                None else end_date_dt
            data.index = data.index.tz_localize('UTC')
        
        # trim to experiment range
        trimmed_data = data.loc[(data.index >= start_dt) & (data.index <= end_dt)]
        self.logger.debug(f"Trimmed data to experiment range: {len(trimmed_data)} records")
        
        return trimmed_data
    
    def trim_features(
        self, 
        data: pd.DataFrame, 
        features_to_remove: List[str]
    ) -> pd.DataFrame:
        """Trim specified features from the dataframe"""

        # early exits
        if data.empty:
            self.logger.warning("Cannot trim features from empty dataframe")
            return data
        
        if not features_to_remove:
            self.logger.debug("No features specified for removal")
            return data
        
        # check which features actually exist in the dataframe
        existing_features = list(data.columns)
        features_to_remove_existing = [feat for feat in features_to_remove if feat in existing_features]
        features_not_found = [feat for feat in features_to_remove if feat not in existing_features]
        
        if features_not_found:
            self.logger.warning(f"Features not found in dataframe: {features_not_found}")
        
        if not features_to_remove_existing:
            self.logger.info("No specified features found in dataframe to remove")
            return data
        
        # remove the features
        trimmed_data = data.drop(columns=features_to_remove_existing)
        
        self.logger.info(f"Removed features from dataframe: {features_to_remove_existing}")
        self.logger.debug(f"Dataframe now has {len(trimmed_data.columns)} columns: {list(trimmed_data.columns)}")
        
        return trimmed_data



if __name__ == "__main__":
    
    my_api_key = 'Y9EJACE06U0N9CNV'
     
    from src.utils.configurations import load_config
    
    large_cap_experiment = "Large-Cap"
    # medium_cap_experiment = "Medium-Cap"
    # small_cap_experiment = "Small-Cap"
    
    large_cap_config = load_config(large_cap_experiment)
    # medium_cap_config = load_config(medium_cap_experiment)
    # small_cap_config = load_config(small_cap_experiment)
    
    
    # Initialize with API key (or set ALPHA_VANTAGE_API_KEY environment variable)
    inst = DataPipeline(large_cap_experiment, api_key=my_api_key )
    inst.exe_data_pipeline(large_cap_config)
    
    # inst = DataPipeline(medium_cap_experiment, api_key='YOUR_API_KEY_HERE')
    # inst.exe_data_pipeline(medium_cap_config)
    
    # inst = DataPipeline(small_cap_experiment, api_key='YOUR_API_KEY_HERE')
    # inst.exe_data_pipeline(small_cap_config)

