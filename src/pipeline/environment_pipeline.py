
from common.modules import np, pd
from common.modules import List
from common.modules import GramianAngularField, MinMaxScaler
from common.modules import StocksEnv, DummyVecEnv, Monitor, spaces
from src.utils.logging import get_logger


# Global functions for multiprocessing compatibility

def make_single_stock_env(
        ticker: str,  
        gaf_data: pd.DataFrame, 
        gaf_features: List, 
        gaf_timeseries_periods: int, 
        lookback_window: int, 
        logger = None
    ) -> Monitor:
    
    """
    Global function to create a single stock trading environment. This function 
    is outside the class to avoid pickle issues with multiprocessing
    """
    
    try:
        
        # observation dimensions 
        c = len(gaf_features)
        h = w = gaf_timeseries_periods
        
        # create custom environment
        class SingleStockEnv(StocksEnv):
            """Custom environment for single stock trading with GAF data"""
            
            def __init__(self_inner):
                
                # Notice: TradingEnv (parent of StocksEnv) requires a 2D df
                # and asserts df.ndim==2. Create a dummy df to override this
                dummy_df = pd.DataFrame({
                    'dummy': [0] * len(gaf_data['sequences']),
                    'price': gaf_data['targets']
                })

                # use entire dataset starting after lookback_window
                frame_bound = (lookback_window, len(gaf_data['sequences']))
                super().__init__(dummy_df, lookback_window, frame_bound)
            
                self_inner.observation_space = spaces.Box(
                    low = 0.0,
                    high = 1.0, # min/max normalized
                    shape=(c, h, w),  # cnn expects (channels, height, width) format
                    dtype = np.float32
                )
                
            def _get_observation(self_inner):
                """Return observation as a single GAF image in CNN-compatible format"""
                current_gaf = self_inner.signal_features[self_inner._current_tick]
                return current_gaf
                
            def _process_data(self_inner):
                """Returns the entire observation space for an environment"""
                
                # extract GAF sequences for the frame_bound
                i = self_inner.frame_bound[0] - self_inner.window_size
                j = self_inner.frame_bound[1]
                
                gaf_sequences = np.array(gaf_data['sequences'][i:j])
                prices = np.array(gaf_data['targets'][i:j])

                return prices, gaf_sequences

        # create and return monitored environment
        env = Monitor(SingleStockEnv())
        return env
        
    except Exception as e:
        if logger:
            logger.error(f"Error creating environment for {ticker}: {e}")
        raise


class EnvironmentPipeline:
    
    """
    Pipleine for assembling single-process GAF environments and multi-process
    vectorized GAF environments from timeseries data and experiment configs.
    """
    
    def __init__(
            self, 
            experiment_name: str, 
            tickers: List[str], 
            gaf_timeseries_periods: int, 
            gaf_features: List[str], 
            gaf_target: str, 
            lookback_window: int, 
            run_id=None
        ): 
        
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.logger = get_logger(experiment_name, run_id=run_id)
        
        # required items
        self.tickers = tickers
        self.gaf_timeseries_periods = gaf_timeseries_periods
        self.gaf_features = gaf_features
        self.gaf_target = gaf_target
        self.lookback_window = lookback_window
        
        # single process non-vectorized data 
        self.timeseries_data = {}  # {ticker: dataFrame}
        self.gaf_data = {}         # {ticker: GAF sequences}
        self.environments = {}     # {ticker: GAF environment}
        
        # multi-process vectorized environment
        self.vec_environment = None
        self.num_vec_environments = 0
        
        self.logger.info(f"Initialized GAF Pipeline with {len(self.tickers)} tickers")
    
    def build_gaf_dataset(self) -> bool:
        """Transform the pre-loaded timeseries dataset into GAF dataset"""
        
        try:
            self.logger.info("Building GAF dataset")

            # pre-fit the GAF transformer once with sample data
            self.logger.info("Pre-fitting GAF transformer...")
            sample_data = np.random.rand(1,self.gaf_timeseries_periods)
            transformer = GramianAngularField(
                image_size = self.gaf_timeseries_periods,
                method = 'summation'
            )
            transformer.fit(sample_data)
            self.logger.info("GAF transformer pre-fitted successfully")
            
            # assemble the GAF sequences and append to gaf_data
            self.gaf_data = {}
            for ticker, data in self.timeseries_data.items():
                
                self.logger.info(f"Processing GAF transformation for {ticker}")
                
                # prepare features for GAF transformation
                scaler = MinMaxScaler()
                feature_data = data[self.gaf_features].values
                scaled_feature_data = scaler.fit_transform(feature_data)

                # build and stack GAF image sequences
                gaf_sequences = self.build_gaf_sequences(
                    scaled_features = scaled_feature_data, 
                    gaf_transformer = transformer
                )
                
                if gaf_sequences:
                    self.gaf_data[ticker] = {
                        'sequences': gaf_sequences,
                        'targets': data[self.gaf_target].iloc \
                        [self.gaf_timeseries_periods-1:].values.tolist()
                    }
                    self.logger.info(f"Created {len(gaf_sequences)} GAF sequences for {ticker}")
                else:
                    self.logger.warning(f"No GAF sequences created for {ticker}")
            
            if not self.gaf_data:
                self.logger.error("No GAF data created")
                return False
            
            self.logger.info(f"Successfully created GAF data for {len(self.gaf_data)} tickers")
            return True
            
        except Exception as e:
            self.logger.error(f"Error building GAF dataset: {e}")
            return False
    
    def build_gaf_sequences(self, scaled_features: np.ndarray, gaf_transformer: 
                            GramianAngularField) -> List[np.ndarray]:
        """Create GAF sequences from scaled features data and transformer"""
        
        n_samples = len(scaled_features)
        gaf_periods = self.gaf_timeseries_periods
        n_features = len(self.gaf_features)
        
        # create all sliding window indices; extract all sliding windows
        indices = np.arange(gaf_periods-1, n_samples)
        windows = np.array([scaled_features[i-gaf_periods+1:i+1] for i in indices])
        # > shape = (n_windows, gaf_periods, n_features)
        
        # reshape for batch processing
        windows_reshaped = windows.transpose(0, 2, 1).reshape(-1, gaf_periods)
        # > shape = (n_windows * n_features, gaf_periods)
        
        # batch processing: transform all windows
        self.logger.info(f"Performing GAF transformation on {len(windows_reshaped)} windows...")
        all_gaf = gaf_transformer.transform(windows_reshaped)
        # shape = (n_windows * n_features, gaf_periods, gaf_periods)
        
        # vectorized reshape back to individual sequences
        all_gaf = all_gaf.reshape(len(indices), n_features, gaf_periods, gaf_periods)
        # shape = (n_windows, n_features, gaf_periods, gaf_periods)
        
        # convert to list of multi-channel GAF images
        gaf_sequences = []
        for i in range(len(indices)):
            # (stack all feature GAF matrices into a single multi-channel image)
            stacked_gaf = np.stack([all_gaf[i, j] for j in range(n_features)], axis=0)
            gaf_sequences.append(stacked_gaf)
        
        self.logger.info(f"Created {len(gaf_sequences)} GAF sequences")
        return gaf_sequences
    
    def build_individual_environments(self) -> bool:
        """Create individual 'SingleStockEnv' environments for each ticker"""
        
        try:
            self.logger.info("Creating individual trading environments")
            self.environments = {}
            
            for ticker, gaf_data in self.gaf_data.items():
                self.logger.info(f"Creating environment for {ticker}")
                
                # use the global function to create environment
                env = make_single_stock_env(
                    ticker, gaf_data, self.gaf_features, \
                    self.gaf_timeseries_periods, self.lookback_window, logger=self.logger
                )
                self.environments[ticker] = env
                self.logger.info(f"Created environment for {ticker}")
            
            if not self.environments:
                self.logger.error("No environments created")
                return False
            
            self.logger.info(f"Successfully created {len(self.environments)} environments")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating environments: {e}")
            return False
    
    def build_vectorized_environment(self) -> bool:
        """Create a multi-process vectorized environment containing all single
        stock environments using 'SubprocVecEnv' """
        
        if not self.gaf_data:
            self.logger.error("No GAF data available. Call build_gaf_dataset() first.")
            return False
        
        try:
            self.logger.info(f"""Creating vectorized environment with
                        {len(self.tickers)} single stock environments""")
            
            # create environment factory functions using existing environments
            env_fns = []
            for ticker in self.tickers:
                if ticker in self.environments:
                    env_fns.append(lambda env=self.environments[ticker]: env)

            if not env_fns:
                self.logger.error("No valid environment functions created")
                return False
            
            # create SubprocVecEnv
            self.vec_environment = DummyVecEnv(env_fns)
            self.num_vec_environments = len(env_fns)
            
            self.logger.info(f"""Successfully created vectorized environment with
                        {self.num_vec_environments} environments""")
                        
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating vectorized environment: {e}")
            return False
    
    def exe_env_pipeline(self) -> bool:
        """
        Build the multi-process vectorized GAF environment by creating the
        individual gaf datasets and environments. The pipeline assumes that all 
        data is already preprocessed and stored in the data directory
        """
        
        try:
            
            self.logger.info("Building GAF pipeline (data already loaded)...")
            
            # Step 1: Build the GAF dataset
            self.logger.info("Step 1: Building GAF dataset...")
            if not self.build_gaf_dataset():
                self.logger.error("Failed to build GAF dataset")
                return False

            # Step 2: Create the individual GAF environments
            self.logger.info("Step 2: Creating trading environments...")
            if not self.build_individual_environments():
                self.logger.error("Failed to build GAF environments")
                return False
        
            # Step 3: Create the multi-process vectorized environment
            self.logger.info("Step 3: Creating vectorized environment...")
            if not self.build_vectorized_environment():
                self.logger.error("Failed to build vectorized GAF environment")
                return False
            
            self.logger.info("GAF pipeline built successfully")
            return True
        
        except Exception as e:
            self.logger.error(f"Error building GAF pipeline: {e}")
            return False
