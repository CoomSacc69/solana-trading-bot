# Trading Bot Configuration

# Trading Parameters
TRADING_PARAMS = {
    # Position Management
    'position_size_usd': 1000,     # Default position size in USD
    'max_positions': 3,            # Maximum number of concurrent positions
    'max_portfolio_risk': 0.02,    # Maximum portfolio risk per trade (2%)
    
    # Market Requirements
    'min_volume_usd': 500000,      # Minimum 24h volume
    'min_liquidity_usd': 100000,   # Minimum liquidity
    'min_price_usd': 0.00001,      # Minimum token price
    
    # Trade Entry/Exit
    'take_profit_pct': 100,        # Take profit percentage (2x)
    'stop_loss_pct': 15,           # Stop loss percentage
    'trailing_stop_pct': 5,        # Trailing stop percentage
    'max_slippage_pct': 1,         # Maximum allowed slippage
    
    # Price Movement
    'min_price_change': 15,        # Minimum price change to trigger analysis
    'max_price_change': 100,       # Maximum allowed price change
    'volatility_threshold': 0.5,   # Maximum allowed volatility
    
    # Time Constraints
    'min_time_between_trades': 300, # Minimum seconds between trades
    'max_trade_duration': 86400,    # Maximum trade duration (24h)
}

# AI Analysis Parameters
AI_PARAMS = {
    # Confidence Thresholds
    'min_confidence': 0.7,         # Minimum AI prediction confidence
    'min_pattern_score': 30,       # Minimum technical pattern score
    
    # Signal Weights
    'technical_weight': 0.6,       # Weight for technical analysis
    'ai_weight': 0.4,              # Weight for AI predictions
    
    # Pattern Recognition
    'min_pattern_quality': 0.6,    # Minimum pattern quality score
    'pattern_lookback': 30,        # Candles to analyze for patterns
    
    # Model Parameters
    'cnn_input_shape': (30, 5),    # Input shape for CNN model
    'lstm_sequence_length': 50,    # Sequence length for LSTM
}

# API Configuration
API_CONFIG = {
    # Rate Limiting
    'max_requests_per_min': 30,    # Maximum API requests per minute
    'request_delay': 1.0,          # Delay between requests in seconds
    
    # Endpoints
    'base_url': 'https://api.dexscreener.com/latest/dex',
    'websocket_url': 'wss://api.dexscreener.com/ws',
    
    # Headers
    'headers': {
        'User-Agent': 'Mozilla/5.0',
        'Accept': 'application/json',
        'Origin': 'https://dexscreener.com',
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'logs/trading.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': True
        },
    }
}