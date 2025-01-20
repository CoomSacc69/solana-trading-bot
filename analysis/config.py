# AI Analysis Configuration

# Model Architecture
MODEL_CONFIG = {
    # CNN Model Parameters
    'cnn_params': {
        'conv_layers': [
            {'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
            {'filters': 32, 'kernel_size': 3, 'activation': 'relu'}
        ],
        'dense_layers': [
            {'units': 64, 'activation': 'relu'},
            {'units': 32, 'activation': 'relu'},
            {'units': 3, 'activation': 'softmax'}  # Buy, Sell, Hold
        ],
        'dropout_rate': 0.2,
        'input_shape': (30, 5),  # 30 timeframes, 5 features (OHLCV)
    },
    
    # Training Parameters
    'training_params': {
        'batch_size': 32,
        'epochs': 100,
        'validation_split': 0.2,
        'early_stopping_patience': 10
    }
}

# Technical Analysis Parameters
TECHNICAL_PARAMS = {
    # Moving Averages
    'sma_periods': [20, 50, 200],
    'ema_periods': [12, 26],
    
    # Oscillators
    'rsi_period': 14,
    'rsi_thresholds': {
        'oversold': 30,
        'overbought': 70
    },
    
    # MACD
    'macd_params': {
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9
    },
    
    # Bollinger Bands
    'bb_params': {
        'period': 20,
        'std_dev': 2
    },
    
    # Volume Analysis
    'volume_ma_period': 20,
    'volume_threshold': 1.5  # Multiple of average volume
}

# Pattern Recognition
PATTERN_CONFIG = {
    # Pattern List (subset of TALib patterns)
    'patterns': [
        'CDL2CROWS',
        'CDL3BLACKCROWS',
        'CDL3INSIDE',
        'CDL3LINESTRIKE',
        'CDL3OUTSIDE',
        'CDL3WHITESOLDIERS',
        'CDLABANDONEDBABY',
        'CDLADVANCEBLOCK',
        'CDLBELTHOLD',
        'CDLBREAKAWAY',
        'CDLDARKCLOUDCOVER',
        'CDLDOJI',
        'CDLDOJISTAR',
        'CDLDRAGONFLYDOJI',
        'CDLENGULFING',
        'CDLEVENINGDOJISTAR'
    ],
    
    # Pattern Weights
    'pattern_weights': {
        'trend_patterns': 1.0,      # Patterns that confirm trends
        'reversal_patterns': 1.2,   # Patterns that signal reversals
        'continuation_patterns': 0.8 # Patterns that signal continuation
    },
    
    # Minimum Pattern Quality
    'min_pattern_quality': 0.6
}

# Signal Generation
SIGNAL_CONFIG = {
    # Component Weights
    'weights': {
        'technical_analysis': 0.6,
        'ai_prediction': 0.4
    },
    
    # Score Thresholds
    'thresholds': {
        'strong_buy': 60,
        'buy': 20,
        'strong_sell': -60,
        'sell': -20
    },
    
    # Confidence Requirements
    'min_confidence': 0.7,
    'min_score': 30
}

# Feature Engineering
FEATURE_CONFIG = {
    # Price Features
    'price_features': [
        'returns',           # Price returns
        'log_returns',       # Log returns
        'volatility',        # Rolling volatility
        'price_momentum',    # Price momentum indicators
        'price_trends'       # Trend indicators
    ],
    
    # Volume Features
    'volume_features': [
        'volume_momentum',   # Volume momentum
        'volume_trends',     # Volume trends
        'volume_ratios'      # Volume ratios
    ],
    
    # Technical Indicators
    'technical_features': [
        'ma_features',       # Moving average features
        'oscillator_features', # Oscillator features
        'pattern_features'    # Pattern recognition features
    ],
    
    # Feature Scaling
    'scaling_method': 'minmax',  # Options: minmax, standard, robust
    'scaling_params': {
        'feature_range': (0, 1)
    }
}

# Performance Metrics
METRICS_CONFIG = {
    # Trading Metrics
    'trading_metrics': [
        'win_rate',
        'profit_factor',
        'sharpe_ratio',
        'max_drawdown',
        'average_profit',
        'average_loss'
    ],
    
    # Model Metrics
    'model_metrics': [
        'accuracy',
        'precision',
        'recall',
        'f1_score'
    ],
    
    # Tracking Windows
    'performance_windows': [
        '1h',
        '4h',
        '1d',
        '1w'
    ]
}