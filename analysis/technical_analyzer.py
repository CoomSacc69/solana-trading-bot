import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from typing import List, Dict, Tuple
import talib
import logging

class AITechnicalAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = MinMaxScaler()
        self.pattern_names = [
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE', 
            'CDL3OUTSIDE', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY', 'CDLADVANCEBLOCK',
            'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
            'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR',
            'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
            'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE'
        ]
        
        self.model = self._build_cnn_model()
        
    def _build_cnn_model(self):
        """Build and compile CNN model for pattern recognition"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(30, 5)),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Conv1D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')  # Buy, Sell, Hold
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare OHLCV data for analysis"""
        try:
            # Ensure data is properly formatted
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.copy()
            df.columns = df.columns.str.lower()
            
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Missing required OHLCV columns")
            
            # Calculate technical indicators
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['rsi'] = talib.RSI(df['close'].values)
            df['macd'], df['macd_signal'], _ = talib.MACD(df['close'].values)
            
            # Calculate candlestick patterns
            for pattern in self.pattern_names:
                pattern_function = getattr(talib, pattern)
                df[pattern] = pattern_function(
                    df['open'].values, 
                    df['high'].values, 
                    df['low'].values, 
                    df['close'].values
                )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            return pd.DataFrame()

    def analyze_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze candlestick patterns and technical indicators"""
        try:
            prepared_data = self.prepare_data(df)
            if prepared_data.empty:
                return {}

            # Identify candlestick patterns
            pattern_signals = {}
            for pattern in self.pattern_names:
                if pattern in prepared_data.columns and prepared_data[pattern].iloc[-1] != 0:
                    pattern_signals[pattern] = prepared_data[pattern].iloc[-1]
            
            # Calculate trend indicators
            current_price = prepared_data['close'].iloc[-1]
            sma_20 = prepared_data['sma_20'].iloc[-1]
            sma_50 = prepared_data['sma_50'].iloc[-1]
            rsi = prepared_data['rsi'].iloc[-1]
            macd = prepared_data['macd'].iloc[-1]
            macd_signal = prepared_data['macd_signal'].iloc[-1]
            
            # Analyze trend
            trend = {
                'short_trend': 'bullish' if current_price > sma_20 else 'bearish',
                'long_trend': 'bullish' if current_price > sma_50 else 'bearish',
                'rsi_signal': 'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral',
                'macd_signal': 'bullish' if macd > macd_signal else 'bearish'
            }
            
            # Calculate strength of signals
            bullish_patterns = sum(1 for v in pattern_signals.values() if v == 100)
            bearish_patterns = sum(1 for v in pattern_signals.values() if v == -100)
            
            # Generate trading signal score (-100 to 100)
            signal_score = self._calculate_signal_score(
                trend, bullish_patterns, bearish_patterns, rsi, macd
            )
            
            return {
                'patterns': pattern_signals,
                'trend': trend,
                'technical_indicators': {
                    'rsi': rsi,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'sma_20': sma_20,
                    'sma_50': sma_50
                },
                'signal_score': signal_score
            }
            
        except Exception as e:
            self.logger.error(f"Error in pattern analysis: {e}")
            return {}

    def _calculate_signal_score(
        self, trend: Dict, bullish_patterns: int, 
        bearish_patterns: int, rsi: float, macd: float
    ) -> float:
        """Calculate overall trading signal score"""
        try:
            score = 0
            
            # Trend scoring (40% weight)
            if trend['short_trend'] == 'bullish': score += 20
            if trend['long_trend'] == 'bullish': score += 20
            if trend['short_trend'] == 'bearish': score -= 20
            if trend['long_trend'] == 'bearish': score -= 20
            
            # Pattern scoring (30% weight)
            pattern_score = (bullish_patterns - bearish_patterns) * 10
            score += min(max(pattern_score, -30), 30)
            
            # RSI scoring (15% weight)
            if rsi < 30: score += 15  # Oversold
            elif rsi > 70: score -= 15  # Overbought
            elif 45 <= rsi <= 55: score += 7.5  # Neutral
            
            # MACD scoring (15% weight)
            if macd > 0: score += 15
            else: score -= 15
            
            return min(max(score, -100), 100)
            
        except Exception as e:
            self.logger.error(f"Error calculating signal score: {e}")
            return 0

    def predict_price_movement(self, historical_data: pd.DataFrame) -> Tuple[str, float]:
        """Predict price movement using the CNN model"""
        try:
            # Prepare data for CNN
            prepared_data = self.prepare_data(historical_data)
            if prepared_data.empty:
                return 'hold', 0.0

            # Get last 30 periods
            features = ['open', 'high', 'low', 'close', 'volume']
            last_30 = prepared_data[features].tail(30).values
            
            # Normalize the data
            normalized_data = self.scaler.fit_transform(last_30)
            
            # Reshape for CNN input (batch_size, timesteps, features)
            X = normalized_data.reshape(1, 30, 5)
            
            # Get prediction
            prediction = self.model.predict(X, verbose=0)
            prediction_idx = np.argmax(prediction[0])
            confidence = prediction[0][prediction_idx]
            
            # Convert to signal
            signals = ['sell', 'hold', 'buy']
            return signals[prediction_idx], confidence
            
        except Exception as e:
            self.logger.error(f"Error in movement prediction: {e}")
            return 'hold', 0.0

    def get_trade_recommendation(self, df: pd.DataFrame) -> Dict:
        """Get final trading recommendation combining all analyses"""
        try:
            # Get pattern analysis
            pattern_analysis = self.analyze_patterns(df)
            if not pattern_analysis:
                return {}
            
            # Get AI prediction
            movement, confidence = self.predict_price_movement(df)
            
            # Combine signals
            signal_score = pattern_analysis['signal_score']
            
            # Weight the signals
            technical_weight = 0.6
            ai_weight = 0.4
            
            # Convert AI prediction to score (-100 to 100)
            ai_score = {
                'buy': 100,
                'hold': 0,
                'sell': -100
            }[movement] * confidence
            
            # Calculate final score
            final_score = (signal_score * technical_weight) + (ai_score * ai_weight)
            
            # Determine action
            if final_score > 60:
                action = 'strong_buy'
            elif final_score > 20:
                action = 'buy'
            elif final_score < -60:
                action = 'strong_sell'
            elif final_score < -20:
                action = 'sell'
            else:
                action = 'hold'
            
            return {
                'action': action,
                'score': final_score,
                'confidence': confidence,
                'technical_analysis': pattern_analysis,
                'ai_prediction': {
                    'movement': movement,
                    'confidence': confidence
                },
                'timestamp': pd.Timestamp.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error in trade recommendation: {e}")
            return {}