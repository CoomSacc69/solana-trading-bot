import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List

class TechnicalAnalyzer:
    def __init__(self):
        self.indicators = {
            'RSI': self._calculate_rsi,
            'MACD': self._calculate_macd,
            'BB': self._calculate_bbands,
            'EMA': self._calculate_ema
        }

    def _calculate_rsi(self, df: pd.DataFrame, length: int = 14) -> pd.Series:
        return df.ta.rsi(length=length)

    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.ta.macd()

    def _calculate_bbands(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.ta.bbands()

    def _calculate_ema(self, df: pd.DataFrame, length: int = 20) -> pd.Series:
        return df.ta.ema(length=length)

    def analyze_patterns(self, df: pd.DataFrame) -> Dict:
        try:
            # Calculate all indicators
            df['RSI'] = self._calculate_rsi(df)
            macd = self._calculate_macd(df)
            df = pd.concat([df, macd], axis=1)
            bb = self._calculate_bbands(df)
            df = pd.concat([df, bb], axis=1)
            df['EMA20'] = self._calculate_ema(df, 20)
            df['EMA50'] = self._calculate_ema(df, 50)

            # Pattern recognition
            patterns = self._identify_patterns(df)
            signals = self._generate_signals(df)

            return {
                'patterns': patterns,
                'signals': signals,
                'indicators': {
                    'rsi': df['RSI'].iloc[-1],
                    'macd': df['MACD_12_26_9'].iloc[-1],
                    'macd_signal': df['MACDs_12_26_9'].iloc[-1],
                    'bb_upper': df['BBU_20_2.0'].iloc[-1],
                    'bb_lower': df['BBL_20_2.0'].iloc[-1]
                }
            }

        except Exception as e:
            print(f"Error in technical analysis: {e}")
            return {}

    def _identify_patterns(self, df: pd.DataFrame) -> List[str]:
        patterns = []
        
        # Trend patterns
        if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1]:
            patterns.append('UPTREND')
        elif df['EMA20'].iloc[-1] < df['EMA50'].iloc[-1]:
            patterns.append('DOWNTREND')

        # RSI patterns
        if df['RSI'].iloc[-1] < 30:
            patterns.append('OVERSOLD')
        elif df['RSI'].iloc[-1] > 70:
            patterns.append('OVERBOUGHT')

        # MACD patterns
        if df['MACD_12_26_9'].iloc[-1] > df['MACDs_12_26_9'].iloc[-1]:
            patterns.append('MACD_BULLISH')
        else:
            patterns.append('MACD_BEARISH')

        return patterns

    def _generate_signals(self, df: pd.DataFrame) -> Dict:
        signals = {
            'strength': 0,
            'direction': 'neutral',
            'confidence': 0.0
        }

        # Calculate signal strength
        rsi = df['RSI'].iloc[-1]
        macd = df['MACD_12_26_9'].iloc[-1]
        macd_signal = df['MACDs_12_26_9'].iloc[-1]
        bb_upper = df['BBU_20_2.0'].iloc[-1]
        bb_lower = df['BBL_20_2.0'].iloc[-1]
        close = df['close'].iloc[-1]

        # Signal strength calculation
        strength = 0
        confidence_factors = []

        # RSI signals
        if rsi < 30:
            strength += 1
            confidence_factors.append(0.8)
        elif rsi > 70:
            strength -= 1
            confidence_factors.append(0.8)

        # MACD signals
        if macd > macd_signal:
            strength += 1
            confidence_factors.append(0.7)
        else:
            strength -= 1
            confidence_factors.append(0.7)

        # Bollinger Bands signals
        if close < bb_lower:
            strength += 1
            confidence_factors.append(0.6)
        elif close > bb_upper:
            strength -= 1
            confidence_factors.append(0.6)

        signals['strength'] = strength
        signals['direction'] = 'bullish' if strength > 0 else 'bearish' if strength < 0 else 'neutral'
        signals['confidence'] = np.mean(confidence_factors) if confidence_factors else 0.0

        return signals