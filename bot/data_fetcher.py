import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import logging
from typing import Dict, List, Optional

class DataFetcher:
    def __init__(self):
        self.base_url = "https://api.dexscreener.com/latest/dex"
        self.session = None
        self.logger = logging.getLogger(__name__)
        self.cache = {}
        self.cache_duration = timedelta(minutes=5)

    async def init_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers={
                    'User-Agent': 'Mozilla/5.0',
                    'Accept': 'application/json'
                }
            )

    async def close(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def get_historical_data(
        self, 
        token_address: str, 
        timeframe: str = '5m',
        limit: int = 200
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data"""
        cache_key = f"{token_address}_{timeframe}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data, cache_time = self.cache[cache_key]
            if datetime.now() - cache_time < self.cache_duration:
                return cached_data

        try:
            await self.init_session()
            endpoint = f"{self.base_url}/pairs/{token_address}/ohlcv/{timeframe}"
            
            async with self.session.get(endpoint) as response:
                if response.status != 200:
                    self.logger.error(f"Error fetching data: {response.status}")
                    return pd.DataFrame()
                
                data = await response.json()
                
                if not data or 'ohlcv' not in data:
                    return pd.DataFrame()
                
                df = pd.DataFrame(
                    data['ohlcv'],
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                
                # Process data
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                # Sort and limit
                df = df.sort_values('timestamp', ascending=True).tail(limit)
                
                # Cache the result
                self.cache[cache_key] = (df, datetime.now())
                
                return df

        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    async def get_token_metrics(self, token_address: str) -> Dict:
        """Get current token metrics"""
        try:
            await self.init_session()
            endpoint = f"{self.base_url}/tokens/{token_address}"
            
            async with self.session.get(endpoint) as response:
                if response.status != 200:
                    return {}
                
                data = await response.json()
                
                if not data or 'pairs' not in data:
                    return {}
                
                # Get the most liquid pair
                pairs = sorted(
                    data['pairs'],
                    key=lambda x: float(x.get('liquidity', {}).get('usd', 0) or 0),
                    reverse=True
                )
                
                if not pairs:
                    return {}
                    
                pair = pairs[0]
                
                return {
                    'price': float(pair.get('priceUsd', 0)),
                    'volume_24h': float(pair.get('volume', {}).get('h24', 0) or 0),
                    'liquidity': float(pair.get('liquidity', {}).get('usd', 0) or 0),
                    'price_change_24h': float(pair.get('priceChange', {}).get('h24', 0) or 0),
                    'txns_24h': int(pair.get('txns', {}).get('h24', 0) or 0)
                }

        except Exception as e:
            self.logger.error(f"Error fetching token metrics: {e}")
            return {}

    async def get_market_snapshot(
        self,
        min_volume: float = 100000,
        min_liquidity: float = 50000
    ) -> pd.DataFrame:
        """Get snapshot of all relevant market pairs"""
        try:
            pairs_data = []
            search_terms = ["solana", "SOL", "RAY", "ORCA", "USDC", "USDT", "BONK"]
            
            for term in search_terms:
                try:
                    endpoint = f"{self.base_url}/search?q={term}"
                    async with self.session.get(endpoint) as response:
                        if response.status != 200:
                            continue
                            
                        data = await response.json()
                        if not data or 'pairs' not in data:
                            continue
                            
                        pairs = data['pairs']
                        solana_pairs = [p for p in pairs if p.get('chainId') == 'solana']
                        
                        for pair in solana_pairs:
                            volume = float(pair.get('volume', {}).get('h24', 0) or 0)
                            liquidity = float(pair.get('liquidity', {}).get('usd', 0) or 0)
                            
                            if volume >= min_volume and liquidity >= min_liquidity:
                                pairs_data.append({
                                    'pair_address': pair.get('pairAddress'),
                                    'dex': pair.get('dexId'),
                                    'token_address': pair.get('baseToken', {}).get('address'),
                                    'symbol': pair.get('baseToken', {}).get('symbol'),
                                    'name': pair.get('baseToken', {}).get('name'),
                                    'price': float(pair.get('priceUsd', 0)),
                                    'volume_24h': volume,
                                    'liquidity': liquidity,
                                    'price_change_24h': float(
                                        pair.get('priceChange', {}).get('h24', 0) or 0
                                    ),
                                    'txns_24h': int(pair.get('txns', {}).get('h24', 0) or 0)
                                })
                    
                    await asyncio.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    self.logger.error(f"Error processing term '{term}': {e}")
                    continue
            
            return pd.DataFrame(pairs_data)
            
        except Exception as e:
            self.logger.error(f"Error in market snapshot: {e}")
            return pd.DataFrame()

    def clear_cache(self):
        """Clear data cache"""
        self.cache.clear()

async def test_fetcher():
    """Test the data fetcher"""
    fetcher = DataFetcher()
    
    try:
        # Get market snapshot
        df = await fetcher.get_market_snapshot()
        print(f"\nFound {len(df)} pairs")
        
        if not df.empty:
            # Test historical data for first token
            token = df.iloc[0]
            hist_data = await fetcher.get_historical_data(token['token_address'])
            print(f"\nHistorical data shape: {hist_data.shape}")
            
            # Test metrics
            metrics = await fetcher.get_token_metrics(token['token_address'])
            print(f"\nToken metrics: {metrics}")
            
    except Exception as e:
        print(f"Error in test: {e}")
        
    finally:
        await fetcher.close()

if __name__ == "__main__":
    asyncio.run(test_fetcher())