import polars as pl
from pathlib import Path
import asyncio
import socketio
import ccxt.async_support as ccxt
from typing import Dict, List, Optional, Union
from loguru import logger
import aiohttp
from datetime import datetime, timedelta

class DataHandler:
    def __init__(self):
        """Initialize the data handler with modern async capabilities"""
        # Setup logging
        logger.add("logs/data.log", rotation="1 day")
        
        # Initialize connections
        self.sio = socketio.AsyncClient()
        self.session = aiohttp.ClientSession()
        self.ws_connections = {}
        self.data_cache = {}
        
        # Setup ccxt
        self.exchanges = {
            'dexscreener': {
                'ws_url': 'wss://api.dexscreener.com/ws',
                'rest_url': 'https://api.dexscreener.com/latest/dex'
            }
        }
        
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    async def cleanup(self):
        """Cleanup resources"""
        await self.session.close()
        await self.sio.disconnect()
        for ws in self.ws_connections.values():
            await ws.close()

    async def fetch_market_data(self, symbols: List[str]) -> pl.DataFrame:
        """Fetch market data using modern async patterns and Polars"""
        try:
            tasks = []
            for symbol in symbols:
                tasks.append(self.session.get(
                    f"{self.exchanges['dexscreener']['rest_url']}/search",
                    params={'q': symbol}
                ))
            
            responses = await asyncio.gather(*tasks)
            data = []
            
            for response in responses:
                json_data = await response.json()
                if 'pairs' in json_data:
                    data.extend(json_data['pairs'])
            
            if not data:
                return pl.DataFrame()
            
            # Use Polars for faster data processing
            df = pl.DataFrame(data)
            
            if len(df) > 0:
                # Optimize data types
                df = df.with_columns([
                    pl.col('priceUsd').cast(pl.Float64),
                    pl.col('volume.h24').alias('volume_24h').cast(pl.Float64),
                    pl.col('liquidity.usd').alias('liquidity_usd').cast(pl.Float64),
                    pl.col('priceChange.h24').alias('price_change_24h').cast(pl.Float64)
                ])
                
                # Add computed columns
                df = df.with_columns([
                    (pl.col('volume_24h') / pl.col('liquidity_usd')).alias('vol_liq_ratio'),
                    pl.col('timestamp').cast(pl.Datetime)
                ])
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return pl.DataFrame()

    async def setup_websocket(self, symbol: str):
        """Setup websocket connection with automatic reconnection"""
        while True:
            try:
                if symbol not in self.ws_connections or self.ws_connections[symbol].closed:
                    ws = await self.session.ws_connect(self.exchanges['dexscreener']['ws_url'])
                    self.ws_connections[symbol] = ws
                    
                    # Subscribe to updates
                    await ws.send_json({
                        'type': 'subscribe',
                        'symbol': symbol
                    })
                    
                    asyncio.create_task(self._handle_websocket_messages(symbol, ws))
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"WebSocket error for {symbol}: {e}")
                await asyncio.sleep(5)

    async def _handle_websocket_messages(self, symbol: str, ws: aiohttp.ClientWebSocketResponse):
        """Handle incoming websocket messages"""
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = msg.json()
                    await self._process_websocket_data(symbol, data)
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error for {symbol}")
                    break
                    
        except Exception as e:
            logger.error(f"Error handling WebSocket messages for {symbol}: {e}")
        finally:
            if not ws.closed:
                await ws.close()

    async def _process_websocket_data(self, symbol: str, data: dict):
        """Process incoming websocket data"""
        try:
            if 'type' in data:
                if data['type'] == 'trade':
                    # Update trade data cache
                    self.data_cache[symbol] = {
                        'last_price': data['price'],
                        'last_trade_time': datetime.now(),
                        'volume_24h': data.get('volume24h'),
                        'price_change_24h': data.get('priceChange24h')
                    }
                elif data['type'] == 'orderbook':
                    # Update orderbook cache
                    self.data_cache[f"{symbol}_orderbook"] = {
                        'bids': data['bids'],
                        'asks': data['asks'],
                        'timestamp': datetime.now()
                    }
                
                # Clean old cache entries
                self._clean_cache()
                
        except Exception as e:
            logger.error(f"Error processing WebSocket data for {symbol}: {e}")

    def _clean_cache(self):
        """Clean old cache entries"""
        current_time = datetime.now()
        keys_to_remove = []
        
        for key, data in self.data_cache.items():
            if 'last_trade_time' in data:
                if current_time - data['last_trade_time'] > timedelta(minutes=30):
                    keys_to_remove.append(key)
                    
        for key in keys_to_remove:
            del self.data_cache[key]

    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str = '1h', 
        limit: int = 1000
    ) -> pl.DataFrame:
        """Get historical data using ccxt"""
        try:
            # Use ccxt for standardized historical data
            exchange = ccxt.gateio()  # You can change this to any supported exchange
            
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return pl.DataFrame()
            
            # Convert to Polars DataFrame
            df = pl.DataFrame(
                ohlcv,
                schema=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Convert timestamp to datetime
            df = df.with_columns([
                pl.col('timestamp').cast(pl.Int64).map(
                    lambda x: datetime.fromtimestamp(x / 1000)
                ).alias('datetime')
            ])
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pl.DataFrame()
        finally:
            await exchange.close()

    def save_data(self, df: pl.DataFrame, filename: str):
        """Save data efficiently using Parquet format"""
        try:
            path = Path('data') / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as parquet for efficient storage and fast reading
            df.write_parquet(str(path))
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def load_data(self, filename: str) -> Optional[pl.DataFrame]:
        """Load data from Parquet file"""
        try:
            path = Path('data') / filename
            if not path.exists():
                return None
                
            return pl.read_parquet(str(path))
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None