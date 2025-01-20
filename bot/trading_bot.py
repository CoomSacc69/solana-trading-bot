import asyncio
import aiohttp
from datetime import datetime
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
import requests
from pathlib import Path
import json
import websockets
from analysis.technical_analysis import TechnicalAnalyzer

class SolanaDexBot:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.base_url = "https://api.dexscreener.com/latest/dex"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        })

        self.parameters = {
            'min_volume_usd': 500000,
            'min_liquidity_usd': 100000,
            'min_price_change': 15,
            'max_price_change': 100,
            'min_vol_liq_ratio': 3,
            'position_size_usd': 1000,
            'take_profit_pct': 100,
            'stop_loss_pct': 15
        }
        
        self.active_trades: Dict[str, dict] = {}
        self.trade_history: List[dict] = []
        self.technical_analyzer = TechnicalAnalyzer()
        
        # Create websocket server
        self.ws_server = None
        self.clients = set()

    async def setup_websocket_server(self):
        """Setup WebSocket server for status updates"""
        async def handler(websocket, path):
            self.clients.add(websocket)
            try:
                async for message in websocket:
                    # Handle any incoming messages if needed
                    pass
            finally:
                self.clients.remove(websocket)

        self.ws_server = await websockets.serve(handler, "localhost", 8765)
        self.logger.info("WebSocket server started on ws://localhost:8765")

    async def broadcast_status(self, data: dict):
        """Broadcast status to all connected clients"""
        if self.clients:
            message = json.dumps(data)
            await asyncio.gather(
                *[client.send(message) for client in self.clients]
            )

    async def fetch_market_data(self) -> pd.DataFrame:
        """Fetch market data from DexScreener"""
        try:
            all_pairs = []
            search_terms = ["solana", "SOL", "RAY", "ORCA", "USDC", "USDT", "BONK"]
            
            for term in search_terms:
                try:
                    endpoint = f"{self.base_url}/search?q={term}"
                    async with self.session.get(endpoint) as response:
                        data = await response.json()
                        pairs = data.get('pairs', [])
                        
                        if pairs:
                            solana_pairs = [
                                pair for pair in pairs 
                                if pair.get('chainId') == 'solana'
                            ]
                            all_pairs.extend(solana_pairs)
                            self.logger.info(
                                f"Found {len(solana_pairs)} Solana pairs for '{term}'"
                            )
                    
                    await asyncio.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    self.logger.error(f"Error fetching pairs for term '{term}': {e}")
                    continue
            
            # Process pairs into DataFrame
            processed_pairs = []
            for pair in all_pairs:
                try:
                    base_token = pair.get('baseToken', {})
                    volume_usd = float(pair.get('volume', {}).get('h24', 0) or 0)
                    liquidity_usd = float(pair.get('liquidity', {}).get('usd', 0) or 0)
                    
                    if volume_usd >= self.parameters['min_volume_usd'] and \
                       liquidity_usd >= self.parameters['min_liquidity_usd']:
                        processed_pairs.append({
                            'address': base_token.get('address'),
                            'symbol': base_token.get('symbol'),
                            'name': base_token.get('name'),
                            'price_usd': float(pair.get('priceUsd', 0)),
                            'volume_usd': volume_usd,
                            'liquidity_usd': liquidity_usd,
                            'price_change_24h': float(
                                pair.get('priceChange', {}).get('h24', 0) or 0
                            ),
                            'dex': pair.get('dexId'),
                            'url': f"https://dexscreener.com/solana/{pair.get('pairAddress')}"
                        })
                        
                except Exception as e:
                    continue
            
            return pd.DataFrame(processed_pairs)
            
        except Exception as e:
            self.logger.error(f"Error in market data fetch: {e}")
            return pd.DataFrame()

    async def analyze_market(self, market_data: pd.DataFrame) -> List[dict]:
        """Analyze market data for trading opportunities"""
        try:
            opportunities = []
            
            for _, token in market_data.iterrows():
                # Calculate volume/liquidity ratio
                vol_liq_ratio = (
                    token['volume_usd'] / token['liquidity_usd'] 
                    if token['liquidity_usd'] > 0 else 0
                )
                
                # Check if meets criteria
                if (vol_liq_ratio >= self.parameters['min_vol_liq_ratio'] and
                    self.parameters['min_price_change'] <= 
                    token['price_change_24h'] <= 
                    self.parameters['max_price_change']):
                    
                    # Technical analysis
                    analysis = self.technical_analyzer.analyze_patterns(
                        self._get_historical_data(token['address'])
                    )
                    
                    if analysis.get('signals', {}).get('direction') == 'bullish':
                        opportunities.append({
                            **token.to_dict(),
                            'vol_liq_ratio': vol_liq_ratio,
                            'technical_analysis': analysis,
                            'score': vol_liq_ratio * token['price_change_24h']
                        })
            
            # Sort by score
            opportunities.sort(key=lambda x: x['score'], reverse=True)
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error in market analysis: {e}")
            return []

    def _get_historical_data(self, address: str) -> pd.DataFrame:
        """Get historical price data for technical analysis"""
        # Implement historical data fetching
        # This would normally fetch OHLCV data from DEX
        return pd.DataFrame()  # Placeholder

    async def execute_trade(self, token_data: dict) -> bool:
        """Execute trade with proper risk management"""
        try:
            # Calculate position size
            position_size = min(
                self.parameters['position_size_usd'],
                token_data['liquidity_usd'] * 0.02  # Max 2% of liquidity
            )
            
            entry_price = token_data['price_usd']
            take_profit_price = entry_price * (1 + (self.parameters['take_profit_pct'] / 100))
            stop_loss_price = entry_price * (1 - (self.parameters['stop_loss_pct'] / 100))
            
            # Record trade
            trade_details = {
                'symbol': token_data['symbol'],
                'entry_price': entry_price,
                'position_size': position_size,
                'take_profit': take_profit_price,
                'stop_loss': stop_loss_price,
                'entry_time': datetime.now(),
                'dex': token_data['dex'],
                'technical_analysis': token_data.get('technical_analysis', {})
            }
            
            self.active_trades[token_data['symbol']] = trade_details
            
            # Broadcast update
            await self.broadcast_status({
                'type': 'new_trade',
                'data': trade_details
            })
            
            self.logger.info(f"\nNew trade opened for {token_data['symbol']}:")
            self.logger.info(f"Entry: ${entry_price:.4f}")
            self.logger.info(f"Size: ${position_size:.2f}")
            self.logger.info(f"Take Profit: ${take_profit_price:.4f}")
            self.logger.info(f"Stop Loss: ${stop_loss_price:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return False

    async def monitor_positions(self, market_data: pd.DataFrame):
        """Monitor active positions for take profit or stop loss"""
        try:
            for symbol in list(self.active_trades.keys()):
                current_data = market_data[market_data['symbol'] == symbol]
                
                if current_data.empty:
                    continue
                
                current_price = current_data.iloc[0]['price_usd']
                trade = self.active_trades[symbol]
                
                # Check take profit and stop loss
                if current_price >= trade['take_profit']:
                    await self.close_position(symbol, 'take_profit', current_price)
                elif current_price <= trade['stop_loss']:
                    await self.close_position(symbol, 'stop_loss', current_price)
                
        except Exception as e:
            self.logger.error(f"Error monitoring positions: {e}")

    async def close_position(self, symbol: str, reason: str, current_price: float):
        """Close a position and record results"""
        try:
            trade = self.active_trades[symbol]
            duration = datetime.now() - trade['entry_time']
            profit_loss = (current_price - trade['entry_price']) / trade['entry_price'] * 100
            
            trade_result = {
                **trade,
                'exit_price': current_price,
                'exit_time': datetime.now(),
                'duration': str(duration),
                'profit_loss_pct': profit_loss,
                'close_reason': reason
            }
            
            self.trade_history.append(trade_result)
            
            # Broadcast update
            await self.broadcast_status({
                'type': 'close_trade',
                'data': trade_result
            })
            
            self.logger.info(f"\nPosition closed for {symbol}")
            self.logger.info(f"Reason: {reason}")
            self.logger.info(f"Profit/Loss: {profit_loss:.2f}%")
            self.logger.info(f"Duration: {duration}")
            
            del self.active_trades[symbol]
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")

    async def run(self):
        """Main bot loop"""
        self.logger.info("Starting trading bot...")
        
        # Setup WebSocket server
        await self.setup_websocket_server()
        
        while True:
            try:
                # Fetch and analyze market data
                market_data = await self.fetch_market_data()
                if market_data.empty:
                    await asyncio.sleep(60)
                    continue

                # Monitor existing positions
                await self.monitor_positions(market_data)

                # Find new opportunities
                opportunities = await self.analyze_market(market_data)
                
                # Execute trades for top opportunities
                for opportunity in opportunities[:3]:  # Max 3 concurrent positions
                    if len(self.active_trades) < 3:
                        await self.execute_trade(opportunity)

                # Broadcast status update
                await self.broadcast_status({
                    'type': 'status_update',
                    'data': {
                        'active_trades': list(self.active_trades.values()),
                        'opportunities': opportunities[:5],
                        'timestamp': datetime.now().isoformat()
                    }
                })

                await asyncio.sleep(60)

            except KeyboardInterrupt:
                self.logger.info("\nBot stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)

def main():
    bot = SolanaDexBot()
    asyncio.run(bot.run())

if __name__ == "__main__":
    main()