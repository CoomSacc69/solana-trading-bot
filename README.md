# Solana Trading Bot

An AI-powered Solana trading bot with Chrome extension interface and real-time market analysis.

## Features

- Real-time market monitoring
- AI-powered technical analysis
- Customizable trading parameters
- Chrome extension interface
- Pattern recognition
- Position management

## Structure

- `bot/` - Core trading bot implementation
- `chrome-extension/` - Chrome extension files
- `analysis/` - Technical analysis modules

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Load the Chrome extension:
   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked" and select the `chrome-extension` directory

## Usage

### Running the Bot
```python
from bot.trading_bot import SolanaDexBot

bot = SolanaDexBot()
bot.run()
```

### Configuration
Adjust settings through the Chrome extension interface:
- Trading parameters
- Market analysis thresholds
- Price movement settings
- AI analysis settings
- Update intervals

## Requirements
See `requirements.txt` for a complete list of dependencies.