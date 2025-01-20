# Setup Instructions

## Prerequisites
1. Python 3.8 or higher
2. Google Chrome browser
3. pip package manager

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/CoomSacc69/solana-trading-bot.git
cd solana-trading-bot
```

### 2. Install Dependencies
The modern ta-lib-python package handles the installation of TA-Lib automatically, so you can simply run:
```bash
pip install -r requirements.txt
```

If you encounter any issues with ta-lib-python, you can install it separately first:
```bash
pip install --upgrade ta-lib-python
```

### 3. Install Chrome Extension
1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" in the top right
3. Click "Load unpacked" button
4. Select the `chrome-extension` directory from the cloned repository

## Configuration

### 1. Trading Parameters
Edit `bot/config.py` to set your trading parameters:
```python
TRADING_PARAMS = {
    'position_size_usd': 1000,    # Position size in USD
    'max_positions': 3,           # Maximum concurrent positions
    'min_volume_usd': 500000,     # Minimum 24h volume
    'min_liquidity_usd': 100000,  # Minimum liquidity
    'take_profit_pct': 100,       # Take profit percentage
    'stop_loss_pct': 15,          # Stop loss percentage
}
```

### 2. AI Analysis Settings
Configure AI parameters in `analysis/config.py`:
```python
AI_PARAMS = {
    'min_confidence': 0.7,        # Minimum AI prediction confidence
    'technical_weight': 0.6,      # Weight for technical analysis
    'ai_weight': 0.4,            # Weight for AI predictions
}
```

## Usage

### 1. Start the Trading Bot
```bash
python bot/trading_bot.py
```

### 2. Use the Chrome Extension
1. Click the extension icon in Chrome
2. Use the dashboard to:
   - Monitor active positions
   - View market analysis
   - Track trade history
   - Adjust settings

### 3. Monitor Logs
The bot creates detailed logs in the `logs` directory:
- `trading.log`: Trading activities
- `analysis.log`: Analysis results
- `error.log`: Error messages

## Troubleshooting

### Common Issues

1. Package Installation Errors
```bash
# If you encounter any issues, try:
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir
```

2. Chrome Extension Not Loading
- Ensure all files are in the correct directory structure
- Check Chrome's developer console for errors
- Try reloading the extension

3. API Rate Limits
The bot implements rate limiting but you might need to adjust the delays:
```python
# In bot/trading_bot.py
self.parameters.update({
    'api_delay': 1.0,  # Delay between API calls
})
```

## Safety Notes

1. Start with small position sizes for testing
2. Monitor the bot regularly
3. Use stop losses
4. Be aware of market risks
5. Keep API keys secure

## Updates and Maintenance

### Updating the Bot
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

### Backing Up Data
The bot stores data in:
- `data/`: Market data
- `models/`: Trained AI models
- `logs/`: Activity logs

Back these up regularly to preserve your trading history and AI training.

## Support and Resources

- Check the [GitHub Issues](https://github.com/CoomSacc69/solana-trading-bot/issues) for known problems
- Review the [Wiki](https://github.com/CoomSacc69/solana-trading-bot/wiki) for detailed documentation
- Join our [Discord](https://discord.gg/yourserver) for community support