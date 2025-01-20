import subprocess
import sys
import webbrowser
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_bot():
    """Start all bot components"""
    try:
        # Start trading bot
        trading_bot = subprocess.Popen([sys.executable, 'bot/trading_bot.py'])
        logger.info("Started trading bot")

        # Start WebSocket server
        websocket_server = subprocess.Popen([sys.executable, 'server/websocket_server.py'])
        logger.info("Started WebSocket server")

        # Open Chrome extensions page
        webbrowser.open('chrome://extensions/')
        logger.info("Opened Chrome extensions page")
        
        print("\nBot started successfully!")
        print("Please ensure the Chrome extension is loaded:")
        print("1. Enable Developer mode in chrome://extensions/")
        print("2. Click 'Load unpacked'")
        print("3. Select the 'chrome-extension' folder")
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down bot...")
            trading_bot.terminate()
            websocket_server.terminate()
            print("\nBot stopped successfully!")

    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        sys.exit(1)

if __name__ == '__main__':
    start_bot()