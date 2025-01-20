#!/usr/bin/env python
import sys
import json
import struct
import subprocess
import os
from pathlib import Path
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def send_message(message):
    """Send message to Chrome extension"""
    message_json = json.dumps(message)
    message_bytes = message_json.encode('utf-8')
    sys.stdout.buffer.write(struct.pack('I', len(message_bytes)))
    sys.stdout.buffer.write(message_bytes)
    sys.stdout.buffer.flush()

def read_message():
    """Read message from Chrome extension"""
    length_bytes = sys.stdin.buffer.read(4)
    if not length_bytes:
        return None
    length = struct.unpack('I', length_bytes)[0]
    message_bytes = sys.stdin.buffer.read(length)
    return json.loads(message_bytes.decode('utf-8'))

def start_process(script_path):
    """Start a Python script as a subprocess"""
    try:
        return subprocess.Popen([sys.executable, script_path])
    except Exception as e:
        logger.error(f"Error starting {script_path}: {e}")
        return None

def start_bot():
    """Start all bot components"""
    try:
        # Get root directory
        root_dir = Path(__file__).parent.parent
        
        # Start trading bot
        bot_process = start_process(root_dir / 'bot/trading_bot.py')
        if not bot_process:
            send_message({"status": "error", "error": "Failed to start trading bot"})
            return
            
        # Start WebSocket server
        server_process = start_process(root_dir / 'server/websocket_server.py')
        if not server_process:
            bot_process.terminate()
            send_message({"status": "error", "error": "Failed to start WebSocket server"})
            return
            
        # Monitor processes
        def monitor_processes():
            while True:
                if bot_process.poll() is not None or server_process.poll() is not None:
                    send_message({
                        "status": "error",
                        "error": "A bot component has stopped unexpectedly"
                    })
                    # Clean up
                    if bot_process.poll() is None:
                        bot_process.terminate()
                    if server_process.poll() is None:
                        server_process.terminate()
                    break
                    
        # Start monitoring in a separate thread
        monitor_thread = threading.Thread(target=monitor_processes)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Notify success
        send_message({"status": "running"})
        
    except Exception as e:
        send_message({"status": "error", "error": str(e)})

def main():
    """Main entry point"""
    try:
        message = read_message()
        if message and message.get('action') == 'start':
            start_bot()
        else:
            send_message({"status": "error", "error": "Invalid message"})
    except Exception as e:
        logger.error(f"Error in main: {e}")
        send_message({"status": "error", "error": str(e)})

if __name__ == '__main__':
    main()