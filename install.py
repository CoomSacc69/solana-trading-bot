#!/usr/bin/env python
import os
import sys
import json
import shutil
from pathlib import Path
import platform

def install_native_messaging_host():
    """Install native messaging host for Chrome extension"""
    # Get absolute path of bot_launcher.py
    root_dir = Path(__file__).parent
    launcher_path = root_dir / 'native-messaging-host/bot_launcher.py'
    manifest_path = root_dir / 'native-messaging-host/com.solanabot.launcher.json'
    
    # Make launcher executable
    launcher_path.chmod(0o755)
    
    # Load and update manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    manifest['path'] = str(launcher_path)
    
    # Determine OS-specific manifest location
    if platform.system() == 'Windows':
        target_dir = Path(os.environ['LOCALAPPDATA']) / 'Google/Chrome/NativeMessagingHosts'
    elif platform.system() == 'Darwin':  # macOS
        target_dir = Path.home() / 'Library/Application Support/Google/Chrome/NativeMessagingHosts'
    else:  # Linux
        target_dir = Path.home() / '.config/google-chrome/NativeMessagingHosts'
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Write updated manifest
    target_manifest = target_dir / 'com.solanabot.launcher.json'
    with open(target_manifest, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"Native messaging host installed to {target_manifest}")

def install_dependencies():
    """Install Python dependencies"""
    try:
        import pip
        pip.main(['install', '-r', 'requirements.txt'])
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def main():
    """Main installation function"""
    print("Installing Solana Trading Bot...")
    
    # Install Python dependencies
    print("\nInstalling dependencies...")
    install_dependencies()
    
    # Install native messaging host
    print("\nInstalling native messaging host...")
    install_native_messaging_host()
    
    print("\nInstallation complete!")
    print("Please install the Chrome extension and click the 'Quick Start' button to begin.")

if __name__ == '__main__':
    main()