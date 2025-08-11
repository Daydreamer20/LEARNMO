#!/usr/bin/env python3
"""
Setup script for Weights & Biases integration
"""

import subprocess
import sys
import os

def install_wandb():
    """Install wandb package"""
    try:
        import wandb
        print("✅ wandb is already installed")
        return True
    except ImportError:
        print("📦 Installing wandb...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
            print("✅ wandb installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install wandb")
            return False

def setup_wandb():
    """Setup wandb configuration"""
    try:
        import wandb
        
        print("\n🔧 Setting up Weights & Biases...")
        print("1. Go to https://wandb.ai/authorize to get your API key")
        print("2. Run 'wandb login' in your terminal")
        print("3. Or set your API key as environment variable: WANDB_API_KEY")
        
        # Check if already logged in
        try:
            api_key = wandb.api.api_key
            if api_key:
                print("✅ wandb is already configured")
                return True
        except:
            pass
        
        # Try to login
        print("\nAttempting to login to wandb...")
        try:
            wandb.login()
            print("✅ wandb login successful")
            return True
        except Exception as e:
            print(f"⚠️  wandb login failed: {e}")
            print("You can login manually later with 'wandb login'")
            return False
            
    except ImportError:
        print("❌ wandb not installed")
        return False

def create_wandb_config():
    """Create wandb configuration file"""
    config = {
        "project": "trading-bot-optimization",
        "entity": None,  # Will use default entity
        "settings": {
            "console": "off",  # Disable console logging
            "save_code": True,  # Save code with runs
            "log_code": True   # Log code changes
        }
    }
    
    try:
        import yaml
        with open('.wandb_config.yaml', 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print("✅ Created .wandb_config.yaml")
    except ImportError:
        # Create JSON config instead
        import json
        with open('.wandb_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("✅ Created .wandb_config.json")

def main():
    """Main setup function"""
    print("🚀 Setting up Weights & Biases for Trading Bot Training")
    print("=" * 60)
    
    # Install wandb
    if not install_wandb():
        print("❌ Setup failed: Could not install wandb")
        return False
    
    # Setup wandb
    setup_success = setup_wandb()
    
    # Create config
    create_wandb_config()
    
    print("\n📋 Next Steps:")
    print("1. Run 'python wandb_integration.py' to test the integration")
    print("2. Check your wandb dashboard at https://wandb.ai")
    print("3. Use 'python run_hyperparameter_sweep.py' for parameter optimization")
    
    if setup_success:
        print("\n✅ Setup completed successfully!")
    else:
        print("\n⚠️  Setup completed with warnings. You may need to login manually.")
    
    return True

if __name__ == "__main__":
    main()