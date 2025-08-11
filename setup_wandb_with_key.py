#!/usr/bin/env python3
"""
Setup wandb with provided API key
"""

import os
import subprocess
import sys

def setup_wandb_with_key():
    """Setup wandb with the provided API key"""
    
    # Your API key
    api_key = "acff3ce9376b1acc92a538394739adc9a19b1c99"
    
    print("🚀 Setting up Weights & Biases...")
    
    # Install wandb if not already installed
    try:
        import wandb
        print("✅ wandb is already installed")
    except ImportError:
        print("📦 Installing wandb...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
        import wandb
        print("✅ wandb installed successfully")
    
    # Set API key as environment variable
    os.environ['WANDB_API_KEY'] = api_key
    
    # Login to wandb
    try:
        wandb.login(key=api_key)
        print("✅ wandb login successful!")
        
        # Test connection
        test_run = wandb.init(project="test-connection", name="setup-test", mode="online")
        wandb.log({"test_metric": 1.0})
        wandb.finish()
        print("✅ wandb connection test successful!")
        
        return True
        
    except Exception as e:
        print(f"❌ wandb setup failed: {e}")
        return False

def create_wandb_config():
    """Create wandb configuration file"""
    config_content = f"""# wandb configuration
export WANDB_API_KEY=acff3ce9376b1acc92a538394739adc9a19b1c99
export WANDB_PROJECT=trading-bot-optimization
export WANDB_ENTITY=your-username  # Replace with your wandb username if needed
"""
    
    with open('.env', 'w') as f:
        f.write(config_content)
    
    print("✅ Created .env file with wandb configuration")

def main():
    """Main setup function"""
    print("🎯 Setting up wandb for Trading Bot Training")
    print("=" * 50)
    
    if setup_wandb_with_key():
        create_wandb_config()
        
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next Steps:")
        print("1. Run 'python wandb_example.py' to test the integration")
        print("2. Run 'python run_hyperparameter_sweep.py' for optimization")
        print("3. Check your dashboard at https://wandb.ai")
        
        return True
    else:
        print("\n❌ Setup failed. Please check your internet connection and try again.")
        return False

if __name__ == "__main__":
    main()