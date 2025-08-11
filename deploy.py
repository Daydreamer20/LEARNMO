#!/usr/bin/env python3
"""
Simple deployment script to prepare files for Railway
"""

import os
import json
import shutil
from datetime import datetime

def create_deployment_package():
    """Create a clean deployment package"""
    
    print("🚀 Preparing Railway deployment package...")
    
    # Create deployment directory
    deploy_dir = "railway_deployment"
    if os.path.exists(deploy_dir):
        shutil.rmtree(deploy_dir)
    os.makedirs(deploy_dir)
    
    # Core files to deploy
    files_to_copy = [
        "railway_trading_bot.py",
        "requirements.txt",
        "Procfile",
        "railway.json",
        "runtime.txt",
        ".env.example"
    ]
    
    # Copy files
    for file in files_to_copy:
        if os.path.exists(file):
            shutil.copy2(file, deploy_dir)
            print(f"✅ Copied {file}")
        else:
            print(f"⚠️ Missing {file}")
    
    # Create README for deployment
    readme_content = f"""# Railway Trading Bot Deployment

## 🚀 Deployed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This is a continuous learning trading bot that:
- Trades SOONUSDT on Bybit
- Optimizes parameters every hour using WANDB
- Runs 24/7 on Railway.com
- Stores all data in SQLite database

## Environment Variables Needed:
- WANDB_API_KEY (optional but recommended)
- INITIAL_BALANCE=1000
- PAPER_TRADING=true

## Health Check:
- Endpoint: /health
- Port: Automatically set by Railway

## Monitoring:
- Railway logs for bot activity
- WANDB dashboard for performance metrics
"""
    
    with open(os.path.join(deploy_dir, "README.md"), "w", encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"✅ Created deployment package in '{deploy_dir}' directory")
    print("\n📋 Next steps:")
    print("1. Go to https://railway.app")
    print("2. Sign in with GitHub")
    print("3. Click 'New Project' → 'Deploy from GitHub repo'")
    print("4. Upload the files from the 'railway_deployment' directory")
    print("5. Set environment variables in Railway dashboard")
    print("6. Deploy and monitor!")
    
    return deploy_dir

if __name__ == "__main__":
    create_deployment_package()