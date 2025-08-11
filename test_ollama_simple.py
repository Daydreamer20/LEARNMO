#!/usr/bin/env python3
"""
Simple Ollama test
"""

import requests
import json

def test_ollama():
    """Test Ollama connection"""
    print("Testing Ollama...")
    
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "gpt-oss",
            "prompt": "Respond with just 'WORKING' if you can read this.",
            "stream": False
        }
        
        print("Sending request to Ollama...")
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Ollama response: {result.get('response', 'No response')}")
            return True
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_ollama()