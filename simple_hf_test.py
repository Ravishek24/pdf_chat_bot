#!/usr/bin/env python3
"""
Simple test to see what's happening with your HF token
"""

import os
from dotenv import load_dotenv
import requests
from huggingface_hub import InferenceClient

def test_everything():
    load_dotenv()
    
    token = os.getenv("HF_TOKEN")
    print(f"Token from .env: {token[:10] if token else 'None'}...")
    
    if not token:
        print("❌ No token in .env file")
        return
    
    # Test 1: Direct requests
    print("\n🧪 Test 1: Direct API call")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get("https://huggingface.co/api/models?limit=1", headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Direct API works!")
        else:
            print(f"❌ Direct API failed: {response.text}")
    except Exception as e:
        print(f"❌ Direct API error: {e}")
    
    # Test 2: HuggingFace Hub client
    print("\n🧪 Test 2: HF Hub client")
    try:
        client = InferenceClient(token=token)
        print("✅ Client created successfully")
        
        # Try a simple inference
        result = client.text_generation("Hello", model="gpt2", max_new_tokens=5)
        print(f"✅ Inference works: {result}")
        
    except Exception as e:
        print(f"❌ HF Hub error: {e}")
    
    # Test 3: Try with your model
    print(f"\n🧪 Test 3: Your specific model")
    try:
        client = InferenceClient(model="google/flan-t5-large", token=token)
        result = client.text_generation("What is AI?", max_new_tokens=20)
        print(f"✅ Your model works: {result}")
    except Exception as e:
        print(f"❌ Your model error: {e}")

if __name__ == "__main__":
    test_everything()