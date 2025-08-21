#!/usr/bin/env python3
"""
Quick script to test your HF token before adding to .env
"""

import requests
import json

def test_token_quickly():
    print("🔑 Quick HF Token Test")
    print("=" * 30)
    
    # Get token from user
    token = input("Paste your HF token here: ").strip()
    
    if not token:
        print("❌ No token provided")
        return False
    
    if not token.startswith("hf_"):
        print("❌ Token should start with 'hf_'")
        print(f"Your token starts with: {token[:5]}...")
        return False
    
    print(f"✅ Token format looks good: {token[:10]}...{token[-4:]}")
    print(f"✅ Token length: {len(token)} characters")
    
    # Test the token
    print("\n🧪 Testing token with HF API...")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        # Test 1: Check if token is valid
        response = requests.get("https://huggingface.co/api/whoami", headers=headers, timeout=10)
        
        if response.status_code == 200:
            user_info = response.json()
            print(f"✅ Token is VALID! User: {user_info.get('name', 'Unknown')}")
            
            # Test 2: Test inference API
            print("\n🤖 Testing inference API...")
            
            test_data = {
                "inputs": "Hello world",
                "parameters": {"max_new_tokens": 10}
            }
            
            inference_response = requests.post(
                "https://api-inference.huggingface.co/models/gpt2",
                headers=headers,
                json=test_data,
                timeout=30
            )
            
            if inference_response.status_code == 200:
                print("✅ Inference API working!")
                result = inference_response.json()
                print(f"Test response: {result}")
                
                print(f"\n🎉 SUCCESS! Your token works perfectly!")
                print(f"\n📝 Add this to your .env file:")
                print(f"HF_TOKEN={token}")
                return True
                
            elif inference_response.status_code == 503:
                print("⚠️ Model is loading, but token works! This is normal.")
                print(f"\n✅ Your token is working!")
                print(f"\n📝 Add this to your .env file:")
                print(f"HF_TOKEN={token}")
                return True
                
            else:
                print(f"⚠️ Inference API issue: {inference_response.status_code}")
                print(f"Response: {inference_response.text}")
                print("But your token authentication works!")
                return True
                
        elif response.status_code == 401:
            print("❌ Token is INVALID or EXPIRED")
            print("Create a new token at: https://huggingface.co/settings/tokens")
            return False
            
        else:
            print(f"❌ Unexpected error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⚠️ Request timed out - might be network issue")
        print("Your token might be fine, try again later")
        return False
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_token_quickly()