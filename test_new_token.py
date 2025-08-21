#!/usr/bin/env python3
"""
Simple script to test a new Hugging Face token
"""

import requests

def test_token(token):
    """Test a Hugging Face token"""
    print(f"🧪 Testing token: {token[:10]}...{token[-4:]}")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Test 1: Check user info
    print("Testing user authentication...")
    try:
        response = requests.get("https://huggingface.co/api/whoami", headers=headers)
        if response.status_code == 200:
            user_info = response.json()
            print(f"✅ SUCCESS! User: {user_info.get('name', 'Unknown')}")
            return True
        else:
            print(f"❌ FAILED: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔑 Hugging Face Token Tester")
    print("=" * 30)
    
    # Get token from user
    token = input("Enter your new HF token: ").strip()
    
    if not token:
        print("❌ No token provided")
        exit(1)
    
    if not token.startswith("hf_"):
        print("❌ Token should start with 'hf_'")
        exit(1)
    
    # Test the token
    if test_token(token):
        print("\n🎉 Your token is working! You can now use it in your .env file.")
        print(f"Add this line to your .env file:")
        print(f"HF_TOKEN={token}")
    else:
        print("\n❌ Token test failed. Please check:")
        print("1. Token is copied correctly")
        print("2. Token has 'read' permissions")
        print("3. Token is not expired")
        print("4. You're using the right account")
