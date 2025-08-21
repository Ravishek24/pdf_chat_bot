#!/usr/bin/env python3
"""
Debug network connectivity and HF API access
"""

import requests
import socket
import json

def test_basic_connectivity():
    """Test basic internet and DNS"""
    print("🌐 Testing Basic Connectivity")
    print("-" * 30)
    
    # Test 1: Basic internet
    try:
        response = requests.get("https://httpbin.org/ip", timeout=10)
        print(f"✅ Internet: Working (IP: {response.json().get('origin', 'Unknown')})")
    except Exception as e:
        print(f"❌ Internet: Failed - {e}")
        return False
    
    # Test 2: DNS resolution for HF
    try:
        ip = socket.gethostbyname("huggingface.co")
        print(f"✅ HF DNS: Working (IP: {ip})")
    except Exception as e:
        print(f"❌ HF DNS: Failed - {e}")
        return False
    
    # Test 3: HF website accessibility
    try:
        response = requests.get("https://huggingface.co", timeout=10)
        print(f"✅ HF Website: Accessible (Status: {response.status_code})")
    except Exception as e:
        print(f"❌ HF Website: Failed - {e}")
        return False
    
    return True

def test_hf_api_endpoints():
    """Test HF API endpoints without authentication"""
    print("\n🔗 Testing HF API Endpoints")
    print("-" * 30)
    
    endpoints = [
        "https://huggingface.co/api/models",
        "https://api-inference.huggingface.co/",
        "https://huggingface.co/api/whoami"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=10)
            print(f"✅ {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"❌ {endpoint}: Failed - {e}")

def test_token_with_different_methods(token):
    """Test token using different approaches"""
    print(f"\n🔑 Testing Token: {token[:10]}...{token[-4:]}")
    print("-" * 40)
    
    # Method 1: Standard headers
    print("Method 1: Standard Authorization header")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get("https://huggingface.co/api/whoami", headers=headers, timeout=15)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Success: {response.json()}")
            return True
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Method 2: Different endpoint
    print("\nMethod 2: Different endpoint")
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get("https://huggingface.co/api/models?limit=1", headers=headers, timeout=15)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Token works with models endpoint!")
            return True
    except Exception as e:
        print(f"Error: {e}")
    
    # Method 3: Inference API
    print("\nMethod 3: Inference API")
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        data = {"inputs": "test"}
        response = requests.post(
            "https://api-inference.huggingface.co/models/gpt2",
            headers=headers,
            json=data,
            timeout=20
        )
        print(f"Status: {response.status_code}")
        if response.status_code in [200, 503]:  # 503 = model loading
            print("✅ Token works with inference API!")
            return True
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
    
    return False

def check_account_status():
    """Check for common account issues"""
    print("\n👤 Account Status Checks")
    print("-" * 25)
    
    print("Please manually verify:")
    print("1. ✅ Your HF account is verified (check email)")
    print("2. ✅ You're logged into the correct account")
    print("3. ✅ No account restrictions or suspensions")
    print("4. ✅ Token was created in the last few minutes")
    print("5. ✅ You're in a supported region")

def main():
    print("🚨 HF Token Troubleshooting Tool")
    print("=" * 40)
    
    # Step 1: Test connectivity
    if not test_basic_connectivity():
        print("\n❌ Basic connectivity failed. Check your internet connection.")
        return
    
    # Step 2: Test HF API
    test_hf_api_endpoints()
    
    # Step 3: Test token
    token = input("\n🔑 Enter your HF token to test: ").strip()
    
    if not token:
        print("No token provided")
        return
    
    if not token.startswith("hf_"):
        print("❌ Token should start with 'hf_'")
        return
    
    if len(token) != 37:
        print(f"❌ Token should be 37 characters, yours is {len(token)}")
        return
    
    # Test token with multiple methods
    if test_token_with_different_methods(token):
        print("\n🎉 SUCCESS! Your token works!")
    else:
        print("\n❌ All token tests failed")
        check_account_status()
        
        print("\n🔧 Troubleshooting Steps:")
        print("1. Try creating token from a different browser")
        print("2. Try incognito/private browsing mode") 
        print("3. Clear browser cache and cookies")
        print("4. Try from a different network (mobile hotspot)")
        print("5. Wait 10-15 minutes and try again")
        print("6. Contact HF support if issue persists")

if __name__ == "__main__":
    main()