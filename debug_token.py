#!/usr/bin/env python3
"""
Debug script to troubleshoot Hugging Face token issues
"""

import os
from dotenv import load_dotenv
import requests

def test_hf_token():
    """Test Hugging Face token functionality"""
    print("🔍 Debugging Hugging Face Token...")
    
    # Load environment variables
    load_dotenv()
    
    # Check if token exists
    token = os.getenv("HF_TOKEN")
    if not token:
        print("❌ HF_TOKEN not found in environment variables")
        print("💡 Make sure you have a .env file with HF_TOKEN=your_token")
        return False
    
    print(f"✅ Token found: {token[:10]}...{token[-4:]}")
    
    # Test token with HF API
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # Test 1: Check user info
    print("\n🧪 Test 1: Checking user info...")
    try:
        response = requests.get("https://huggingface.co/api/whoami", headers=headers)
        if response.status_code == 200:
            user_info = response.json()
            print(f"✅ User authenticated: {user_info.get('name', 'Unknown')}")
        else:
            print(f"❌ Authentication failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error checking user info: {str(e)}")
        return False
    
    # Test 2: Test inference API
    print("\n🧪 Test 2: Testing inference API...")
    try:
        test_payload = {
            "inputs": "Hello, how are you?",
            "parameters": {
                "max_new_tokens": 50,
                "temperature": 0.7
            }
        }
        
        response = requests.post(
            "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium",
            headers=headers,
            json=test_payload
        )
        
        if response.status_code == 200:
            print("✅ Inference API working!")
            result = response.json()
            print(f"Response: {result}")
        else:
            print(f"❌ Inference API failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing inference API: {str(e)}")
        return False
    
    print("\n🎉 All tests passed! Your token is working correctly.")
    return True

def check_env_file():
    """Check if .env file exists and has correct format"""
    print("\n📁 Checking .env file...")
    
    if not os.path.exists(".env"):
        print("❌ .env file not found!")
        print("💡 Create a .env file with your HF_TOKEN")
        return False
    
    print("✅ .env file found")
    
    with open(".env", "r") as f:
        content = f.read()
        lines = content.split("\n")
        
        for line in lines:
            if line.strip() and not line.startswith("#"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    if key.strip() == "HF_TOKEN":
                        token = value.strip()
                        if token and token != "your_huggingface_token_here":
                            print(f"✅ HF_TOKEN found: {token[:10]}...{token[-4:]}")
                            return True
                        else:
                            print("❌ HF_TOKEN is placeholder value")
                            return False
    
    print("❌ HF_TOKEN not found in .env file")
    return False

def create_env_file():
    """Create a proper .env file"""
    print("\n📝 Creating .env file...")
    
    if os.path.exists(".env"):
        print("⚠️ .env file already exists. Backing up...")
        os.rename(".env", ".env.backup")
    
    token = input("🔑 Enter your Hugging Face token: ").strip()
    
    if not token:
        print("❌ No token provided")
        return False
    
    env_content = f"""# Hugging Face API Configuration
HF_TOKEN={token}

# LLM Model Configuration
LLM_MODEL=google/flan-t5-large

# Embedding Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# PDF Processing Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ .env file created successfully!")
    return True

if __name__ == "__main__":
    print("🚀 Hugging Face Token Debug Tool")
    print("=" * 40)
    
    # Check environment file
    env_ok = check_env_file()
    
    if not env_ok:
        print("\n💡 Let's create a proper .env file...")
        if create_env_file():
            env_ok = True
        else:
            print("❌ Failed to create .env file")
            exit(1)
    
    # Test token
    if env_ok:
        test_hf_token()
    
    print("\n🔧 Troubleshooting Tips:")
    print("1. Make sure your token is from: https://huggingface.co/settings/tokens")
    print("2. Token should have 'read' permissions")
    print("3. Check if token is expired or revoked")
    print("4. Ensure no extra spaces or characters in .env file")
