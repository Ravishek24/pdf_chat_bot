#!/usr/bin/env python3
"""
Test script to verify the improved fallback system
"""

from src.llm_client import SimpleLLM
from langchain.docstore.document import Document

def test_improved_fallback():
    """Test the improved fallback system"""
    print("🧪 Testing Improved Fallback System...")
    
    # Create test documents
    test_docs = [
        Document(
            page_content="This document discusses OTP gateway integration for payment systems. The OTP gateway provides secure authentication for online transactions. It supports multiple authentication methods including SMS, email, and push notifications.",
            metadata={"chunk_id": 0}
        ),
        Document(
            page_content="For Node.js integration, you need to implement the OTP verification logic using the gateway's API endpoints. The integration process involves setting up webhooks, handling callbacks, and managing user sessions.",
            metadata={"chunk_id": 1}
        ),
        Document(
            page_content="The gateway supports multiple authentication methods including SMS, email, and push notifications. Each method has its own configuration and security requirements.",
            metadata={"chunk_id": 2}
        )
    ]
    
    # Test questions
    test_questions = [
        "What is this document about?",
        "How can we integrate OTP gateway in Node.js?",
        "What authentication methods are supported?",
        "Tell me about payment systems",
        "Explain the integration process"
    ]
    
    llm = SimpleLLM()
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        answer = llm.answer_from_documents(test_docs, question)
        print(f"🤖 Answer: {answer}")
        print("-" * 50)
    
    print("\n✅ Improved fallback test completed successfully!")

if __name__ == "__main__":
    test_improved_fallback()
