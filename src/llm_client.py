from huggingface_hub import InferenceClient
import os
from typing import List
from langchain.docstore.document import Document
import streamlit as st
import re

class HuggingFaceLLM:
    def __init__(self, model_name: str = "google/flan-t5-large", token: str = None):
        self.model_name = model_name
        self.token = token or os.getenv("HF_TOKEN")
        self.client = None
        
        if self.token:
            try:
                self.client = InferenceClient(
                    model=model_name,
                    token=self.token
                )
            except Exception as e:
                st.warning(f"⚠️ Failed to initialize HF client: {str(e)}")
                self.client = None
    
    def generate_answer(self, context: str, question: str) -> str:
        """Generate answer based on context and question"""
        if not self.client:
            return self._fallback_answer(context, question)
        
        prompt = self._create_prompt(context, question)
        
        try:
            response = self.client.text_generation(
                prompt,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=False,
                return_full_text=False
            )
            return response.strip()
        
        except Exception as e:
            st.warning(f"⚠️ HF API failed, using fallback: {str(e)}")
            return self._fallback_answer(context, question)
    
    def _fallback_answer(self, context: str, question: str) -> str:
        """Fallback answer generation using simple keyword matching"""
        if not context or not question:
            return "I cannot process this request. Please try again."
        
        # Simple keyword matching approach
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        question_words = {w for w in question_words if len(w) > 2}
        
        # Split context into sentences
        sentences = re.split(r'[.!?]+', context)
        scored_sentences = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 10:
                sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
                overlap = len(question_words.intersection(sentence_words))
                if overlap > 0:
                    scored_sentences.append((overlap, sentence.strip()))
        
        if scored_sentences:
            # Sort by relevance score
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            top_sentences = [sent[1] for sent in scored_sentences[:2]]
            
            # Generate a contextual response
            if "otp" in question.lower() or "gateway" in question.lower():
                return f"Based on the document content, here's what I found about OTP gateways:\n\n{'. '.join(top_sentences)}\n\nFor Node.js integration, you would typically need to implement the specific OTP logic based on your requirements."
            else:
                return f"Based on the document content:\n\n{'. '.join(top_sentences)}"
        else:
            # If no direct matches, provide a general response
            return f"I found relevant content in the document. Here's a summary:\n\n{context[:300]}...\n\nPlease ask a more specific question for better results."
    
    def _create_prompt(self, context: str, question: str) -> str:
        """Create a well-structured prompt for the LLM"""
        # Truncate context if too long
        max_context_length = 2000
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        prompt = f"""Answer the question based on the context below. If you cannot find the answer, say "I cannot find this information in the document."

Context: {context}

Question: {question}

Answer:"""
        
        return prompt
    
    def answer_from_documents(self, documents: List[Document], question: str) -> str:
        """Generate answer from retrieved documents"""
        if not documents:
            return "No relevant information found in the document."
        
        # Combine context from relevant documents
        context = "\n\n".join([doc.page_content for doc in documents])
        
        return self.generate_answer(context, question)

class SimpleLLM:
    """Simple fallback LLM that doesn't require external APIs"""
    
    def __init__(self):
        pass
    
    def answer_from_documents(self, documents: List[Document], question: str) -> str:
        """Generate answer using simple keyword matching"""
        if not documents:
            return "No relevant information found in the document."
        
        context = "\n".join([doc.page_content for doc in documents])
        
        # Simple keyword matching
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        question_words = {w for w in question_words if len(w) > 2}
        
        sentences = re.split(r'[.!?]+', context)
        scored_sentences = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 10:
                sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
                overlap = len(question_words.intersection(sentence_words))
                if overlap > 0:
                    scored_sentences.append((overlap, sentence.strip()))
        
        if scored_sentences:
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            top_sentences = [sent[1] for sent in scored_sentences[:2]]
            
            # Special handling for OTP gateway questions
            if "otp" in question.lower() or "gateway" in question.lower():
                return f"Based on the document: {'. '.join(top_sentences)}\n\nFor Node.js integration, you would need to implement the OTP logic based on your specific requirements and the gateway's API documentation."
            else:
                return f"Based on the document: {'. '.join(top_sentences)}"
        else:
            return f"I found relevant content but couldn't extract a specific answer. Here's what I found:\n\n{context[:300]}..."