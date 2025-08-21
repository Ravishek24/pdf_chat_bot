from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, pipeline
import torch
import os
from typing import List
from langchain.docstore.document import Document
import streamlit as st
import re
import gc

class LocalTransformerLLM:
    def __init__(self, model_name: str = None):
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        st.info(f"🔧 Initializing local AI model on {self.device.upper()}")
        
        # Try to load a working local model
        self._load_best_available_model()
    
    def _load_best_available_model(self):
        """Try different models until one loads successfully"""
        
        # List of models to try (ordered by preference and size)
        models_to_try = [
            # Small, fast models for CPU
            ("microsoft/DialoGPT-small", "causal", "Small dialogue model (117M)"),
            ("distilgpt2", "causal", "Distilled GPT-2 (82M)"),
            ("gpt2", "causal", "GPT-2 base (124M)"),
            
            # Text-to-text models (often better for Q&A)
            ("google/flan-t5-small", "seq2seq", "FLAN-T5 Small (80M)"),
            ("t5-small", "seq2seq", "T5 Small (60M)"),
            
            # Slightly larger but still manageable
            ("microsoft/DialoGPT-medium", "causal", "Medium dialogue model (345M)"),
            ("google/flan-t5-base", "seq2seq", "FLAN-T5 Base (250M)"),
        ]
        
        for model_name, model_type, description in models_to_try:
            try:
                st.write(f"📥 Trying to load: {model_name} ({description})")
                
                if self._load_model(model_name, model_type):
                    st.success(f"✅ Successfully loaded: {model_name}")
                    st.success(f"🚀 Local AI ready on {self.device.upper()}!")
                    return True
                else:
                    st.write(f"❌ Failed to load {model_name}")
                    
            except Exception as e:
                st.write(f"❌ Error with {model_name}: {str(e)}")
                continue
        
        st.error("❌ Could not load any local model. Using enhanced fallback.")
        return False
    
    def _load_model(self, model_name: str, model_type: str) -> bool:
        """Load a specific model"""
        try:
            # Clear any existing model from memory
            if self.model is not None:
                del self.model
                del self.tokenizer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model based on type
            if model_type == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                )
                task = "text2text-generation"
            else:  # causal
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                )
                task = "text-generation"
            
            # Move to appropriate device
            self.model = self.model.to(self.device)
            
            # Create pipeline
            self.pipeline = pipeline(
                task,
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Test the model
            test_result = self._test_model()
            if test_result:
                self.model_name = model_name
                self.model_type = model_type
                return True
            else:
                return False
                
        except Exception as e:
            st.write(f"Model loading error: {e}")
            return False
    
    def _test_model(self) -> bool:
        """Test if the loaded model works"""
        try:
            if self.model_type == "seq2seq":
                result = self.pipeline("What is AI?", max_length=50, num_return_sequences=1)
            else:
                result = self.pipeline("Hello", max_new_tokens=10, num_return_sequences=1)
            
            return len(result) > 0 and len(result[0].get('generated_text', '')) > 0
            
        except Exception as e:
            st.write(f"Model test failed: {e}")
            return False
    
    def generate_answer(self, context: str, question: str) -> str:
        """Generate answer using local transformer model"""
        if self.pipeline is None:
            return self._fallback_answer(context, question)
        
        try:
            # Create prompt based on model type
            prompt = self._create_prompt(context, question)
            
            with st.spinner("🤖 Generating answer with local AI..."):
                if self.model_type == "seq2seq":
                    # For T5/FLAN models
                    result = self.pipeline(
                        prompt,
                        max_length=200,
                        min_length=20,
                        do_sample=True,
                        temperature=0.7,
                        num_return_sequences=1
                    )
                    answer = result[0]['generated_text']
                else:
                    # For GPT-style models
                    result = self.pipeline(
                        prompt,
                        max_new_tokens=150,
                        do_sample=True,
                        temperature=0.7,
                        repetition_penalty=1.1,
                        num_return_sequences=1,
                        return_full_text=False
                    )
                    answer = result[0]['generated_text']
                
                # Clean and format the answer
                cleaned_answer = self._clean_answer(answer, question)
                
                if len(cleaned_answer.strip()) > 10:
                    st.success("✅ Local AI response generated")
                    return cleaned_answer
                else:
                    st.warning("⚠️ Poor AI response, using enhanced fallback")
                    return self._fallback_answer(context, question)
                    
        except Exception as e:
            st.warning(f"⚠️ Local AI failed: {str(e)}")
            return self._fallback_answer(context, question)
    
    def _create_prompt(self, context: str, question: str) -> str:
        """Create appropriate prompt based on model type"""
        # Truncate context to fit in model's context window
        max_context_length = 400 if self.device == "cpu" else 800
        
        if len(context) > max_context_length:
            # Try to keep the most relevant parts
            context = context[:max_context_length] + "..."
        
        if self.model_type == "seq2seq":
            # T5/FLAN style prompt
            prompt = f"Answer the question based on the context.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            # GPT style prompt
            prompt = f"Based on this document:\n\n{context}\n\nQuestion: {question}\nAnswer:"
        
        return prompt
    
    def _clean_answer(self, answer: str, question: str) -> str:
        """Clean and improve the generated answer"""
        if not answer:
            return "I couldn't generate a proper answer."
        
        # Remove common artifacts
        answer = answer.strip()
        
        # Remove repetitive content
        sentences = answer.split('.')
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence.lower() not in seen and len(sentence) > 5:
                unique_sentences.append(sentence)
                seen.add(sentence.lower())
        
        cleaned = '. '.join(unique_sentences[:3])
        
        # Ensure proper ending
        if cleaned and not cleaned.endswith('.'):
            cleaned += '.'
        
        # Add context if answer is too short
        if len(cleaned) < 30:
            cleaned += "\n\n(Note: This answer is based on the document content provided.)"
        
        return cleaned
    
    def _fallback_answer(self, context: str, question: str) -> str:
        """Enhanced fallback when local model fails"""
        st.info("🔄 Using enhanced keyword matching")
        
        # Smart keyword extraction
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'what', 'when', 'where', 'will', 'with'}
        question_words = {w for w in question_words if len(w) > 2 and w not in stop_words}
        
        # Find relevant sentences
        sentences = re.split(r'[.!?]+', context)
        scored_sentences = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 15:
                sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
                overlap = len(question_words.intersection(sentence_words))
                
                if overlap > 0:
                    # Boost for exact matches
                    for word in question_words:
                        if word in sentence.lower():
                            overlap += 0.5
                    scored_sentences.append((overlap, sentence.strip()))
        
        if scored_sentences:
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            top_sentences = [sent[1] for sent in scored_sentences[:2]]
            
            # Generate contextual response
            if any(kw in question.lower() for kw in ['how', 'implement', 'integrate', 'setup']):
                return f"**Implementation Guide:**\n\n{'. '.join(top_sentences)}\n\n**Next Steps:** Follow the specific implementation steps mentioned in your documentation."
            elif any(kw in question.lower() for kw in ['what', 'explain', 'describe']):
                return f"**Explanation:**\n\n{'. '.join(top_sentences)}"
            else:
                return f"**Based on the document:**\n\n{'. '.join(top_sentences)}"
        else:
            return f"**Document Summary:**\n\n{context[:300]}...\n\n*Please ask a more specific question for better results.*"
    
    def answer_from_documents(self, documents: List[Document], question: str) -> str:
        """Generate answer from retrieved documents"""
        if not documents:
            return "No relevant information found in the document."
        
        # Combine context from relevant documents
        context = "\n\n".join([doc.page_content for doc in documents])
        
        return self.generate_answer(context, question)

class SimpleLLM:
    """Fallback when transformers can't be loaded"""
    
    def __init__(self):
        st.info("🔄 Using Simple Keyword Matching (Transformers not available)")
    
    def answer_from_documents(self, documents: List[Document], question: str) -> str:
        if not documents:
            return "No relevant information found in the document."
        
        context = "\n".join([doc.page_content for doc in documents])
        
        # Use basic keyword matching
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        question_words = {w for w in question_words if len(w) > 3}
        
        sentences = re.split(r'[.!?]+', context)
        scored_sentences = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 15:
                sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
                overlap = len(question_words.intersection(sentence_words))
                if overlap > 0:
                    scored_sentences.append((overlap, sentence.strip()))
        
        if scored_sentences:
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            top_sentences = [sent[1] for sent in scored_sentences[:2]]
            return f"**Based on the document:**\n\n{'. '.join(top_sentences)}"
        else:
            return f"**Document content:**\n\n{context[:300]}..."