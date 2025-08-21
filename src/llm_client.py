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
        self.model_type = None  # Initialize model_type
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
                # Test with a simple Q&A prompt
                test_prompt = "Context: This is a test document about artificial intelligence.\n\nQuestion: What is AI?\n\nAnswer:"
                result = self.pipeline(test_prompt, max_length=50, num_return_sequences=1)
            else:
                # Test with a simple generation prompt
                if "dialo" in self.model_name.lower():
                    # Test prompt for DialoGPT
                    test_prompt = "Human: I have a document that says: This is a test document about artificial intelligence.\n\nBased on this document, can you answer: What is AI?\n\nAssistant: Based on the document, "
                    result = self.pipeline(test_prompt, max_new_tokens=30, num_return_sequences=1)
                else:
                    # Test prompt for other GPT models
                    test_prompt = "Based on this document content:\n\nThis is a test document about artificial intelligence.\n\nQuestion: What is AI?\n\nPlease provide a clear and helpful answer based on the information above:"
                    result = self.pipeline(test_prompt, max_new_tokens=20, num_return_sequences=1)
            
            # Check if we got a meaningful response
            if len(result) > 0:
                generated_text = result[0].get('generated_text', '')
                if len(generated_text.strip()) > 5:
                    st.write(f"✅ Model test successful: {generated_text[:50]}...")
                    return True
                else:
                    st.write(f"❌ Model test failed: Response too short ({len(generated_text)} chars)")
                    return False
            
            return False
            
        except Exception as e:
            st.write(f"Model test failed: {e}")
            return False
    
    def generate_answer(self, context: str, question: str) -> str:
        """Generate answer using local transformer model"""
        if self.pipeline is None or self.model_type is None:
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
                    # For GPT-style models - optimized for DialoGPT
                    if "dialo" in self.model_name.lower():
                        # Special parameters for DialoGPT
                        result = self.pipeline(
                            prompt,
                            max_new_tokens=200,  # Increased from 150
                            min_length=30,        # Added min_length
                            do_sample=True,
                            temperature=0.8,      # Slightly higher temperature
                            repetition_penalty=1.2, # Increased repetition penalty
                            num_return_sequences=1,
                            return_full_text=False,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    else:
                        # Standard parameters for other GPT models
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
                
                # Debug: Show the raw response
                st.write(f"🔍 Raw AI response: {answer[:100]}...")
                st.write(f"🔍 Response length: {len(answer)} characters")
                st.write(f"🔍 Model type: {self.model_type}")
                st.write(f"🔍 Model name: {self.model_name}")
                
                # Clean and format the answer
                cleaned_answer = self._clean_answer(answer, question)
                
                if cleaned_answer is not None:
                    st.success("✅ Local AI response generated")
                    st.write(f"🔍 Cleaned answer length: {len(cleaned_answer)} characters")
                    return cleaned_answer
                else:
                    # Force AI response if it's available, even if short
                    st.warning("⚠️ AI response was short, but using it anyway")
                    if answer and len(answer.strip()) > 3:
                        # Use the short AI response instead of fallback
                        cleaned_short = answer.strip()
                        if not cleaned_short.endswith(('.', '!', '?')):
                            cleaned_short += '.'
                        return f"**AI Response:** {cleaned_short}\n\n*Note: This is a brief AI-generated answer.*"
                    else:
                        st.warning("⚠️ AI response too short, using enhanced fallback")
                        st.write(f"🔍 Answer was rejected because it was too short")
                        return self._fallback_answer(context, question)
                    
        except Exception as e:
            st.error(f"❌ Local AI error: {str(e)}")
            return self._fallback_answer(context, question)
    
    def _create_prompt(self, context: str, question: str) -> str:
        """Create appropriate prompt based on model type"""
        # Truncate context if too long for the model
        max_context_length = 800 if self.device == "cpu" else 1200
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
        
        # Detect task type for better prompting
        question_lower = question.lower()
        if any(word in question_lower for word in ['summarize', 'summary', 'overview']):
            task = "summarize"
        elif any(word in question_lower for word in ['explain', 'describe', 'what is']):
            task = "explain"
        elif any(word in question_lower for word in ['how', 'implement', 'steps']):
            task = "how_to"
        else:
            task = "general"
        
        if self.model_type == "seq2seq":
            # For T5/FLAN models - more structured prompt
            if task == "summarize":
                return f"Summarize the following document content in a clear and concise way:\n\n{context}\n\nSummary:"
            else:
                return f"Based on the following context, answer the question clearly and concisely.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            # For GPT-style models - conversational prompt optimized for DialoGPT
            if "dialo" in self.model_name.lower():
                # Special prompt for DialoGPT models
                if task == "summarize":
                    return f"Human: I have a document that says: {context}\n\nCan you provide a summary of this document?\n\nAssistant: Based on the document, here's a summary: "
                else:
                    return f"Human: I have a document that says: {context}\n\nBased on this document, can you answer: {question}\n\nAssistant: Based on the document, "
            else:
                # Standard prompt for other GPT models
                if task == "summarize":
                    return f"Please provide a summary of the following document content:\n\n{context}\n\nSummary:"
                else:
                    return f"Based on this document content:\n\n{context}\n\nQuestion: {question}\n\nPlease provide a clear and helpful answer based on the information above:"
    
    def _clean_answer(self, answer: str, question: str) -> str:
        """Clean and format the AI-generated answer"""
        if not answer or len(answer.strip()) < 3:  # Reduced from 5 to 3
            return None  # Return None for very short answers
        
        # Remove the prompt if it's included
        if "Context:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        # Remove extra whitespace and newlines
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # Ensure proper sentence ending
        if answer and not answer.endswith(('.', '!', '?')):
            answer += '.'
        
        # If answer is too short, return None to trigger fallback
        if len(answer) < 10:  # Reduced from 20 to 10
            return None
        
        return answer
    
    def _fallback_answer(self, context: str, question: str) -> str:
        """Enhanced fallback when local model fails"""
        st.info("🔄 Using enhanced keyword matching")
        
        # Smart keyword extraction
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'what', 'when', 'where', 'will', 'with'}
        question_words = {w for w in question_words if len(w) > 2 and w not in stop_words}
        
        # If no specific keywords, use general terms
        if not question_words:
            question_words = {'document', 'content', 'information', 'text'}
        
        # Find relevant sentences
        sentences = re.split(r'[.!?]+', context)
        scored_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15:
                sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
                overlap = len(question_words.intersection(sentence_words))
                
                if overlap > 0:
                    # Boost for exact matches
                    for word in question_words:
                        if word in sentence.lower():
                            overlap += 0.5
                    scored_sentences.append((overlap, sentence))
        
        if scored_sentences:
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            top_sentences = [sent[1] for sent in scored_sentences[:3]]
            
            # Generate contextual response based on question type
            if any(kw in question.lower() for kw in ['how', 'implement', 'integrate', 'setup']):
                return f"**Implementation Guide:**\n\n{'. '.join(top_sentences)}\n\n**Next Steps:** Follow the specific implementation steps mentioned in your documentation."
            elif any(kw in question.lower() for kw in ['what', 'explain', 'describe', 'tell me about']):
                return f"**Document Summary:**\n\n{'. '.join(top_sentences)}"
            elif any(kw in question.lower() for kw in ['why', 'reason', 'cause']):
                return f"**Explanation:**\n\n{'. '.join(top_sentences)}"
            else:
                return f"**Based on the document:**\n\n{'. '.join(top_sentences)}"
        else:
            # If no keyword matches, provide a general summary
            # Split context into meaningful chunks
            words = context.split()
            if len(words) > 100:
                # Take first 100 words for summary
                summary = ' '.join(words[:100]) + "..."
            else:
                summary = context
            
            return f"**Document Content Summary:**\n\n{summary}\n\n*This appears to be a document with {len(words)} words. Please ask a more specific question for better results.*"
    
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
            sentence = sentence.strip()
            if len(sentence) > 15:
                sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
                overlap = len(question_words.intersection(sentence_words))
                if overlap > 0:
                    scored_sentences.append((overlap, sentence))
        
        if scored_sentences:
            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            top_sentences = [sent[1] for sent in scored_sentences[:2]]
            return f"**Based on the document:**\n\n{'. '.join(top_sentences)}"
        else:
            return f"**Document content:**\n\n{context[:300]}..."