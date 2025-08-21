import streamlit as st
import os
from dotenv import load_dotenv
from src.pdf_processor import PDFProcessor
from src.embeddings import EmbeddingManager

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF Chatbot",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Simple fallback LLM
class SimpleLLM:
    def answer_from_documents(self, documents, question):
        if not documents:
            return "No relevant information found in the document."
        
        context = "\n".join([doc.page_content for doc in documents])
        
        # Simple keyword matching
        import re
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
            return f"Based on the document: {'. '.join(top_sentences)}"
        else:
            return f"I found relevant content but couldn't extract a specific answer. Here's what I found:\n\n{context[:300]}..."

@st.cache_resource
def load_components():
    pdf_processor = PDFProcessor(
        chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200))
    )
    embedding_manager = EmbeddingManager(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    
    # Try to load HF LLM, fallback to simple LLM
    try:
        from src.llm_client import HuggingFaceLLM
        llm_client = HuggingFaceLLM(
            model_name=os.getenv("LLM_MODEL", "microsoft/DialoGPT-medium"),
            token=os.getenv("HF_TOKEN")
        )
        st.success("✅ Using Hugging Face LLM")
    except Exception as e:
        st.warning(f"⚠️ HF LLM failed, using simple fallback: {e}")
        llm_client = SimpleLLM()
    
    return pdf_processor, embedding_manager, llm_client

def main():
    st.title("📄 PDF Chatbot")
    st.markdown("Upload a PDF document and ask questions about its content!")
    
    # Load components
    pdf_processor, embedding_manager, llm_client = load_components()
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📁 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to chat with"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing PDF..."):
                    try:
                        # Process PDF
                        documents = pdf_processor.process_pdf(uploaded_file)
                        
                        # Build vector store
                        embedding_manager.build_vector_store(documents)
                        
                        st.session_state.vector_store_ready = True
                        st.session_state.embedding_manager = embedding_manager
                        st.session_state.llm_client = llm_client
                        st.session_state.chat_history = []
                        
                        st.success(f"✅ Document processed! Created {len(documents)} chunks.")
                    
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
    
    # Main chat interface
    if st.session_state.vector_store_ready:
        st.header("💬 Chat with your document")
        
        # Display chat history
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**You:** {question}")
                st.markdown(f"**Assistant:** {answer}")
                st.divider()
        
        # Question input
        question = st.text_input(
            "Ask a question about your document:",
            placeholder="What is this document about?",
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("Ask Question", type="primary")
        with col2:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        
        if ask_button and question:
            with st.spinner("Thinking..."):
                try:
                    # Retrieve relevant documents
                    relevant_docs = st.session_state.embedding_manager.similarity_search(
                        question, k=3
                    )
                    
                    # Generate answer
                    answer = st.session_state.llm_client.answer_from_documents(
                        relevant_docs, question
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append((question, answer))
                    
                    # Clear input and rerun
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
    
    else:
        st.info("👈 Please upload and process a PDF document to start chatting!")
        
        # Show example
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            1. **Upload PDF**: Use the sidebar to upload your PDF document
            2. **Process**: Click "Process Document" to prepare it for chat
            3. **Ask Questions**: Type questions about your document content
            4. **Get Answers**: The AI will answer based on the document content
            
            **Note**: If Hugging Face API is unavailable, the app will use a simple fallback system.
            """)

if __name__ == "__main__":
    main()