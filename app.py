import streamlit as st
import os
from dotenv import load_dotenv
from src.pdf_processor import PDFProcessor
from src.embeddings import EmbeddingManager

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF Chatbot - Local AI",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'llm_type' not in st.session_state:
    st.session_state.llm_type = "Unknown"

# Initialize components
@st.cache_resource
def load_components():
    pdf_processor = PDFProcessor(
        chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200))
    )
    embedding_manager = EmbeddingManager(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    
    # Try to load local transformer LLM
    llm_client = None
    llm_type = "Unknown"
    
    try:
        # Check if transformers is available
        import transformers
        from src.llm_client import LocalTransformerLLM, SimpleLLM
        
        with st.spinner("🚀 Loading local AI model..."):
            try:
                llm_client = LocalTransformerLLM()
                
                if llm_client.pipeline is not None and llm_client.model_type is not None:
                    llm_type = f"Local AI ({llm_client.model_name})"
                    st.success(f"✅ Local AI model loaded successfully!")
                else:
                    llm_type = "Enhanced Fallback"
                    llm_client = SimpleLLM()
                    st.info("ℹ️ Local AI failed to initialize, using enhanced fallback")
                    
            except Exception as e:
                st.warning(f"⚠️ Local AI initialization failed: {str(e)}")
                llm_type = "Enhanced Fallback"
                llm_client = SimpleLLM()
                
    except ImportError:
        st.warning("⚠️ Transformers library not available")
        from src.llm_client import SimpleLLM
        llm_type = "Simple Fallback (No Transformers)"
        llm_client = SimpleLLM()
    except Exception as e:
        st.warning(f"⚠️ Unexpected error: {str(e)}")
        from src.llm_client import SimpleLLM
        llm_type = "Simple Fallback (Error)"
        llm_client = SimpleLLM()
    
    return pdf_processor, embedding_manager, llm_client, llm_type

def main():
    st.title("📄 PDF Chatbot - Local AI")
    st.markdown("Upload a PDF document and ask questions using **local AI models**!")
    
    # Load components
    pdf_processor, embedding_manager, llm_client, llm_type = load_components()
    
    # Display LLM status
    if "Local AI" in llm_type:
        st.success(f"🤖 {llm_type}")
        st.info("💡 Your data stays completely private - no external API calls!")
    else:
        st.info(f"🔄 {llm_type}")
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📁 Upload Document")
        
        # System info
        with st.expander("🖥️ System Info"):
            try:
                import torch
                st.write(f"**Device**: {'GPU' if torch.cuda.is_available() else 'CPU'}")
                if torch.cuda.is_available():
                    st.write(f"**GPU**: {torch.cuda.get_device_name()}")
                    st.write(f"**Memory**: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
            except ImportError:
                st.write("**Device**: CPU (PyTorch not available)")
            
            st.write(f"**AI Model**: {llm_type}")
        
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
                        st.session_state.llm_type = llm_type
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
                    # Try to provide a fallback answer
                    try:
                        fallback_answer = "I encountered an error processing your question. Let me try a different approach..."
                        if relevant_docs:
                            fallback_answer += f"\n\nBased on the document content I found:\n\n{relevant_docs[0].page_content[:200]}..."
                        st.session_state.chat_history.append((question, fallback_answer))
                        st.rerun()
                    except:
                        st.error("Unable to provide any answer. Please try again.")
    
    else:
        st.info("👈 Please upload and process a PDF document to start chatting!")
        
        # Show example
        with st.expander("ℹ️ How to use"):
            st.markdown("""
            1. **Upload PDF**: Use the sidebar to upload your PDF document
            2. **Process**: Click "Process Document" to prepare it for chat
            3. **Ask Questions**: Type questions about your document content
            4. **Get Answers**: The AI will answer based on the document content
            
            **Note**: The app will automatically use the best available AI model. If local models fail, it will use intelligent fallback systems.
            """)
        
        # Show LLM status
        with st.expander("🔧 System Status"):
            st.markdown(f"""
            - **LLM Type**: {llm_type}
            - **Transformers**: {'✅ Available' if 'Local AI' in llm_type else '❌ Not Available'}
            - **Status**: {'🟢 Ready' if 'Local AI' in llm_type else '🟡 Using Fallback'}
            """)

if __name__ == "__main__":
    main()