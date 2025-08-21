# 📄 PDF Chatbot - Local AI Edition

A powerful Streamlit-based PDF chatbot that allows you to upload PDF documents and ask questions about their content using **local AI models** - no external API calls required!

## ✨ Features

- **PDF Processing**: Upload and process PDF documents with intelligent text chunking
- **Semantic Search**: Find relevant content using advanced embeddings and vector search
- **Local AI-Powered Q&A**: Generate intelligent answers using local transformer models
- **Multiple AI Options**: Local models with intelligent fallback systems
- **Complete Privacy**: Your data never leaves your computer
- **User-Friendly Interface**: Clean Streamlit UI with real-time feedback
- **Persistent Chat**: Maintain conversation history during your session
- **Error Resilience**: Graceful fallbacks when local models fail

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd pdf_project
```

### 2. Install Dependencies

#### Option A: Full Local AI Setup (Recommended)
```bash
pip install -r requirements.txt
```

#### Option B: Minimal Setup (Fallback Only)
```bash
pip install -r requirements-minimal.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🔧 How It Works

### Architecture

1. **PDF Processing** (`PDFProcessor`): Extracts text and creates intelligent chunks
2. **Embedding Management** (`EmbeddingManager`): Creates vector embeddings for semantic search
3. **Local AI Integration** (`LocalTransformerLLM`): Generates AI-powered answers using local models
4. **Fallback Systems**: Intelligent keyword matching when AI models fail

### AI Model Options

The app automatically tries to load local models in this order:

1. **Small Models** (Fast, CPU-friendly):
   - `microsoft/DialoGPT-small` (117M parameters)
   - `distilgpt2` (82M parameters)
   - `gpt2` (124M parameters)

2. **Text-to-Text Models** (Better for Q&A):
   - `google/flan-t5-small` (80M parameters)
   - `t5-small` (60M parameters)

3. **Medium Models** (Better quality, slower):
   - `microsoft/DialoGPT-medium` (345M parameters)
   - `google/flan-t5-base` (250M parameters)

### Workflow

1. **Upload PDF**: Use the sidebar to upload your document
2. **Process Document**: Click "Process Document" to prepare it for chat
3. **AI Model Loading**: App automatically loads the best available local model
4. **Ask Questions**: Type questions about your document content
5. **Get Answers**: Local AI generates contextual answers based on document content

## 🎯 Use Cases

- **Document Analysis**: Extract insights from research papers, reports, manuals
- **Knowledge Base**: Create searchable document repositories
- **Content Summarization**: Get specific answers from long documents
- **Research Assistance**: Query academic or technical documents
- **Business Intelligence**: Analyze business reports and documents
- **Private Data Processing**: Handle sensitive documents without external APIs

## 🛠️ System Requirements

### Minimum Requirements
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space for models
- **Python**: 3.8+

### Recommended Requirements
- **RAM**: 8GB+ for better performance
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: 5GB+ for multiple model options

## 🔄 Fallback Systems

The application includes multiple fallback mechanisms:

1. **Primary**: Local transformer models (if available)
2. **Secondary**: Enhanced keyword matching with context awareness
3. **Final**: Simple keyword-based matching

This ensures the app works even when local models fail to load.

## 📁 Project Structure

```
pdf_project/
├── app.py                 # Main Streamlit application
├── src/
│   ├── pdf_processor.py  # PDF text extraction and chunking
│   ├── embeddings.py     # Vector embedding management
│   └── llm_client.py     # Local AI integration and fallbacks
├── requirements.txt       # Full dependencies (with local AI)
├── requirements-minimal.txt # Minimal dependencies (fallback only)
├── config.env.example    # Configuration template
└── README.md            # This file
```

## 🐛 Troubleshooting

### Common Issues

1. **"Model loading error"**
   - Check if you have enough RAM (4GB+ required)
   - Try the minimal requirements: `pip install -r requirements-minimal.txt`
   - The app will automatically use fallback systems

2. **"CUDA out of memory"**
   - Use smaller models (they're automatically selected)
   - Close other applications to free up memory
   - The app will fall back to CPU models

3. **Slow performance**
   - First run is slower (model downloading)
   - Use GPU if available
   - Smaller models are faster but less accurate

4. **PDF Processing Errors**
   - Ensure the PDF is not corrupted
   - Check if the PDF contains extractable text
   - Verify file permissions

### Performance Tips

- **GPU Users**: Models will automatically use CUDA for faster inference
- **CPU Users**: Smaller models are automatically selected for better performance
- **Memory**: Close other applications for better model loading
- **First Run**: Models are downloaded automatically (may take a few minutes)

## 🔒 Privacy & Security

- **Complete Privacy**: All processing happens locally on your computer
- **No Data Transmission**: Your documents never leave your machine
- **No API Keys**: No external services or authentication required
- **Offline Capable**: Works without internet after initial setup

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Streamlit** for the web framework
- **LangChain** for document processing
- **Hugging Face** for transformer models
- **PyTorch** for deep learning framework
- **FAISS** for vector search
- **Sentence Transformers** for embeddings

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Try the minimal requirements setup
3. Check the app's System Status
4. Open an issue with detailed error information

---

**Happy Local AI Document Chatting! 🤖📚💬**
