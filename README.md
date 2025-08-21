# 📄 PDF Chatbot

A powerful Streamlit-based PDF chatbot that allows you to upload PDF documents and ask questions about their content using AI-powered document understanding.

## ✨ Features

- **PDF Processing**: Upload and process PDF documents with intelligent text chunking
- **Semantic Search**: Find relevant content using advanced embeddings and vector search
- **AI-Powered Q&A**: Get intelligent answers based on document content
- **Multiple LLM Options**: Hugging Face API integration with smart fallback systems
- **User-Friendly Interface**: Clean Streamlit UI with real-time feedback
- **Persistent Chat**: Maintain conversation history during your session
- **Error Resilience**: Graceful fallbacks when external services are unavailable

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd pdf_project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the project root:

```bash
# Copy the example config
cp config.env.example .env

# Edit .env with your settings
HF_TOKEN=your_huggingface_token_here
LLM_MODEL=google/flan-t5-large
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

**Get your Hugging Face token from**: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 4. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🔧 How It Works

### Architecture

1. **PDF Processing** (`PDFProcessor`): Extracts text and creates intelligent chunks
2. **Embedding Management** (`EmbeddingManager`): Creates vector embeddings for semantic search
3. **LLM Integration** (`HuggingFaceLLM`): Generates AI-powered answers with fallback options

### Workflow

1. **Upload PDF**: Use the sidebar to upload your document
2. **Process Document**: Click "Process Document" to prepare it for chat
3. **Ask Questions**: Type questions about your document content
4. **Get Answers**: AI generates contextual answers based on document content

## 🎯 Use Cases

- **Document Analysis**: Extract insights from research papers, reports, manuals
- **Knowledge Base**: Create searchable document repositories
- **Content Summarization**: Get specific answers from long documents
- **Research Assistance**: Query academic or technical documents
- **Business Intelligence**: Analyze business reports and documents

## 🛠️ Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HF_TOKEN` | Hugging Face API token | Required |
| `LLM_MODEL` | LLM model to use | `google/flan-t5-large` |
| `EMBEDDING_MODEL` | Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Text chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap | `200` |

### Models

- **LLM Models**: `google/flan-t5-large`, `microsoft/DialoGPT-medium`
- **Embedding Models**: `sentence-transformers/all-MiniLM-L6-v2`

## 🔄 Fallback Systems

The application includes multiple fallback mechanisms:

1. **Primary**: Hugging Face API with specified model
2. **Secondary**: Alternative Hugging Face model if primary fails
3. **Final**: Intelligent keyword-based matching system

This ensures the app works even when external APIs are unavailable.

## 📁 Project Structure

```
pdf_project/
├── app.py                 # Main Streamlit application
├── src/
│   ├── pdf_processor.py  # PDF text extraction and chunking
│   ├── embeddings.py     # Vector embedding management
│   └── llm_client.py     # LLM integration and fallbacks
├── requirements.txt       # Python dependencies
├── config.env.example    # Configuration template
└── README.md            # This file
```

## 🐛 Troubleshooting

### Common Issues

1. **"Error generating answer"**
   - Check if your HF_TOKEN is set correctly
   - The app will automatically use fallback systems
   - Check the System Status in the app

2. **PDF Processing Errors**
   - Ensure the PDF is not corrupted
   - Check if the PDF contains extractable text
   - Verify file permissions

3. **Performance Issues**
   - Reduce chunk size for faster processing
   - Use smaller embedding models for better speed
   - Check available system memory

### Debug Information

The app shows real-time status information:
- **LLM Type**: Which AI system is being used
- **HF Token Status**: Whether your token is configured
- **System Status**: Overall application health

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
- **Hugging Face** for AI models
- **FAISS** for vector search
- **Sentence Transformers** for embeddings

## 📞 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify your configuration settings
3. Check the app's System Status
4. Open an issue with detailed error information

---

**Happy Document Chatting! 📚💬**
