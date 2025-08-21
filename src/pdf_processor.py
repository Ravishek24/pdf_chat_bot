import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import tempfile
import os
from typing import List

class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_path = tmp_file.name
            
            # Extract text
            text = ""
            with open(tmp_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
            return text
        
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def create_chunks(self, text: str) -> List[Document]:
        """Split text into chunks"""
        chunks = self.text_splitter.split_text(text)
        documents = []
        
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={"chunk_id": i}
            )
            documents.append(doc)
        
        return documents
    
    def process_pdf(self, pdf_file) -> List[Document]:
        """Complete PDF processing pipeline"""
        text = self.extract_text_from_pdf(pdf_file)
        chunks = self.create_chunks(text)
        return chunks