#document_loader.py
"""
Enhanced Document Loader Module
Handles loading of multiple PDF documents with parallel processing.
"""

import os
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

class DocumentLoader:
    """Base class for document loaders"""
    
    def load(self, file_paths: List[str]) -> Dict[str, List[Document]]:
        """Load multiple documents from file paths"""
        raise NotImplementedError("Subclasses must implement load method")
    
    @staticmethod
    def get_loader() -> 'DocumentLoader':
        """Factory method to get appropriate loader"""
        return PDFLoader()

class PDFLoader(DocumentLoader):
    """Loader for multiple PDF documents with parallel processing"""
    
    def load(self, file_paths: List[str]) -> Dict[str, List[Document]]:
        """Load multiple PDF documents in parallel"""
        all_docs = {}
        
        def load_single_pdf(file_path: str) -> tuple:
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                for i, doc in enumerate(docs, start=1):
                    doc.metadata.update({
                        'page_number': i,
                        'source': os.path.basename(file_path),
                        'file_path': file_path
                    })
                return (file_path, docs)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
                return (file_path, None)
        
        with ThreadPoolExecutor() as executor:
            results = executor.map(load_single_pdf, file_paths)
            
            for file_path, docs in results:
                if docs:
                    all_docs[file_path] = docs
        
        return all_docs