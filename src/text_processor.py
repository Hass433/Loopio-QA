"""
Text Processor Module
Handles text extraction, cleaning, and chunking from documents.
"""

from typing import List,Tuple
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextProcessor:
    """Processes text from documents for Q&A generation"""
    
    def __init__(self, 
                 ques_chunk_size: int = 300, 
                 ques_chunk_overlap: int = 100,
                 ans_chunk_size: int = 2000,
                 ans_chunk_overlap: int = 30):
        """
        Initialize TextProcessor with separate chunking parameters for Q&A
        
        Args:
            ques_chunk_size: Size of text chunks for question generation
            ques_chunk_overlap: Overlap between chunks for question generation
            ans_chunk_size: Size of text chunks for answer generation
            ans_chunk_overlap: Overlap between chunks for answer generation
        """
        self.ques_chunk_size = ques_chunk_size
        self.ques_chunk_overlap = ques_chunk_overlap
        self.ans_chunk_size = ans_chunk_size
        self.ans_chunk_overlap = ans_chunk_overlap
        
        self.ques_splitter = RecursiveCharacterTextSplitter(
            chunk_size=ques_chunk_size,
            chunk_overlap=ques_chunk_overlap
        )
        
        self.ans_splitter = RecursiveCharacterTextSplitter(
            chunk_size=ans_chunk_size,
            chunk_overlap=ans_chunk_overlap
        )
    
    def process(self, documents: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        Process documents with separate chunking strategies for Q&A generation
        
        Args:
            documents: List of Document objects
            
        Returns:
            Tuple of (question_gen_chunks, answer_gen_chunks)
        """
        # Clean the documents first
        cleaned_docs = self._clean_documents(documents)
        
        # Split for question generation (process each document separately)
        ques_gen_docs = []
        for doc in cleaned_docs:
            chunks = self.ques_splitter.split_documents([doc])
            ques_gen_docs.extend(chunks)
        
        # Split for answer generation (process each question chunk separately)
        ans_gen_chunks = []
        for doc in ques_gen_docs:
            chunks = self.ans_splitter.split_documents([doc])
            ans_gen_chunks.extend(chunks)
        
        return ques_gen_docs, ans_gen_chunks
    
    def _clean_documents(self, documents: List[Document]) -> List[Document]:
        """
        Clean documents by removing irrelevant content
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of cleaned Document objects
        """
        
        cleaned_docs = []
        for doc in documents:

            text = doc.page_content
            
            cleaned_docs.append(
                Document(
                    page_content=text,
                    metadata=doc.metadata
                )
            )
        
        return cleaned_docs