"""
Text Processor Module with Semantic Chunking
Handles text extraction, cleaning, and chunking from documents using semantic meaning.
"""

from typing import List, Tuple
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

class TextProcessor:
    """Processes text from documents for Q&A generation using semantic chunking"""
    
    def __init__(self, min_chunk_size=100,max_workers=100):
        """
        Initialize TextProcessor with semantic chunking parameters for Q&A
        
        Args:
            min_chunk_size: Minimum size of text chunks to ensure meaningful content
        """
        self.min_chunk_size = min_chunk_size
        self.max_workers = max_workers

        # Initialize the embeddings model for semantic chunking
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )        
        # Initialize semantic chunkers for questions and answers
        # Using different methods for question and answer generation
        self.ques_splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile",  # Good for granular question chunks
            breakpoint_threshold_amount=85.0,  # Slightly more aggressive splitting for questions
            min_chunk_size=self.min_chunk_size
        )
        
        self.ans_splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="standard_deviation",  # Better for longer answer context
            breakpoint_threshold_amount=2.0,  # Less aggressive splitting for answers
            min_chunk_size=self.min_chunk_size
        )
    
    def process(self, documents: List[Document]) -> Tuple[List[Document], List[Document]]:
        """
        Process documents with separate semantic chunking strategies for Q&A generation
        
        Args:
            documents: List of Document objects
            
        Returns:
            Tuple of (question_gen_chunks, answer_gen_chunks)
        """
        # Clean the documents first
        cleaned_docs = self._clean_documents(documents)
        
        # Process each document separately to maintain document source metadata
        all_ques_chunks = []
        all_ans_chunks = []
        
        for doc in cleaned_docs:
            # Skip empty documents
            if not doc.page_content.strip():
                continue
                
            # Split for question generation (more granular chunks)
            ques_chunks = self.ques_splitter.split_documents([doc])
            all_ques_chunks.extend(ques_chunks)
            
            # Split for answer generation (larger, more contextual chunks)
            ans_chunks = self.ans_splitter.split_documents([doc])
            all_ans_chunks.extend(ans_chunks)
        
        return all_ques_chunks, all_ans_chunks
    
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
            # Basic cleaning - could be enhanced with more sophisticated content filtering
            text = doc.page_content
            
            # Skip empty documents
            if not text.strip():
                continue
                
            cleaned_docs.append(
                Document(
                    page_content=text,
                    metadata=doc.metadata
                )
            )
        
        return cleaned_docs