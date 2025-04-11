# main.py (updated with automatic question generation)
import os
import time
from typing import List
from langchain.schema import Document
import streamlit as st
from document_loader import DocumentLoader
from text_processor import TextProcessor
from qa_generator import QAGenerator
from output_formatter import ExcelFormatter
#from qa_evaluator import QAEvaluator
import pandas as pd

# Initialize components at module level
loader = DocumentLoader.get_loader()
processor = TextProcessor(ques_chunk_size=300, ques_chunk_overlap=50,
                         ans_chunk_size=2000, ans_chunk_overlap=500)
generator = QAGenerator(max_workers=40)
formatter = ExcelFormatter()

def display_new_batches(container, qa_pairs):
    """Display Q&A pairs in batches"""
    batch_size = 15
    total_batches = (len(qa_pairs) // batch_size) + 1
    
    with container:
        for batch_num in range(total_batches):
            batch_start = batch_num * batch_size
            batch_end = min((batch_num + 1) * batch_size, len(qa_pairs))
            batch = qa_pairs[batch_start:batch_end]
            
            st.subheader(f"Questions {batch_start+1}-{batch_end}")
            
            for i, pair in enumerate(batch, start=batch_start+1):
                with st.expander(f"Q{i}: {pair['question']}"):
                    st.markdown(f"**Answer:** {pair['answer']}")
                    st.caption(f"Source: {pair.get('source', '')} | Pages: {pair.get('page', '')}")
            
            if batch_end < len(qa_pairs):
                st.markdown("---")

def calculate_target_pairs(documents: List[Document]) -> int:
    """Calculate target number of Q&A pairs based on document content"""
    total_words = sum(len(doc.page_content.split()) for doc in documents)
    
    # Base target: ~1 question per 500 words
    base_target = max(5, total_words // 500)
    
    # Adjust based on number of documents
    num_docs = len(documents)
    adjusted_target = base_target * (1 + num_docs // 3)
    
    # Cap at reasonable maximum
    return min(adjusted_target, 200)

def main():
    st.set_page_config(page_title="Q&A Generator", layout="wide")
    st.title("Document Q&A Generator")
    
    if 'qa_pairs' not in st.session_state:
        st.session_state.qa_pairs = []
    if 'displayed_batches' not in st.session_state:
        st.session_state.displayed_batches = 0

    status_text = st.empty()
    progress_bar = st.progress(0)

    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files", 
            type="pdf", 
            accept_multiple_files=True
        )
        
        generate_btn = st.button("Generate Q&A")
    
    if not uploaded_files:
        st.info("Please upload one or more PDF files to begin")
        return
    
    if generate_btn:
        st.session_state.qa_pairs = []
        st.session_state.displayed_batches = 0
        
        temp_files = []
        for uploaded_file in uploaded_files:
            temp_file = f"temp_{uploaded_file.name}"
            with open(temp_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            temp_files.append(temp_file)
        
        try:
            # Step 1: Document Loading
            status_text.text("Loading documents...")
            documents_dict = loader.load(temp_files)
            documents = []
            for doc_list in documents_dict.values():
                documents.extend(doc_list)
            progress_bar.progress(20)
            
            # Step 2: Text Processing
            status_text.text("Processing document content...")
            ques_gen_chunks, ans_gen_chunks = processor.process(documents)
            progress_bar.progress(40)
            
            # Step 3: Q&A Generation
            status_text.text("Generating Q&A pairs...")
            display_container = st.container()
            
            # Generate all questions at once (number determined automatically)
            new_pairs = generator.generate_qa_pairs(
                ques_gen_chunks=ques_gen_chunks,
                ans_gen_chunks=ans_gen_chunks
            )
            
            if new_pairs:
                st.session_state.qa_pairs = new_pairs
                progress_bar.progress(100)
                status_text.text(f"Completed! Generated {len(st.session_state.qa_pairs)} Q&A pairs.")
                display_new_batches(display_container, st.session_state.qa_pairs)
            
        finally:
            # Clean up temp files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

if __name__ == "__main__":
    main()