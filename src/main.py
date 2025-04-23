import os
import time
from typing import List, Dict
from langchain.schema import Document
import streamlit as st
from document_loader import DocumentLoader
from text_processor import TextProcessor
from qa_generator import QAGenerator
import pandas as pd

# Initialize components at module level
loader = DocumentLoader.get_loader()
processor = TextProcessor(min_chunk_size=100, max_workers=100)
generator = QAGenerator(max_workers=40,
                        hierarchy_excel_path="Loopio Library Structure.xlsx")  

def display_qa_pairs(container, pairs: List[Dict]):
    """Display all Q&A pairs on a single page"""
    with container:
        for i, pair in enumerate(pairs, start=1):
            with st.expander(f"Q{i}: {pair['question']}"):
                st.markdown(f"**Answer:** {pair['answer']}")
                # Create metadata display with classification
                metadata_lines = [
                    f"**Source:** {pair.get('source', '')}",
                    f"**Pages:** {pair.get('page', '')}",
                    f"**Stack:** {pair.get('stack', 'Unclassified')}",
                    f"**Category:** {pair.get('category', 'General')}",
                    f"**Subcategory:** {pair.get('subcategory', 'Other')}"
                ]
                
                # Display metadata as a single block with line breaks
                st.caption(" | ".join(metadata_lines))

def calculate_target_pairs(documents: List[Document]) -> int:
    """Calculate target number of Q&A pairs based on document content"""
    total_words = sum(len(doc.page_content.split()) for doc in documents)
    base_target = max(5, total_words // 500)
    num_docs = len(documents)
    adjusted_target = base_target * (1 + num_docs // 3)
    return min(adjusted_target, 200)

def reset_app():
    """Reset the app state completely as if first opened"""
    # Clear all session state variables
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Re-initialize essential session state variables
    st.session_state.qa_pairs = []
    st.session_state.processing = False
    st.session_state.completed = False
    st.session_state.previously_displayed_count = 0  # Also reset this counter
    st.session_state.file_uploader_key = str(time.time())  # Force file uploader to reset


def main():
    st.set_page_config(page_title="Q&A Generator", layout="wide")
    st.title("Document Q&A Generator")
    
    # Initialize session state variables if they don't exist
    if 'qa_pairs' not in st.session_state:
        st.session_state.qa_pairs = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'completed' not in st.session_state:
        st.session_state.completed = False
    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = "default"
    if 'previously_displayed_count' not in st.session_state:
        st.session_state.previously_displayed_count = 0
    
    # Always show sidebar
    with st.sidebar:
        st.header("Upload Documents")
        # Use a dynamic key for the file uploader to force reset
        uploaded_files = st.file_uploader(
            "Upload PDF files", 
            type="pdf", 
            accept_multiple_files=True,
            key=st.session_state.file_uploader_key
        )
        
        # Disable the generate button if already processing or no files uploaded
        # Or if completed (we'll show the Reset button instead)
        generate_disabled = st.session_state.processing or not uploaded_files or st.session_state.completed
        
        if not st.session_state.completed:
            generate_btn = st.button(
                "Generate Q&A", 
                disabled=generate_disabled,
                type="primary"
            )
        else:
            generate_btn = False  # When completed, don't show Generate button
            
        # Show Reset button only when completed
        if st.session_state.completed:
            if st.button("Reset", type="primary"):
                reset_app()
                st.rerun()
                
        # Show processing status in sidebar
        if st.session_state.processing:
            st.info("Processing in progress...")
        elif st.session_state.completed:
            st.success(f"Generated {len(st.session_state.qa_pairs)} Q&A pairs.")
    
    # Main content area
    status_text = st.empty()
    progress_bar = st.progress(0)
    display_container = st.container()
    
    # Display existing Q&A pairs if available
    if st.session_state.qa_pairs:
        display_qa_pairs(display_container, st.session_state.qa_pairs)
    
    # Handle generation process
    if generate_btn and uploaded_files and not st.session_state.processing:
        st.session_state.processing = True
        st.session_state.qa_pairs = []  # Reset Q&A pairs
        st.session_state.completed = False
        st.session_state.previously_displayed_count = 0  # Reset display counter
        
        temp_files = []
        try:
            # Save uploaded files to temporary location
            for uploaded_file in uploaded_files:
                temp_file = f"temp_{uploaded_file.name}"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                temp_files.append(temp_file)
            
            # Step 1: Document Loading
            status_text.text("Loading documents...")
            documents_dict = loader.load(temp_files)
            documents = []
            for doc_list in documents_dict.values():
                documents.extend(doc_list)
            progress_bar.progress(20)
            
            # Step 2: Text Processing - Performance improvements
            status_text.text("Processing document content...")
            # Performance optimization: process in larger batches
            ques_gen_chunks, ans_gen_chunks = processor.process(documents)
            progress_bar.progress(40)
            
            # Step 3: Q&A Generation - streaming batches with performance improvements
            status_text.text("Generating Q&A pairs...")
            
            # We'll process in batches and display as we go
            batch_size = 15
            total_chunks = len(ques_gen_chunks)
            
            if total_chunks == 0:
                status_text.warning("No content chunks were extracted. Check if your documents contain extractable text.")
                progress_bar.progress(100)
            else:
                for i in range(0, total_chunks, batch_size):
                    batch_chunks = ques_gen_chunks[i:i+batch_size]
                    new_pairs = generator.generate_qa_pairs(
                        ques_gen_chunks=batch_chunks,
                        ans_gen_chunks=ans_gen_chunks  # Using all answer chunks for context
                    )
                    
                    if new_pairs:
                        # Add new pairs to the session state
                        st.session_state.qa_pairs.extend(new_pairs)
                        progress = min(40 + (i / total_chunks * 60), 99)
                        progress_bar.progress(int(progress))
                        
                        # Clear display container
                        display_container.empty()
                        
                        # Display all Q&A pairs, including the new ones
                        display_qa_pairs(display_container, st.session_state.qa_pairs)
                        
                        # Update counter for next batch
                        st.session_state.previously_displayed_count = len(st.session_state.qa_pairs)
                        
                        # Only small delay for UI updates
                        time.sleep(0.05)
                
                progress_bar.progress(100)
                status_text.text(f"Completed! Generated {len(st.session_state.qa_pairs)} Q&A pairs.")
            
            st.session_state.completed = True
            
        except Exception as e:
            status_text.error(f"Error: {str(e)}")
            st.exception(e)
            progress_bar.progress(0)
            
        finally:
            # Clean up temp files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            st.session_state.processing = False
            # Force rerun to update button states
            st.rerun()

if __name__ == "__main__":
    main()