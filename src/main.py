import os
import time
from typing import List, Dict
from langchain.schema import Document
import streamlit as st
from document_loader import DocumentLoader
from text_processor import TextProcessor
from qa_generator import QAGenerator

# Initialize components
loader = DocumentLoader.get_loader()
processor = TextProcessor(min_chunk_size=100, max_workers=10)  # Reduced workers
generator = QAGenerator(max_workers=20,  # Reduced workers
                       hierarchy_excel_path="Loopio Library Structure.xlsx")

def display_qa_batch(container, pairs: List[Dict], start_index: int = 0):
    """Display a batch of Q&A pairs with proper indexing"""
    with container:
        for i, pair in enumerate(pairs, start=start_index + 1):
            with st.expander(f"Q{i}: {pair['question']}"):
                st.markdown(f"**Answer:** {pair['answer']}")
                metadata = [
                    f"**Source:** {pair.get('source', '')}",
                    f"**Pages:** {pair.get('page', '')}",
                    f"**Stack:** {pair.get('stack', 'Unclassified')}",
                    f"**Category:** {pair.get('category', 'General')}",
                    f"**Subcategory:** {pair.get('subcategory', 'Other')}"
                ]
                st.caption(" | ".join(metadata))

def reset_app():
    """Reset the app state completely"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.qa_pairs = []
    st.session_state.batch_containers = []
    st.session_state.processing = False
    st.session_state.completed = False
    st.session_state.file_uploader_key = str(time.time())

def main():
    st.set_page_config(page_title="Q&A Generator", layout="wide")
    st.title("Document Q&A Generator")
    
    # Initialize session state
    if 'qa_pairs' not in st.session_state:
        st.session_state.qa_pairs = []
    if 'batch_containers' not in st.session_state:
        st.session_state.batch_containers = []
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'completed' not in st.session_state:
        st.session_state.completed = False
    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = "default"

    # Sidebar
    with st.sidebar:
        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files", 
            type="pdf", 
            accept_multiple_files=True,
            key=st.session_state.file_uploader_key
        )
        
        generate_disabled = st.session_state.processing or not uploaded_files or st.session_state.completed
        
        if not st.session_state.completed:
            if st.button("Generate Q&A", disabled=generate_disabled, type="primary"):
                st.session_state.processing = True
                st.session_state.qa_pairs = []
                st.session_state.batch_containers = []
                st.session_state.completed = False
                st.rerun()
                
        if st.session_state.completed:
            if st.button("Reset", type="primary"):
                reset_app()
                st.rerun()
                
        if st.session_state.processing:
            st.info("Processing in progress...")
        elif st.session_state.completed:
            st.success(f"Generated {len(st.session_state.qa_pairs)} Q&A pairs.")

    # Main content
    status_text = st.empty()
    progress_bar = st.progress(0)
    main_container = st.container()

    # Display existing batches
    if st.session_state.get('qa_pairs'):
        with main_container:
            start_idx = 0
            for batch in st.session_state.get('batch_containers', []):
                display_qa_batch(main_container, batch, start_idx)
                start_idx += len(batch)
                st.markdown("---")

    # Processing logic
    if st.session_state.processing and uploaded_files and not st.session_state.completed:
        temp_files = []
        try:
            # Save uploaded files
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
            
            # Step 2: Text Processing
            status_text.text("Processing document content...")
            ques_gen_chunks, ans_gen_chunks = processor.process(documents)
            progress_bar.progress(40)
            
            # Step 3: Q&A Generation in batches
            status_text.text("Generating Q&A pairs...")
            batch_size = 15
            total_chunks = len(ques_gen_chunks)
            
            if total_chunks == 0:
                status_text.warning("No content chunks extracted. Check document text.")
                progress_bar.progress(100)
            else:
                for i in range(0, total_chunks, batch_size):
                    batch_chunks = ques_gen_chunks[i:i+batch_size]
                    new_pairs = generator.generate_qa_pairs(
                        ques_gen_chunks=batch_chunks,
                        ans_gen_chunks=ans_gen_chunks
                    )
                    
                    if new_pairs:
                        st.session_state.batch_containers.append(new_pairs)
                        st.session_state.qa_pairs.extend(new_pairs)
                        
                        # Calculate start index
                        start_idx = sum(len(b) for b in st.session_state.batch_containers[:-1])
                        
                        # Update progress
                        progress = min(40 + (i / total_chunks * 60), 99)
                        progress_bar.progress(int(progress))
                        
                        # Display this batch immediately
                        with main_container:
                            st.subheader(f"Batch {len(st.session_state.batch_containers)}")
                            display_qa_batch(main_container, new_pairs, start_idx)
                            st.markdown("---")
                        
                        # Force UI update
                        st.rerun()
                
                progress_bar.progress(100)
                status_text.text(f"Completed! Generated {len(st.session_state.qa_pairs)} Q&A pairs.")
                st.session_state.completed = True
                st.rerun()
            
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
            st.rerun()

if __name__ == "__main__":
    main()