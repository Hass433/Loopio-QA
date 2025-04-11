"""
Parallel Q&A Generator Module with separate Q&A generation
"""

import os
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

class QAGenerator:
    """Generates Q&A pairs using separate strategies for question and answer generation"""
    
    def __init__(self, max_workers: int = 5):
        """
        Initialize with parallel processing capability
        
        Args:
            max_workers: Number of parallel threads to use
        """
        self.llm = AzureChatOpenAI(
            openai_api_version=st.secrets["AZURE_OPENAI_API_VERSION"],
            azure_deployment=st.secrets["AZURE_OPENAI_DEPLOYMENT"],
            openai_api_key=st.secrets["AZURE_OPENAI_API_KEY"],
            azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
            temperature=0
        )
        
        self.max_workers = max_workers
        
        # Separate prompts for question and answer generation
        self.question_prompt = ChatPromptTemplate.from_template(
            """As an Request for Proposal (RFP) reponse specialist, analyze this document section and generate {num_questions} high-value questions.
            
            Focus on questions that are related to RFP which has prospects as targets.
            Questions should be of type WH Questions as it will be used as a reference for the questions that can be asked when submitting an RFP but of course related to the document.

            Do Not generate questions : 
            - On the document's table of contents.
            - On the document titles and headings.

            Question should probe critical document information.

            Document Section:
            {text}
            
            Return each question on a new line prefixed with "Q: "."""
        )
        
        self.answer_prompt = ChatPromptTemplate.from_template(
            """Given the following question and document context, provide a precise answer 
            that can be directly verified from the text. Include exact figures, dates, 
            or specifications when available.

            Question: {question}

            When providing your answer:
            1. Answer must be precise, directly verifiable from the text and fully completed
            2. Include exact figures, dates, or specifications when available
            3. Only include information you can verify in the context
            4. Be specific about which document contains each fact
            
            Document Context:
            {context}
            
            Return the answer prefixed with "A: "."""
        )
    
    #"Simple" Solution for multiple sources in answers 
    def _get_most_relevant_chunks(self, question: str, chunks: List[Document], top_n: int = 3) -> List[Document]:
        """Select top N most relevant chunks to the question using TF-IDF cosine similarity"""
        if not chunks:
            return []
            
        # Combine question and chunks for vectorization
        texts = [question] + [chunk.page_content for chunk in chunks]
        
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Calculate similarity between question and each chunk
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
            
            # Pair chunks with their similarity scores and sort
            scored_chunks = sorted(zip(chunks, similarities), key=lambda x: x[1], reverse=True)
            
            # Return top N chunks
            return [chunk for chunk, score in scored_chunks[:top_n]]
            
        except Exception as e:
            print(f"Error calculating chunk relevance: {str(e)}")
            return chunks[:top_n]  # Fallback to first N chunks if error occurs
    
    def _generate_questions(self, chunk: Document, num_questions: int) -> List[str]:
        """Generate questions from a document chunk"""
        if not chunk.page_content.strip():
            return []
            
        try:
            response = self.llm.invoke(self.question_prompt.format(
                text=chunk.page_content,
                num_questions=num_questions
            ))
            
            questions = []
            for line in response.content.split('\n'):
                line = line.strip()
                if line.startswith('Q:'):
                    questions.append(line[2:].strip())
            
            return questions
            
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            return []
    
    def _generate_answer(self, question: str, context_chunks: List[Document]) -> Dict[str, str]:
        """Generate answer for a question using top 3 most relevant context chunks"""
        try:
            # Select top 3 most relevant chunks
            relevant_chunks = self._get_most_relevant_chunks(question, context_chunks, top_n=3)
            
            if not relevant_chunks:
                return None
                
            # Track exactly which pages were used
            used_pages = set()
            sources = set()
            context_pieces = []
            
            for chunk in relevant_chunks:
                # Get the original page number (not the combined metadata)
                page_num = str(chunk.metadata.get('orig_page_number', chunk.metadata.get('page_number', '?')))
                used_pages.add(page_num)
                if 'source' in chunk.metadata:
                    sources.add(chunk.metadata['source'])
                context_pieces.append(chunk.page_content)
            
            # Generate answer using only the relevant context
            response = self.llm.invoke(self.answer_prompt.format(
                question=question,
                context="\n\n".join(context_pieces)
            ))
            
            answer = response.content.strip()
            if answer.startswith('A:'):
                answer = answer[2:].strip()
            

            sorted_pages = sorted(used_pages, key=lambda x: int(x))

            
            return {
                'question': question,
                'answer': answer,
                'source': ", ".join(sources) if sources else "",
                'page': ", ".join(sorted_pages) if used_pages else "?"
            }
            
        except Exception as e:
            print(f"Error generating answer for '{question}': {str(e)}")
            return None
    
    def generate_qa_pairs(self, 
                        ques_gen_chunks: List[Document],
                        ans_gen_chunks: List[Document]) -> List[Dict[str, str]]:
        """Generate Q&A pairs using separate chunked documents"""
        qa_pairs = []
        
        # Calculate dynamic number of questions per chunk based on content
        def calculate_questions_per_chunk(chunk: Document) -> int:
            # Base number of questions
            base_questions = 2
            
            word_count = len(chunk.page_content.split())
            if word_count > 500:
                return base_questions + 2
            elif word_count > 300:
                return base_questions + 1
            return base_questions
        
        # First generate all potential questions
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            question_futures = []
            for chunk in ques_gen_chunks:
                num_questions = calculate_questions_per_chunk(chunk)
                question_futures.append(
                    executor.submit(
                        self._generate_questions,
                        chunk,
                        num_questions
                    )
                )
            
            # Collect all generated questions
            all_questions = []
            for future in question_futures:
                try:
                    questions = future.result()
                    if questions:
                        all_questions.extend(questions)
                except Exception as e:
                    print(f"Error in question generation: {str(e)}")
                    continue
        
        # Then generate answers for each question using all context chunks
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            answer_futures = []
            for question in all_questions:
                answer_futures.append(
                    executor.submit(
                        self._generate_answer,
                        question,
                        ans_gen_chunks
                    )
                )
            
            # Collect all Q&A pairs
            for future in answer_futures:
                try:
                    pair = future.result()
                    if pair:
                        qa_pairs.append(pair)
                except Exception as e:
                    print(f"Error in answer generation: {str(e)}")
                    continue
        
        return qa_pairs