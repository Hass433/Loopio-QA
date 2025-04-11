"""
Excel Formatter
Exports Q&A pairs to properly formatted Excel files.
"""

import os
from typing import List, Dict
import pandas as pd

class ExcelFormatter:
    def __init__(self):
        """
        Initialize Excel formatter.
        Make sure xlsxwriter is installed: pip install xlsxwriter
        """
        pass

    def format(self, qa_pairs: List[Dict[str, str]], output_path: str) -> str:
        """
        Format Q&A pairs into Excel with proper columns
        
        Args:
            qa_pairs: List of dictionaries containing question-answer pairs
            output_path: Path to save the Excel file (.xlsx)
            
        Returns:
            Path to the saved Excel file
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Ensure .xlsx extension
        if not output_path.lower().endswith('.xlsx'):
            output_path = os.path.splitext(output_path)[0] + '.xlsx'
        
        # Create DataFrame with proper column names
        df = pd.DataFrame(qa_pairs).rename(columns={
            'question': 'Question',
            'answer': 'Answer',
            'source': 'Source Document(s)',
            'page': 'Page Number(s)'
        })
        
        # Ensure all required columns exist
        for col in ['Question', 'Answer', 'Source Document(s)', 'Page Number(s)']:
            if col not in df.columns:
                df[col] = ''
        
        # Reorder columns
        df = df[['Question', 'Answer', 'Source Document(s)', 'Page Number(s)']]
        
        # Write to Excel
        df.to_excel(output_path, index=False, engine='xlsxwriter')
        
        return output_path