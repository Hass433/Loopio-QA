# loopio_classifier.py
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class LoopioClassifier:
    def __init__(self, excel_path: str):
        self.hierarchy = self._load_hierarchy(excel_path)
        self.llm = AzureChatOpenAI(
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            temperature=0
        )
        self.prompt_template = ChatPromptTemplate.from_template(
            """Classify the following Q&A pair into the appropriate Stack, Category, and Subcategory based on the provided hierarchy and definitions.

            Q&A Pair:
            Question: {question}
            Answer: {answer}

            Hierarchy and Definitions:
            {hierarchy_context}

            Examples:
            1. Question: "What are your helpdesk support hours?"
            Answer: "The online Customer Support Portal and Helpdesk is available 24 hours a day..."
            Classification: Stack: "Medius General Organization and Support", Category: "Support", Subcategory: "Day-to-Day Support"

            2. Question: "What is your risk approach?"
            Answer: "Medius is aware that every project comes with risks..."
            Classification: Stack: "Medius General Organization and Support", Category: "Implementation Services", Subcategory: "Risk Management"

            3. Question: "What is the process for invoice validation?"
            Answer: "Invoice validation involves checking for discrepancies..."
            Classification: Stack: "Medius Procurement and APA", Category: "AP Automation", Subcategory: "Validation and Approval"

            4. Question: "How does the system handle user access control?"
            Answer: "Access control is managed via SSO and MFA..."
            Classification: Stack: "Security & Due Diligence", Category: "Access Control", Subcategory: "Methods"

            Return the classification in the format:
            Stack: "[stack_name]"
            Category: "[category_name]"
            Subcategory: "[subcategory_name]"
            """
        )

    def _load_hierarchy(self, excel_path: str) -> str:
        try:
            df = pd.read_excel(excel_path, sheet_name="Sheet1")
            hierarchy = {}
            current_stack = ""
            current_category = ""
            
            for _, row in df.iterrows():
                if pd.notna(row['Stack']):
                    current_stack = row['Stack']
                    hierarchy[current_stack] = {}
                if pd.notna(row['Categories']):
                    current_category = row['Categories']
                    hierarchy[current_stack][current_category] = {}
                if pd.notna(row['Subcategories']):
                    subcategory = row['Subcategories']
                    definition = row['Definition'] if pd.notna(row['Definition']) else "No definition provided."
                    hierarchy[current_stack][current_category][subcategory] = definition
            
            # Convert hierarchy to a string for prompting
            hierarchy_str = ""
            for stack, categories in hierarchy.items():
                hierarchy_str += f"Stack: {stack}\n"
                for category, subcategories in categories.items():
                    hierarchy_str += f"  Category: {category}\n"
                    for subcategory, definition in subcategories.items():
                        hierarchy_str += f"    Subcategory: {subcategory}\n"
                        hierarchy_str += f"      Definition: {definition}\n"
            return hierarchy_str
        except Exception as e:
            print(f"Error loading hierarchy from Excel: {str(e)}")
            # Return a minimal hierarchy string to prevent complete failure
            return "Stack: Unclassified\n  Category: General\n    Subcategory: Other\n      Definition: Default category for unclassified content.\n"

    def classify(self, question: str, answer: str) -> dict:
        try:
            prompt = self.prompt_template.format(
                question=question,
                answer=answer,
                hierarchy_context=self.hierarchy
            )
            response = self.llm.invoke(prompt)
            return self._parse_response(response.content)
        except Exception as e:
            print(f"Classification error: {str(e)}")
            # Return default classification in case of error
            return {
                'stack': 'Unclassified',
                'category': 'General',
                'subcategory': 'Other'
            }

    def _parse_response(self, response: str) -> dict:
        lines = response.split('\n')
        classification = {
            'stack': 'Unclassified',
            'category': 'General',
            'subcategory': 'Other'
        }
        
        for line in lines:
            if line.startswith("Stack:"):
                classification['stack'] = line.split(":")[1].strip().strip('"')
            elif line.startswith("Category:"):
                classification['category'] = line.split(":")[1].strip().strip('"')
            elif line.startswith("Subcategory:"):
                classification['subcategory'] = line.split(":")[1].strip().strip('"')
        
        return classification