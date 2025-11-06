import os
from langchain_community.document_loaders import PyPDFLoader
from docling.document_converter import DocumentConverter

class AdvancedPDFHandler:
    """
    Handles PDF extraction using Docling (for tables/structured data)
    and LangChain PyPDFLoader (for text).
    Saves the extracted text under ../data/pdf_texts relative to the script.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        
        # Resolve output directory outside 'rag' → ../data/pdf_texts
        current_dir = os.path.dirname(os.path.abspath(__file__))  # .../agentic_rag/rag
        project_root = os.path.dirname(current_dir)               # .../agentic_rag
        self.output_dir = os.path.join(project_root, "data", "pdf_texts")
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_with_docling(self):
        """
        Use Docling to extract structured PDF content (tables, lists, etc.)
        """
        try:
            print("Trying structured extraction with Docling...")
            converter = DocumentConverter()
            result = converter.convert(self.pdf_path)
            markdown_text = result.document.export_to_markdown()
            markdown_text = result.document.export_to_dict()


            if len(markdown_text.strip()) < 100:
                raise ValueError("Docling extracted too little text")

            print("Docling extraction successful.")
            return {"text": markdown_text, "method": "docling"}

        except Exception as e:
            print( f"Docling extraction failed: {e}")
            return None

    def extract_with_langchain(self):
        """
        Fallback text extraction using LangChain's PyPDFLoader.
        """
        print(" Falling back to LangChain PyPDFLoader...")
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()
        text = "\n".join([doc.page_content for doc in docs])
        return {"text": text, "method": "langchain"}

    def save_output(self, content: dict):
        """
        Save the extracted text file in ../data/pdf_texts/
        """
        base_name = os.path.basename(self.pdf_path)
        file_name = os.path.splitext(base_name)[0] + ".txt"
        output_path = os.path.join(self.output_dir, file_name)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content["text"])

        print(f" Saved extracted data to: {output_path}")
        return output_path

    def process_pdf(self):
        """
        Automatically extract PDF using Docling (preferred)
        or LangChain (fallback).
        """
        print(f"\n Processing PDF: {self.pdf_path}")

        content = self.extract_with_docling()
        if not content:
            content = self.extract_with_langchain()

        return self.save_output(content)


# if __name__ == "__main__":
#     pdf_file = r"C:\Users\AI_ML PC_4\Desktop\Swarnalatha\Agentic_ai_with_lnaggraph\agentic_rag\data\2408.09869v5.pdf"
#     handler = AdvancedPDFHandler(pdf_file)
#     handler.process_pdf()
