import os
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


class TextChunker:
    def __init__(self, input_dir="../data", output_dir="../data/chunks", chunk_size=1000, chunk_overlap=150):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        os.makedirs(self.output_dir, exist_ok=True)

    def read_text_files(self):
        """Reads all .txt files from input directory and returns a dict of filename -> text"""
        texts = {}
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(".txt"):
                    path = os.path.join(root, file)
                    with open(path, "r", encoding="utf-8") as f:
                        texts[file] = f.read()
        return texts

    # --------------------------------------------------------------------
    #  SEMANTIC CHUNKING FUNCTION
    # --------------------------------------------------------------------
    def semantic_chunk_text(self, text):
        """
        Splits text semantically using natural separators.
        Better for paragraphs, logical sections, or sentences.
        """
        semantic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )
        return semantic_splitter.split_text(text)

    # --------------------------------------------------------------------
    #  RECURSIVE TEXT SPLITTING FUNCTION
    # --------------------------------------------------------------------
    def recursive_split_text(self, text):
        """
        Simpler recursive splitter — useful when text is not well-structured.
        """
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return recursive_splitter.split_text(text)


# --------------------------------------------------------------------
#  MARKDOWN CHUNKER CLASS
# --------------------------------------------------------------------
class MarkdownChunker(TextChunker):
    """
    Extends TextChunker to handle Markdown structure (tables, headings, etc.)
    """

    def markdown_chunk_text(self, text):
        """
        Preserves Markdown sections (headers, tables) and splits semantically within them.
        """
        # Step 1: Split by markdown headers
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
        )
        md_docs = md_splitter.split_text(text)

        # Step 2: Further split within each markdown section
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " "]
        )

        chunks = []
        for section in md_docs:
            section_text = section.page_content
            section_chunks = recursive_splitter.split_text(section_text)
            chunks.extend(section_chunks)

        return chunks

    # --------------------------------------------------------------------
    # PROCESS ALL FILES WITH MARKDOWN SUPPORT
    # --------------------------------------------------------------------
    def process_all_files(self, mode="markdown"):
        """
        Processes all text files using 'semantic', 'recursive', or 'markdown' mode.
        Saves chunked text files in the output folder.
        """
        texts = self.read_text_files()

        for filename, content in texts.items():
            # Choose mode
            if mode == "semantic":
                chunks = self.semantic_chunk_text(content)
            elif mode == "recursive":
                chunks = self.recursive_split_text(content)
            elif mode == "markdown":
                chunks = self.markdown_chunk_text(content)
            else:
                raise ValueError("Invalid mode. Choose 'semantic', 'recursive', or 'markdown'.")

            # Write chunks to output file
            base_name = os.path.splitext(filename)[0]
            output_file = os.path.join(self.output_dir, f"{base_name}_{mode}_chunks.txt")

            with open(output_file, "w", encoding="utf-8") as f:
                for i, chunk in enumerate(chunks, 1):
                    f.write(f"[Chunk {i}]\n{chunk}\n\n")

            print(f"✅ {len(chunks)} chunks ({mode}) saved for {filename} → {output_file}")

        print("\nAll files chunked successfully!")


# --------------------------------------------------------------------
# 🚀 RUN AS SCRIPT
# --------------------------------------------------------------------
# if __name__ == "__main__":
#     chunker = MarkdownChunker(
#         input_dir="../data",
#         output_dir="../data/chunks",
#         chunk_size=1000,
#         chunk_overlap=150
#     )

#     # Options:
#     # chunker.process_all_files(mode="semantic")
#     # chunker.process_all_files(mode="recursive")
#     chunker.process_all_files(mode="markdown")
