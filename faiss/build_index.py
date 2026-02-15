import os
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from utils import DATA_JSONL, load_jsonl_data, FAISS_DIR

def main():
    figures = load_jsonl_data(DATA_JSONL)
    
    docs_title = []
    docs_abstract = []
    docs_caption = []

    print(f"Processing {len(figures)} figures...")

    for idx, figure in enumerate(figures):
        # 1. Prepare Metadata
        # Copy the full dictionary and add the ID
        metadata = figure.copy()
        metadata['id'] = idx 

        # 2. Create Title Document
        title_text = figure['title']
        docs_title.append(Document(
            page_content=title_text,
            metadata=metadata
        ))

        # 3. Create Abstract Document
        abstract_text = figure['abstract']
        docs_abstract.append(Document(
            page_content=abstract_text,
            metadata=metadata
        ))

        # 4. Create Caption Document
        caption_text = figure['figure_caption']
        
        docs_caption.append(Document(
            page_content=caption_text,
            metadata=metadata
        ))

    # Initialize Embedding Model
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

    # Build and Save Title Index
    print(f"Building Title Index with {len(docs_title)} documents...")
    db_title = FAISS.from_documents(docs_title, embedding_model)
    db_title.save_local(os.path.join(FAISS_DIR, "title_index"))

    # Build and Save Abstract Index
    print(f"Building Abstract Index with {len(docs_abstract)} documents...")
    db_abstract = FAISS.from_documents(docs_abstract, embedding_model)
    db_abstract.save_local(os.path.join(FAISS_DIR, "abstract_index"))

    # Build and Save Caption Index
    print(f"Building Caption Index with {len(docs_caption)} documents...")
    db_caption = FAISS.from_documents(docs_caption, embedding_model)
    db_caption.save_local(os.path.join(FAISS_DIR, "caption_index"))

    print("Hierarchical indexing complete.")

if __name__ == "__main__":
    main()