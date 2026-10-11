import os

# macOS ships multiple OpenMP runtimes (faiss-cpu bundles one, numpy/MKL
# another); loading both aborts with `OMP: Error #15`. Set this BEFORE the
# faiss import so the runtime check is silenced.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from PIL import Image

from utils import FAISS_DIR, FIG_RAG_DIR

# Global cache to avoid reloading indices
FAISS_DB_CACHE = {}

def load_faiss_indices():
    """Load all three indices into memory if not already loaded."""
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    
    indices = ["title_index", "abstract_index", "caption_index"]
    for idx_name in indices:
        if idx_name not in FAISS_DB_CACHE:
            print(f"Loading {idx_name}...")
            path = f"{FAISS_DIR}/{idx_name}"
            FAISS_DB_CACHE[idx_name] = FAISS.load_local(
                path, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )

def hierarchical_retrieval(query_title: str, query_abstract: str, query_caption: str,
                         t1: int = 100, t2: int = 30, k: int = 3,
                         exclude_platform_id: str = None):
    """
    3-stage hierarchical retrieval:
      1. Search Title Index  -> Top t1 papers
      2. Search Abstract Index (filtered by t1 IDs) -> Top t2 papers
      3. Search Caption Index (filtered by t2 IDs) -> Top k figures

    Parameters
    ----------
    exclude_platform_id : str, optional
        If set, exclude all figures from this paper (leave-one-out evaluation).
    """

    # Ensure DBs are loaded
    load_faiss_indices()

    q_title = query_title
    q_abstract = query_abstract
    q_caption = query_caption

    # --- LEVEL 1: TITLE SEARCH ---
    print(f"Searching Titles for top {t1} candidates...")
    if exclude_platform_id:
        filter_s1 = lambda metadata: metadata['platform_id'] != exclude_platform_id
        docs_title = FAISS_DB_CACHE["title_index"].similarity_search(
            q_title, k=t1, filter=filter_s1, fetch_k=t1 * 100)
    else:
        docs_title = FAISS_DB_CACHE["title_index"].similarity_search(q_title, k=t1)

    # Extract IDs to filter the next stage
    ids_stage_1 = {doc.metadata['id'] for doc in docs_title}

    if not ids_stage_1:
        return []

    print(f"Stage 1 (Title) passed {len(ids_stage_1)} unique IDs.")

    # --- LEVEL 2: ABSTRACT SEARCH ---
    print(f"Refining with Abstract for top {t2} candidates...")

    def filter_stage_1(metadata):
        if metadata['id'] not in ids_stage_1:
            return False
        if exclude_platform_id and metadata['platform_id'] == exclude_platform_id:
            return False
        return True

    # Search a larger pool to ensure we find overlap with the filter
    docs_abstract = FAISS_DB_CACHE["abstract_index"].similarity_search(
        q_abstract,
        k=t2,
        filter=filter_stage_1,
        fetch_k=len(ids_stage_1) * 100
    )
    ids_stage_2 = {doc.metadata['id'] for doc in docs_abstract}

    if not ids_stage_2:
        return []
    print(f"Stage 2 (Abstract) passed {len(ids_stage_2)} unique IDs.")

    # --- LEVEL 3: CAPTION SEARCH ---
    print(f"Finalizing with Caption for top {k} results...")

    def filter_stage_2(metadata):
        if metadata['id'] not in ids_stage_2:
            return False
        if exclude_platform_id and metadata['platform_id'] == exclude_platform_id:
            return False
        return True

    # Search very deep to find the specific items surviving stage 2
    docs_final = FAISS_DB_CACHE["caption_index"].similarity_search(
        q_caption,
        k=k,
        filter=filter_stage_2,
        fetch_k=len(ids_stage_2) * 1000
    )
    return docs_final


def caption_only_retrieval(query_caption: str, k: int = 3,
                           exclude_platform_id: str = None):
    """Flat caption-index search (no title/abstract filtering).

    Used as a baseline to compare against hierarchical_retrieval().
    """
    load_faiss_indices()
    filter_fn = None
    if exclude_platform_id:
        filter_fn = lambda m: m['platform_id'] != exclude_platform_id
    if filter_fn:
        return FAISS_DB_CACHE["caption_index"].similarity_search(
            query_caption, k=k, filter=filter_fn, fetch_k=k * 1000
        )
    else:
        return FAISS_DB_CACHE["caption_index"].similarity_search(
            query_caption, k=k
        )

def main():
    # Example Usage
    q_title = "Transformers for image recognition"
    q_abstract = "This paper proposes a new attention mechanism for vision tasks."
    q_caption = "Figure 1: The architecture of the Vision Transformer."

    results = hierarchical_retrieval(q_title, q_abstract, q_caption, t1=100, t2=30, k=3)

    print("\n=== Final Retrieved Figures ===")
    print(len(results))
    for i, doc in enumerate(results):
        print(f"Rank {i+1}: {doc.metadata['title']}")
        path = os.path.join(FIG_RAG_DIR, doc.metadata['figure_path'])
        print(f"Path: {path}")
        img = Image.open(path)
        img.show()
        print(f"Caption: {doc.page_content[:100]}...\n")

if __name__ == "__main__":
    main()