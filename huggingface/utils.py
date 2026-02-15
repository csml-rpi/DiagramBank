import os

# file directories
FIG_RAG_DIR = os.environ.get("FIG_RAG_DIR")
ICLR_DIR = os.path.join(FIG_RAG_DIR, "OpenReview", "ICLR") # scratch folder for data from ICLR
ICML_DIR = os.path.join(FIG_RAG_DIR, "OpenReview", "ICML") # scratch folder for data from ICML
NeurIPS_DIR = os.path.join(FIG_RAG_DIR, "OpenReview", "NeurIPS") # scratch folder for data from NeurIPS
TMLR_DIR = os.path.join(FIG_RAG_DIR, "OpenReview", "TMLR") # scratch folder for data from TMLR
FAISS_DIR = os.path.join(FIG_RAG_DIR, "faiss")

# database
ICLR_DB = os.path.join(ICLR_DIR, "research.db")
ICML_DB = os.path.join(ICML_DIR, "research.db")
NeurIPS_DB = os.path.join(NeurIPS_DIR, "research.db")
TMLR_DB = os.path.join(TMLR_DIR, "research.db")
DATA_DB = os.path.join(FAISS_DIR, "research.db")
