import os
import json
import shutil
import duckdb
from tqdm import tqdm

from utils import run_pdffigures2, ICML_DIR, DB_PATH, FIG_RAG_DIR

# --- Configuration ---
# Papers source (for reading PDFs)
PAPERS_BASE = os.path.join(ICML_DIR, "papers")

# Figures destination
OUTPUT_BASE = os.path.join(ICML_DIR, "figures")

# Temporary holding ground for raw PDFFigures2 output
TEMP_DIR = os.path.join(OUTPUT_BASE, "temp-raw-output")
TEMP_FIGURES = os.path.join(TEMP_DIR, "figures")
TEMP_JSON = os.path.join(TEMP_DIR, "metadata")

def get_papers_metadata():
    """Get papers (pid, year, decision_slug) from Papers_Staging."""
    con = duckdb.connect(DB_PATH)
    rows = con.execute(
        """
        SELECT
            platform_id,
            year,
            TRIM(REGEXP_REPLACE(LOWER(decision), '[^a-z0-9]+', '_', 'g'), '_') AS clean_decision
        FROM Papers_Staging
        """
    ).fetchall()
    con.close()
    return rows

def init_db_table():
    """
    Ensure the Figures table exists in DuckDB.
    """
    con = duckdb.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS Figures (
            platform VARCHAR,
            platform_id VARCHAR,
            figure_number VARCHAR,
            figure_path VARCHAR,
            figure_caption VARCHAR,
            figure_context VARCHAR,
            figure_type VARCHAR,
            confidence FLOAT
        )
    """)
    con.close()

def create_index():
    """Unique index on (platform_id, figure_number) for idempotent upserts."""
    print("Creating indexes...")
    con = duckdb.connect(DB_PATH)
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS figures_index ON Figures(platform_id, figure_number)")
    con.close()

def main():
    # 1. Setup Directories & DB
    for d in [TEMP_DIR, TEMP_FIGURES, TEMP_JSON, OUTPUT_BASE]:
        os.makedirs(d, exist_ok=True)
    
    init_db_table()
    create_index()

    # 2. Get Paper List from DB
    print("Querying DuckDB for papers...")
    papers = get_papers_metadata()
    
    # Group papers by year/decision to process in batches (optimizes pdffigures2 calls)
    batch_map = {}
    
    for pid, year, decision in papers:
        # Construct path to the PDF we downloaded earlier
        pdf_path = os.path.join(PAPERS_BASE, str(year), decision, pid, "main.pdf")
        
        if os.path.exists(pdf_path):
            key = (year, decision)
            if key not in batch_map:
                batch_map[key] = []
            batch_map[key].append((pid, pdf_path))

    # 3. Process Batches
    con = duckdb.connect(DB_PATH) 
    
    print(f"Processing {len(papers)} papers across {len(batch_map)} batches...")
    
    for (year, decision), paper_list in tqdm(batch_map.items(), desc="Batches"):
        
        # Create temp input directory for this batch
        batch_input_dir = os.path.join(TEMP_DIR, "input", str(year), decision)
        os.makedirs(batch_input_dir, exist_ok=True)
        
        # Symlink PDFs to temp dir so pdffigures2 can scan them
        valid_pids = []
        for pid, src_pdf in paper_list:
            dst_link = os.path.join(batch_input_dir, f"{pid}.pdf")
            if not os.path.exists(dst_link):
                os.symlink(src_pdf, dst_link)
            valid_pids.append(pid)

        # Output dirs for this batch's raw extraction
        batch_json_dir = os.path.join(TEMP_JSON, str(year), decision)
        batch_fig_dir = os.path.join(TEMP_FIGURES, str(year), decision)
        os.makedirs(batch_json_dir, exist_ok=True)
        os.makedirs(batch_fig_dir, exist_ok=True)

        # Run Java Tool
        run_pdffigures2(batch_input_dir, batch_json_dir, batch_fig_dir)

        # Process Results
        db_entries = []
        
        for pid in valid_pids:
            json_file = os.path.join(batch_json_dir, f"{pid}.json")
            
            if not os.path.exists(json_file):
                continue

            try:
                with open(json_file, 'r') as f:
                    figures_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping corrupt JSON for {pid}")
                continue

            # Final Destination Directory
            final_figure_dir = os.path.join(OUTPUT_BASE, str(year), decision, pid)
            os.makedirs(final_figure_dir, exist_ok=True)

            for fig in figures_data:
                if fig["figType"] == "Figure": 
                    src_img_path = fig["renderURL"]
                    
                    # New filename: Figure{Number}.png
                    fig_num = fig["name"]
                    dest_filename = f"Figure{fig_num}.png"
                    dest_img_path = os.path.join(final_figure_dir, dest_filename)

                    # Move file if it exists
                    if os.path.exists(src_img_path):
                        shutil.move(src_img_path, dest_img_path)
                        
                        # Stores: platform/venue/figures/year/decision/platform_id/{figure_number}.png
                        relative_figure_path = os.path.relpath(dest_img_path, start=FIG_RAG_DIR)

                        db_entries.append((
                            'OpenReview',          # platform
                            pid,                   # platform_id
                            fig_num,               # figure_number
                            relative_figure_path,  # figure_path
                            fig['caption'],        # figure_caption
                            "",                    # figure_context (placeholder)
                            "",                    # figure_type (placeholder)
                            0.0                    # confidence (placeholder)
                        ))

        # Bulk Insert into DuckDB
        if db_entries:
            con.executemany("INSERT OR REPLACE INTO Figures VALUES (?, ?, ?, ?, ?, ?, ?, ?)", db_entries)

        # Cleanup Batch Input (symlinks)
        shutil.rmtree(batch_input_dir)

    con.close()

    create_index()
    
    # # Final Cleanup
    # try:
    #    shutil.rmtree(TEMP_DIR)
    #    print("Temporary files cleaned up.")
    # except:
    #    print("Warning: Could not remove temp directory.")

if __name__ == "__main__":
    main()