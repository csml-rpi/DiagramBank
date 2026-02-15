import duckdb
import os
from tqdm import tqdm
from utils import DB_PATH, ICLR_DIR, get_clean_paper_text, extract_figure_contexts

def main():
    con = duckdb.connect(DB_PATH)

    # 1. Fetch Metadata (Join Papers & Figures)
    # We need 'decision' and 'year' from Papers to find the PDF, 
    # and 'figure_number' from Figures to know what to look for.
    print("Fetching metadata from DuckDB...")
    query = f"""
        SELECT 
            p.platform_id, 
            p.year, 
            -- Slugify decision to match file path structure
            TRIM(
                REGEXP_REPLACE(
                    LOWER(decision), 
                    '[^a-z0-9]+', 
                    '_', 
                    'g'
                ), 
                '_'
            ) AS clean_decision,
            f.figure_number
        FROM Papers_Staging p, Figures f
        WHERE p.platform_id = f.platform_id
    """
    rows = con.execute(query).fetchall()

    # 2. Group by Paper (to open each PDF only once)
    # Structure: paper_map = { pid: { 'path_info': (year, decision), 'figs': [fig_num1, fig_num2...] } }
    paper_map = {}
    for pid, year, decision, fnum in rows:
        if pid not in paper_map:
            paper_map[pid] = {'year': year, 'decision': decision, 'figs': []}
        paper_map[pid]['figs'].append(str(fnum))

    # 3. Process Papers
    updates = [] # List of tuples: (new_context, pid, fnum)
    PAPERS_BASE = os.path.join(ICLR_DIR, "papers")

    print(f"Processing {len(paper_map)} papers...")
    
    for pid, data in tqdm(paper_map.items()):
        year = data['year']
        decision = data['decision']
        target_figs = data['figs']
        
        # Reconstruct PDF Path (decision/year can change; use a fallback search)
        pdf_path = os.path.join(PAPERS_BASE, str(year), decision, pid, "main.pdf")

        if not os.path.exists(pdf_path):
            continue
            
        # Extract Text
        try:
            full_text = get_clean_paper_text(pdf_path)
        except Exception as e:
            # print(f"Error reading PDF {pid}: {e}")
            continue
            
        if not full_text:
            continue

        # Extract Contexts using your utility
        # Returns dict: { '1': ['paragraph text...', ...], '2': ... }
        context_map = extract_figure_contexts(full_text, target_figs)
        
        # Format for DB
        for fnum in target_figs:
            paragraphs = context_map.get(fnum, [])
            if paragraphs:
                formatted_context = ""
                for i, para in enumerate(paragraphs, 1):
                    formatted_context += f"<paragraph_{i}>{para}</paragraph_{i}>\n"
                
                updates.append((formatted_context.strip(), pid, fnum))

    # 4. Batch Update DuckDB
    if updates:
        print(f"Updating {len(updates)} figure contexts in DuckDB...")
        con.execute("BEGIN TRANSACTION")
        try:
            con.executemany("""
                UPDATE Figures 
                SET figure_context = ? 
                WHERE platform_id = ? AND figure_number = ?
            """, updates)
            con.execute("COMMIT")
            print("Update committed successfully.")
        except Exception as e:
            con.execute("ROLLBACK")
            print(f"CRITICAL ERROR: Failed to commit updates. {e}")
    else:
        print("No contexts extracted to update.")

    con.close()

if __name__ == "__main__":
    main()