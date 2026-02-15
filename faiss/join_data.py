import duckdb
from utils import ICLR_DB, ICML_DB, NeurIPS_DB, TMLR_DB, DATA_DB, DATA_JSONL

THRESHOLD = 0.85
# --- JOIN RELATIONS ---
SQL = f"""
create or replace table Data as
select p.platform, p.venue, p.year, p.title, p.abstract, p.keywords, p.areas, p.tldr, p.scores, p.decision, p.authors, p.author_ids, p.cdate, p.url, p.platform_id, p.bibtex, f.figure_path, f.figure_number, f.figure_caption, f.figure_context, f.confidence
from Papers p, Figures f
where p.platform_id = f.platform_id and f.confidence > {THRESHOLD} and f.figure_type = 'diagram' and
-- keep only diagrams from accepted papers
(lower(p.decision) like '%accept%' or lower(p.decision) like '%spotlight%' or lower(p.decision) like '%poster%' or lower(p.decision) like '%oral%');
"""

def join_table(db_path):
    con = duckdb.connect(db_path)
    results = con.execute(SQL).fetchall()
    print(results)
    print(f"Constructed table Data at {db_path}")
    con.close()

def append_table(db_path):
    """
    Attaches the source database, creates the Data table in the target DB (if missing),
    and appends the new records.
    """
    con = duckdb.connect(DATA_DB)
    
    # 1. Attach the source database (e.g., ICLR.db) as 'source_db'
    con.execute(f"ATTACH '{db_path}' AS source_db")
    
    # 2. Create the destination table schema if it doesn't exist.
    # We use 'WHERE 1=0' to copy the column structure without copying data yet.
    con.execute("CREATE TABLE IF NOT EXISTS Data AS SELECT * FROM source_db.Data WHERE 1=0")
    
    # 3. Create Indices
    # Unique index ensures we don't insert the same figure twice (idempotency)
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_fig ON Data(platform_id, figure_number)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_venue ON Data(venue)")
    
    # 4. Insert data from the attached source_db into the main Data table
    # INSERT OR IGNORE skips rows that violate the unique index
    con.execute("INSERT OR IGNORE INTO Data SELECT * FROM source_db.Data")
    
    print(f"Appended data from {db_path} to {DATA_DB}")
    con.close()

def main():
    DB_PATHS = [ICLR_DB, ICML_DB, NeurIPS_DB, TMLR_DB]
    
    # 1. Process each database
    for db_path in DB_PATHS:
        # Create the temp 'Data' table inside the source DB
        join_table(db_path)
        # Move that data into the main DATA_DB
        append_table(db_path)
    
    # 2. Summary Statistics
    print("\n--- Summary Statistics ---")
    con = duckdb.connect(DATA_DB)
    
    # Query to count diagrams per venue
    stats_query = """
    SELECT venue, COUNT(*) as diagram_count 
    FROM Data 
    GROUP BY venue 
    ORDER BY diagram_count DESC
    """
    
    results = con.execute(stats_query).fetchall()
    
    print(f"{'Venue':<10} | {'Diagrams'}")
    print("-" * 25)
    for venue, count in results:
        print(f"{venue:<10} | {count}")

    sql = f"""
    copy Data to '{DATA_JSONL}' (format jsonl)
    """
    print(f"save joined data to {DATA_JSONL}")
    result = con.execute(sql).fetchone()
    print(result)
    con.close()

if __name__ == "__main__":
    main()
    """
    --- Summary Statistics ---
Venue      | Diagrams
-------------------------
ICLR       | 13241
NeurIPS    | 12177
ICML       | 7207
TMLR       | 3677"""