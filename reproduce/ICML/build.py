import duckdb

from utils import DB_PATH

PAPERS_JSONL = "icml-papers.jsonl"

PENDING_VALUE = "pending/unknown"


def main():
    con = duckdb.connect(DB_PATH)
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS Papers AS 
        SELECT * FROM read_json_auto('{PAPERS_JSONL}')
    """)
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS papers_index ON Papers(platform_id)")
    print("Papers table created.")

    print("Loading staging data...")
    con.execute(
        f"CREATE OR REPLACE TABLE Papers_Staging AS SELECT * FROM read_json_auto('{PAPERS_JSONL}')"
    )

    # Insert new tuples (platform_id not already present in Papers)
    inserted = con.execute(
        """
        SELECT COUNT(*)
        FROM Papers_Staging s
        WHERE NOT EXISTS (
            SELECT 1 FROM Papers p WHERE p.platform_id = s.platform_id
        )
        """
    ).fetchone()[0]

    con.execute(
        """
        INSERT INTO Papers
        SELECT s.*
        FROM Papers_Staging s
        WHERE NOT EXISTS (
            SELECT 1 FROM Papers p WHERE p.platform_id = s.platform_id
        )
        """
    )

    # Count how many decisions will change (compute BEFORE the UPDATE).
    changed = con.execute(
        f"""
        SELECT COUNT(*)
        FROM Papers p
        JOIN Papers_Staging s USING (platform_id)
        WHERE LOWER(s.decision) <> '{PENDING_VALUE}'
          AND LOWER(p.decision) <> LOWER(s.decision)
        """
    ).fetchone()[0]

    # Decision-only update.
    con.execute(
        f"""
        UPDATE Papers p
        SET decision = s.decision
        FROM Papers_Staging s
        WHERE p.platform_id = s.platform_id
          AND LOWER(s.decision) <> '{PENDING_VALUE}'
          AND LOWER(p.decision) <> LOWER(s.decision)
        """
    )

    print(f"DB sync complete. Inserted {inserted} new papers; updated decision for {changed} papers.")
    con.close()


if __name__ == "__main__":
    main()
