import openreview
import json
import argparse
import time
from tqdm import tqdm

from utils import setup_clients, process_paper

def main(limit=None):
    client_v1, client_v2 = setup_clients()
    
    output_file = "tmlr-papers.jsonl"
    TMLR_urls = {}
    TMLR_replies = {}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        total_papers = 0
        total_elapsed = 0
        
        time1 = time.time()
        client = client_v2

        inv_id = "TMLR/-/Submission"
        notes = []
        for _ in range(10):
            try:
                if limit:
                    notes = client.get_notes(invitation=inv_id, limit=limit, details='directReplies')
                else:
                    notes = client.get_all_notes(invitation=inv_id, details='directReplies')
                break # Success, exit retry loop
            except Exception as e:
                print(f"  Error fetching {inv_id}: {e}")
                print("  Retrying in 30 seconds...")
                time.sleep(30)
        # --- RETRY LOGIC END ---
                
        count = len(notes)
        print(f"  Found {count} papers. Extracting data...")
        total_papers += count

        for note in notes:
            try:
                paper_json = process_paper(note)
                url = paper_json['url']
                id = paper_json['platform_id']
                if id not in TMLR_urls: # avoid duplicates across invitations
                    f.write(json.dumps(paper_json) + '\n')
                    TMLR_urls[id] = url
                    TMLR_replies[id] = note.details.get('directReplies', [])
            except Exception as e:
                print(f"  Error processing paper {note.id}: {e}")
        

    time2 = time.time()
    elapsed = time2-time1
    total_elapsed += elapsed
    
    print(f"Total papers processed: {total_papers}.")
    print(f"Unique papers: {len(TMLR_urls)}")
    print(f"Download time: {total_elapsed:.0f} seconds.")
    # Dump the URL map
    with open("tmlr-urls.json", 'w') as f:
        json.dump(TMLR_urls, f, indent=4)

    with open("tmlr-replies.json", 'w') as f:
        json.dump(TMLR_replies, f, indent=4)
    
    print(f"\nDone! Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape TMLR papers.")
    parser.add_argument('--limit', type=int, required=False, help="Limit number of papers per invitation for testing")
    args = parser.parse_args()
    
    main(limit=args.limit)