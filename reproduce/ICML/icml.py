import openreview
import json
import argparse
import time
from tqdm import tqdm

from utils import setup_clients, process_paper

def get_invitations_map():
    """
    Returns the map of invitation IDs verified by the discovery step.
    Structure: Year -> List of tuples (InvitationID, DefaultDecision)
    """
    return {
        2023: [
            ('ICML.cc/2023/Conference/-/Submission', None)
        ],
        2024: [
            ('ICML.cc/2024/Conference/-/Withdrawn_Submission', 'withdrawn'),
            ('ICML.cc/2024/Conference/-/Submission', None),
        ],
        2025: [
            ('ICML.cc/2025/Conference/-/Desk_Rejected_Submission', 'desk reject'),
            ('ICML.cc/2025/Conference/-/Submission', None),
        ]
    }

def main(limit=None):
    client_v1, client_v2 = setup_clients()
    invitations_map = get_invitations_map()
    
    output_file = "icml-papers.jsonl"
    ICML_urls = {}
    ICML_replies = {}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        total_papers = 0
        total_elapsed = 0
        
        years = list(invitations_map.keys())
        for year in tqdm(years):
            time1 = time.time()
            api_version = 2
            client = client_v2
            
            invitation_configs = invitations_map[year]
            
            for inv_id, default_dec in invitation_configs:
                print(f"Processing {year} [{api_version}] - {inv_id}...")
                
                # --- RETRY LOGIC START ---
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
                        paper_json = process_paper(note, year, api_version, default_dec)
                        url = paper_json['url']
                        id = paper_json['platform_id']
                        if id not in ICML_urls: # avoid duplicates across invitations
                            f.write(json.dumps(paper_json) + '\n')
                            ICML_urls[id] = url
                            ICML_replies[id] = note.details.get('directReplies', [])
                    except Exception as e:
                        print(f"  Error processing paper {note.id}: {e}")
                
                # Rate limit safety between major calls
                time.sleep(1)

            time2 = time.time()
            elapsed = time2-time1
            total_elapsed += elapsed
            print(f"Downloading year {year} took {elapsed:.0f} seconds.")
    
    print(f"Total papers processed: {total_papers}.")
    print(f"Unique papers: {len(ICML_urls)}")
    print(f"Download time: {total_elapsed:.0f} seconds.")
    # Dump the URL map
    with open("icml-urls.json", 'w') as f:
        json.dump(ICML_urls, f, indent=4)

    with open("icml-replies.json", 'w') as f:
        json.dump(ICML_replies, f, indent=4)
    
    print(f"\nDone! Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape ICML papers.")
    parser.add_argument('--limit', type=int, required=False, help="Limit number of papers per invitation for testing")
    args = parser.parse_args()
    
    main(limit=args.limit)