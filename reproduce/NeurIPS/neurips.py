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
        2021: [
            ('NeurIPS.cc/2021/Conference/-/Blind_Submission', None)
        ],
        2022: [
            ('NeurIPS.cc/2022/Conference/-/Blind_Submission', None)
        ],
        2024: [
            ('NeurIPS.cc/2024/Conference/-/Withdrawn_Submission', 'withdrawn'),
            ('NeurIPS.cc/2024/Conference/-/Desk_Rejected_Submission', 'desk reject'),
            ('NeurIPS.cc/2024/Conference/-/Submission', None),
        ],
        2025: [
            ('NeurIPS.cc/2025/Conference/-/Withdrawn_Submission', 'withdrawn'),
            ('NeurIPS.cc/2025/Conference/-/Desk_Rejected_Submission', 'desk reject'),
            ('NeurIPS.cc/2025/Conference/-/Submission', None),
        ]
    }

def main(limit=None):
    client_v1, client_v2 = setup_clients()
    invitations_map = get_invitations_map()
    
    output_file = "neurips-papers.jsonl"
    NeurIPS_urls = {}
    NeurIPS_replies = {}
    
    with open(output_file, 'w', encoding='utf-8') as f:
        total_papers = 0
        total_elapsed = 0
        
        years = list(invitations_map.keys())
        for year in tqdm(years):
            time1 = time.time()
            api_version = 1 if year < 2024 else 2
            client = client_v1 if api_version == 1 else client_v2
            
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
                        if id not in NeurIPS_urls: # avoid duplicates across invitations
                            f.write(json.dumps(paper_json) + '\n')
                            NeurIPS_urls[id] = url
                            NeurIPS_replies[id] = note.details.get('directReplies', [])
                    except Exception as e:
                        print(f"  Error processing paper {note.id}: {e}")
                
                # Rate limit safety between major calls
                time.sleep(1)

            time2 = time.time()
            elapsed = time2-time1
            total_elapsed += elapsed
            print(f"Downloading year {year} took {elapsed:.0f} seconds.")
    
    print(f"Total papers processed: {total_papers}.")
    print(f"Unique papers: {len(NeurIPS_urls)}")
    print(f"Download time: {total_elapsed:.0f} seconds.")
    # Dump the URL map
    with open("neurips-urls.json", 'w') as f:
        json.dump(NeurIPS_urls, f, indent=4)

    with open("neurips-replies.json", 'w') as f:
        json.dump(NeurIPS_replies, f, indent=4)
    
    print(f"\nDone! Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape NeurIPS papers.")
    parser.add_argument('--limit', type=int, required=False, help="Limit number of papers per invitation for testing")
    args = parser.parse_args()
    
    main(limit=args.limit)