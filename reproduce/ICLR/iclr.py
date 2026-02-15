import openreview
import json
import argparse
import time
from tqdm import tqdm

from utils import setup_clients, process_paper

"""
Papers(platform, venue, year, title, abstract, keywords, areas, tldr, scores, decision, authors, author_ids, cdate, url, platform_id)
platform: OpenReview
venue: ICLR
year: venue year
title: paper title
abstract: paper abstract
keywords: comma-separated keywords
areas: primary subject areas
tldr: "Too Long; Didn’t Read" (one-sentence summary)
scores: a list of reviewer scores
decision: paper decision
authors: comma-separated authors
author_ids: comma-separated author ids on the platform
cdate: creation date (YYYYMMDD)
url: direct link to the source platform
platform_id: unique paper id given by a platform
bibtex: paper bibtex
"""

def get_invitations_map():
    """
    Returns the map of invitation IDs verified by the discovery step.
    Structure: Year -> List of tuples (InvitationID, DefaultDecision)
    """
    return {
        2017: [('ICLR.cc/2017/conference/-/submission', None)],
        2018: [
            ('ICLR.cc/2018/Conference/-/Withdrawn_Submission', 'withdrawn'),
            ('ICLR.cc/2018/Conference/-/Desk_Rejected_Submission', 'desk reject'),
            ('ICLR.cc/2018/Conference/-/Blind_Submission', None),
        ],
        2019: [
            ('ICLR.cc/2019/Conference/-/Withdrawn_Submission', 'withdrawn'),
            ('ICLR.cc/2019/Conference/-/Desk_Rejected_Submission', 'desk reject'),
            ('ICLR.cc/2019/Conference/-/Blind_Submission', None),
        ],
        2020: [
            ('ICLR.cc/2020/Conference/-/Withdrawn_Submission', 'withdrawn'),
            ('ICLR.cc/2020/Conference/-/Desk_Rejected_Submission', 'desk reject'),
            ('ICLR.cc/2020/Conference/-/Blind_Submission', None),
        ],
        2021: [
            ('ICLR.cc/2021/Conference/-/Withdrawn_Submission', 'withdrawn'),
            ('ICLR.cc/2021/Conference/-/Desk_Rejected_Submission', 'desk reject'),
            ('ICLR.cc/2021/Conference/-/Blind_Submission', None),
        ],
        2022: [
            ('ICLR.cc/2022/Conference/-/Withdrawn_Submission', 'withdrawn'),
            ('ICLR.cc/2022/Conference/-/Desk_Rejected_Submission', 'desk reject'),
            ('ICLR.cc/2022/Conference/-/Blind_Submission', None),
        ],
        2023: [
            ('ICLR.cc/2023/Conference/-/Withdrawn_Submission', 'withdrawn'),
            ('ICLR.cc/2023/Conference/-/Desk_Rejected_Submission', 'desk reject'),
            ('ICLR.cc/2023/Conference/-/Blind_Submission', None),
        ],
        2024: [
            ('ICLR.cc/2024/Conference/-/Withdrawn_Submission', 'withdrawn'),
            ('ICLR.cc/2024/Conference/-/Desk_Rejected_Submission', 'desk reject'),
            ('ICLR.cc/2024/Conference/-/Submission', None),
        ],
        2025: [
            ('ICLR.cc/2025/Conference/-/Withdrawn_Submission', 'withdrawn'),
            ('ICLR.cc/2025/Conference/-/Desk_Rejected_Submission', 'desk reject'),
            ('ICLR.cc/2025/Conference/-/Submission', None),
        ],
        2026: [
            ('ICLR.cc/2026/Conference/-/Withdrawn_Submission', 'withdrawn'),
            ('ICLR.cc/2026/Conference/-/Desk_Rejected_Submission', 'desk reject'),
            ('ICLR.cc/2026/Conference/-/Submission', None),
        ]
    }

def main(limit=None):
    client_v1, client_v2 = setup_clients()
    invitations_map = get_invitations_map()
    
    output_file = "iclr-papers.jsonl"
    iclr_urls = {}
    iclr_replies = {}
    
    # Using 'a' (append) so we can watch it populate, or 'w' to overwrite
    with open(output_file, 'w', encoding='utf-8') as f:
        total_papers = 0
        total_elapsed = 0
        # Iterate over years 2017-2026
        for year in tqdm(range(2017, 2027)):
            time1 = time.time()
            api_version = 1 if year < 2024 else 2
            client = client_v1 if api_version == 1 else client_v2
            
            # Get list of invitations (Main, withdrawn, DeskReject)
            invitation_configs = invitations_map.get(year, [])
            
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
                        print("  Retrying in 60 seconds...")
                        time.sleep(60)
                # --- RETRY LOGIC END ---
                        
                count = len(notes)
                print(f"  Found {count} papers. Extracting data...")
                total_papers += count

                for note in notes:
                    try:
                        paper_json = process_paper(note, year, api_version, default_dec)
                        url = paper_json['url']
                        id = paper_json['platform_id']
                        if id not in iclr_urls: # avoid duplicates across invitations
                            f.write(json.dumps(paper_json) + '\n')
                            iclr_urls[id] = url
                            iclr_replies[id] = note.details.get('directReplies', [])
                    except Exception as e:
                        print(f"  Error processing paper {note.id}: {e}")
                
                # Rate limit safety between major calls
                time.sleep(30)

            time2 = time.time()
            elapsed = time2-time1
            total_elapsed += elapsed
            print(f"Downloading year {year} took {elapsed:.0f} seconds.")
    
    print(f"Total papers processed: {total_papers}.")
    print(f"Unique papers: {len(iclr_urls)}")
    print(f"Download time: {total_elapsed:.0f} seconds.")
    # Dump the URL map
    with open("iclr-urls.json", 'w') as f:
        json.dump(iclr_urls, f, indent=4)

    with open("iclr-replies.json", 'w') as f:
        json.dump(iclr_replies, f, indent=4)
    
    print(f"\nDone! Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape ICLR papers.")
    parser.add_argument('--limit', type=int, required=False, help="Limit number of papers per invitation for testing")
    args = parser.parse_args()
    
    main(limit=args.limit)