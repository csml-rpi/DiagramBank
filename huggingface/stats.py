import os
from pathlib import Path

# --- CONFIGURATION ---
# Assumes the script is running in the root directory alongside 'OpenReview'
ROOT_DIR = Path( os.path.join(os.environ.get("FIG_RAG_DIR"), "OpenReview") )

def get_dir_size(path):
    """Calculates the total size of a directory in bytes."""
    total = 0
    for p in path.rglob('*'):
        if p.is_file():
            total += p.stat().st_size
    return total

def format_size(size_bytes):
    """Converts bytes to human readable string (GB, MB)."""
    if size_bytes == 0: return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(os.math.floor(os.math.log(size_bytes, 1024)))
    p = os.math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])

def is_accepted(decision_name):
    """
    Applies the exact SQL logic provided:
    (lower(decision) like '%accept%' OR '%spotlight%' OR '%poster%' OR '%oral%')
    """
    name = decision_name.lower()
    keywords = ["accept", "spotlight", "poster", "oral"]
    return any(k in name for k in keywords)

def main():
    if not ROOT_DIR.exists():
        print(f"❌ Error: Could not find '{ROOT_DIR}'. Are you in the right folder?")
        return

    stats = [] # Store (Conf, Year, Decision, Size, Category)
    
    # Iterate through the structure
    # OpenReview -> Conference -> figures -> Year -> Decision
    for conf in ROOT_DIR.iterdir():
        if not conf.is_dir(): continue
        
        figures_path = conf / "figures"
        if not figures_path.exists(): continue
        
        for year in figures_path.iterdir():
            if not year.is_dir(): continue
            
            for decision in year.iterdir():
                if not decision.is_dir(): continue
                
                # Calculate size
                size = get_dir_size(decision)
                
                # Categorize
                category = "Accept" if is_accepted(decision.name) else "Reject"
                
                stats.append({
                    "conf": conf.name,
                    "year": year.name,
                    "decision": decision.name,
                    "size": size,
                    "category": category
                })

    # --- AGGREGATE TOTALS ---
    total_accept = sum(s['size'] for s in stats if s['category'] == "Accept")
    total_reject = sum(s['size'] for s in stats if s['category'] == "Reject")
    total_all = total_accept + total_reject

    # --- PRINT DETAILED TABLE ---
    print(f"{'CONFERENCE':<10} | {'YEAR':<6} | {'DECISION TYPE':<45} | {'CATEGORY':<8} | {'SIZE':<10}")
    print("-" * 90)
    
    # Sort by Conference, then Year
    stats.sort(key=lambda x: (x['conf'], x['year'], x['decision']))
    
    for s in stats:
        size_str = format_size(s['size'])
        print(f"{s['conf']:<10} | {s['year']:<6} | {s['decision']:<45} | {s['category']:<8} | {size_str:<10}")

    # --- PRINT SUMMARY ---
    print("\n" + "="*40)
    print("       📊 DATASET SIZE SUMMARY       ")
    print("="*40)
    print(f" ✅ ACCEPTED PAPERS:  {format_size(total_accept):>10}")
    print(f" ❌ REJECTED PAPERS:  {format_size(total_reject):>10}")
    print("-" * 40)
    print(f" 📦 TOTAL DATASET:    {format_size(total_all):>10}")
    print("="*40)

if __name__ == "__main__":
    import math # Import math here for the format_size function
    os.math = math
    main()