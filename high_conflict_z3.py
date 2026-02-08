import json
import csv
import sys

# --- 1. Z3 ENVIRONMENT CHECK ---
try:
    from z3 import *
    _ = Bool('test')
except (ImportError, NameError, AttributeError):
    print("CRITICAL ERROR: 'z3-solver' is not installed.")
    print("Please run: pip install z3-solver")
    sys.exit(1)

def solve_moral_situation_split_stream(post_data):
    """
    Solves a single post using Split-Stream Confidence Logic.
    
    Quality Vector Mapping:
    [0] Justification Strength      -> Logic Stream
    [1] Ethical Principle           -> Ethic Stream
    [2] Deliberative Quality        -> Logic Stream
    [3] Fairness of Blame           -> Ethic Stream
    [4] Non-Biased Language         -> Multiplier (Global Filter)
    """
    
    # Global Variables
    harm = Bool('harm')
    intent = Bool('intent')
    empathy = Bool('empathy')
    apology = Bool('apology')

    opt = Optimize()
    
    comments = post_data.get('processed_comments', [])
    if not comments: 
        return "No Data"
        
    for comment in comments:
        # 1. Content Vector [Harm, Intent, Empathy, Apology]
        c_vec = comment.get('comment_content_vector', [0, 0, 0, 0])
        
        # 2. Quality Vector [Justif, Ethic, Delib, Fairness, NonBias]
        q_vec = comment.get('comment_quality_vector', [1, 1, 1, 1, 1])
        
        # Pad if short (handle legacy data)
        if len(q_vec) < 5:
            q_vec = q_vec + [1] * (5 - len(q_vec))
            
        # Map Variables
        justif     = q_vec[0]
        ethic      = q_vec[1]
        delib      = q_vec[2]
        fair_blame = q_vec[3]
        non_biased = q_vec[4] # Multiplier
        
        # --- SPLIT STREAM WEIGHTS ---
        
        # Logic Stream (Head): Validates Facts (Harm/Intent)
        # Weight = (Justification + Deliberative) * Non_Biased
        w_logic = int((justif + delib) * non_biased)
        
        # Ethic Stream (Heart): Validates Character (Empathy/Apology)
        # Weight = (Ethical_Principle + Fairness_Blame) * Non_Biased
        w_ethic = int((ethic + fair_blame) * non_biased)
        
        # --- APPLY CONSTRAINTS ---
        
        # Logic Constraints
        if c_vec[0] == 1: opt.add_soft(harm, w_logic)
        else:             opt.add_soft(Not(harm), w_logic)
            
        if c_vec[1] == 1: opt.add_soft(intent, w_logic)
        else:             opt.add_soft(Not(intent), w_logic)
            
        # Ethic Constraints
        if c_vec[2] == 1: opt.add_soft(empathy, w_ethic)
        else:             opt.add_soft(Not(empathy), w_ethic)
            
        if c_vec[3] == 1: opt.add_soft(apology, w_ethic)
        else:             opt.add_soft(Not(apology), w_ethic)

    # --- SOLVE ---
    if opt.check() == sat:
        m = opt.model()
        is_harm = is_true(m[harm])
        is_intent = is_true(m[intent])
        is_empathy = is_true(m[empathy])
        is_apology = is_true(m[apology])
        
        # --- FINAL RULES ---
        if not is_harm:
            return "NTA"
        elif is_harm and is_intent:
            return "YTA"
        elif is_harm and not is_intent:
            if is_empathy or is_apology:
                return "NAH"
            else:
                return "ESH"


# --- MAIN EXECUTION ---
# vector_file = 'processed_comments_high_conflict.json'  # Has Vectors
vector_file = 'processed_comments_800_new_posts.json'  # Has Vectors

# gt_file     = 'aita_200_high_conflict_posts_50_comments.json' # Has Ground Truth
gt_file     = 'aita_800_new_posts.json' # Has Ground Truth

output_csv  = 'z3_verdicts.csv'

print(f"Vectors Source:      {vector_file}")
print(f"Ground Truth Source: {gt_file}")
print(f"Output Target:       {output_csv}\n")

try:
    # 1. Load Ground Truth Mapping
    # Creates a dictionary: { "post_id": "YTA", ... }
    print("Loading Ground Truth labels...")
    gt_map = {}
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f) # This file is a standard JSON List
        for post in gt_data:
            pid = post.get('post_id')
            label = post.get('ground_truth_label', 'Unknown')
            if pid:
                gt_map[pid] = label
    print(f"-> Loaded {len(gt_map)} ground truth labels.")

    # 2. Open Output CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['post_id', 'title', 'z3_verdict', 'ground_truth_label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 3. Process Vector File (JSONL) and Solve
        print("Processing Z3 vectors...")
        count = 0
        with open(vector_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        post = json.loads(line)
                        post_id = post.get('post_id', 'Unknown')
                        
                        # Run Solver
                        verdict = solve_moral_situation_split_stream(post)
                        
                        # Look up Ground Truth
                        gt_label = gt_map.get(post_id, "Unknown")
                        
                        # Write Row
                        writer.writerow({
                            'post_id': post_id,
                            'title': post.get('title', 'No Title'),
                            'z3_verdict': verdict,
                            'ground_truth_label': gt_label
                        })
                        count += 1
                        
                        # Optional: Print progress every 20 posts
                        if count % 20 == 0:
                            print(f"  Processed {count} posts...")
                            
                    except json.JSONDecodeError:
                        continue

    print("-" * 40)
    print(f"Success! Processed {count} posts.")
    print(f"Final results saved to {output_csv}")

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")