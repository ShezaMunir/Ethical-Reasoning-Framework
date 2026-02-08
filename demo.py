import json
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
        
    print(f"   > Analyzing {len(comments)} comments...")

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
        w_logic = int((justif + delib) * non_biased)
        
        # Ethic Stream (Heart): Validates Character (Empathy/Apology)
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
        
        # Debug print to see what Z3 decided internally
        print(f"   > Z3 Internal State: Harm={is_harm}, Intent={is_intent}, Empathy={is_empathy}, Apology={is_apology}")

        # --- FINAL RULES ---
        if is_harm and is_intent:
            return "YTA"
        elif not is_harm:
            return "NAH"  
        elif is_harm and not is_intent:
            if is_empathy or is_apology:
                return "NTA" 
            else:
                return "ESH"
                
    return "Unclear"

# --- MAIN EXECUTION ---
input_file = 'demo_eg1.json'

print("="*50)
print(f"Running Z3 Demo on: {input_file}")
print("="*50)

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        # Load the single datapoint
        # Assuming the file contains a single JSON object (dict)
        # If it's a list with one item, we handle that too.
        data = json.load(f)
        
        if isinstance(data, list):
            post = data[0]
        else:
            post = data

        title = post.get('title', 'No Title')
        post_id = post.get('post_id', 'Unknown')
        ground_truth = post.get('reddit_flair', post.get('label', 'Unknown'))

        print(f"Post ID: {post_id}")
        print(f"Title:   {title}")
        print("-" * 50)
        
        # Run Solver
        z3_verdict = solve_moral_situation_split_stream(post)
        
        print("-" * 50)
        print(f"Reddit Flair Verdict:  {ground_truth}")
        print(f"Z3 Solver Verdict:     {z3_verdict}")
        print("="*50)

except FileNotFoundError:
    print(f"Error: Could not find '{input_file}'. Make sure it exists.")
except json.JSONDecodeError:
    print(f"Error: '{input_file}' is not a valid JSON file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")