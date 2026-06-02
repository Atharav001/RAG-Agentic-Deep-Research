import json
import os

def validate_predictions(filepath):
    if not os.path.exists(filepath):
        print(f"❌ MISSING: {filepath}")
        return False
    
    ids_seen = set()
    errors = []
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"Line {line_num}: Invalid JSON")
                continue
                
            # Check 1: ID format
            q_id = data.get("id")
            if not q_id or not q_id.startswith("q"):
                errors.append(f"Line {line_num}: Missing/Bad ID")
                continue
                
            # Check 2: Duplicates
            if q_id in ids_seen:
                errors.append(f"Line {line_num}: Duplicate ID {q_id}")
            ids_seen.add(q_id)
            
            # Check 3: cited_papers format (No versions, no URLs)
            for paper in data.get("cited_papers", []):
                if "v" in paper and paper[-1].isdigit():
                    errors.append(f"Line {line_num} ({q_id}): Version suffix found in {paper}")
                if "arxiv.org" in paper:
                    errors.append(f"Line {line_num} ({q_id}): URL found in {paper}")
                    
    # Check 4: All 30 present
    expected_ids = {f"q{i:02d}" for i in range(1, 31)}
    missing = expected_ids - ids_seen
    if missing:
        errors.append(f"Missing question IDs: {missing}")

    if errors:
        print(f"❌ FAILED: {filepath}")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print(f"✅ PASSED: {filepath}")
        return True


print("Validating submissions...")
configs = ["full_agent", "baseline", "no_planner", "no_reranker",
           "no_reflector", "no_hybrid", "no_citation_verifier", "no_compressor"]
all_good = True
for config in configs:
    if not validate_predictions(f"predictions/{config}.jsonl"):
        all_good = False

if all_good:
    print("\n🎉 ALL FILES VALID. SAFE TO SUBMIT.")
else:
    print("\n🚨 FIX ERRORS ABOVE BEFORE SUBMITTING.")
