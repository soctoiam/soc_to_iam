import json
import random
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any, Optional

# --- KONFIGURATION ---
INPUT = Path("/kaggle/input/artifacts-phase2/extraction_train.jsonl")
OUTPUT = Path("/kaggle/working/artifacts_phase2/extraction_train_100k_final_clean.jsonl")

TOTAL_TARGET = 100_000
SEED = 42

# 1. CLUSTER-BOMBEN ENTSCHÄRFUNG (NEU!)
# Wir verwerfen jedes Beispiel, das mehr als 50 Komponenten hat.
# Das killt die 5000er HP-Listen und die 1000er Qualcomm-Listen.
MAX_COMPONENTS_PER_CVE = 50

# 2. BIAS CONTROL FÜR TARGET SW (WordPress Bias)
MAX_SAME_TARGET_SW_NAME = 600  

# 3. VENDOR CAPS
# Wir können die Caps jetzt etwas lockern, da die "fetten" CVEs wegfallen
MAX_PER_VENDOR_WITH_SW = 10_000 
MAX_PER_VENDOR_NO_SW = 4_000   # Leicht erhöht, um 100k voll zu kriegen

# FILTER
INVALID_VALUES = {"*", "-", "n/a", "none", "unknown", "null", "any"}

def extract_year(cve_id: str) -> Optional[int]:
    try:
        parts = cve_id.split("-")
        if len(parts) >= 2: return int(parts[1])
    except: pass
    return None

def is_valid_value(val: str) -> bool:
    if not val: return False
    if len(val) < 2: return False
    if val in INVALID_VALUES: return False
    return True

def parse_components_from_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not messages or len(messages) < 3: return []
    assistant_content = messages[2].get("content", "")
    try:
        obj = json.loads(assistant_content)
        comps = obj.get("components", [])
        if isinstance(comps, list): return comps
    except: pass
    return []

def load_and_filter_data(path: Path):
    rng = random.Random(SEED)
    examples = []
    
    # Statistik für verworfene "Bomben"
    dropped_bombs = 0
    dropped_bomb_vendors = Counter()
    
    print(f"[Load] Reading {path}...")
    
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            
            # Parsing
            msgs = rec.get("messages", [])
            comps = parse_components_from_messages(msgs)
            
            # --- CHECK 1: CLUSTER BOMB FILTER ---
            if len(comps) > MAX_COMPONENTS_PER_CVE:
                dropped_bombs += 1
                # Welcher Vendor war schuld? (nur für Statistik)
                if comps:
                    v = (comps[0].get("vendor") or "unknown").lower()
                    dropped_bomb_vendors[v] += 1
                continue # SKIP THIS EXAMPLE

            target_sws = []
            vendors = []
            
            for c in comps:
                v = (c.get("vendor") or "").strip().lower()
                tsw = (c.get("target_sw") or "").strip().lower()
                
                if is_valid_value(v): vendors.append(v)
                if is_valid_value(tsw): target_sws.append(tsw)
            
            rec["_target_sws"] = target_sws
            rec["_primary_vendor"] = vendors[0] if vendors else "unknown"
            rec["_has_target_sw"] = (len(target_sws) > 0)
            
            examples.append(rec)
            
    rng.shuffle(examples)
    
    print("\n" + "="*40)
    print("CLUSTER BOMB REMOVAL REPORT")
    print("="*40)
    print(f"Total Examples Dropped (> {MAX_COMPONENTS_PER_CVE} components): {dropped_bombs}")
    print("Top 5 Vendors banned due to size:")
    for v, c in dropped_bomb_vendors.most_common(5):
        print(f"  {v}: {c} examples dropped")
    print("-" * 40 + "\n")
    
    return examples

def sample_balanced_positives(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = []
    sw_name_counter = Counter()
    vendor_counter = Counter()
    
    rng = random.Random(SEED)
    rng.shuffle(examples)
    
    for rec in examples:
        v = rec["_primary_vendor"]
        if vendor_counter[v] >= MAX_PER_VENDOR_WITH_SW: continue
            
        over_limit = False
        for tsw in rec["_target_sws"]:
            if sw_name_counter[tsw] >= MAX_SAME_TARGET_SW_NAME:
                over_limit = True
                break
        
        if over_limit: continue
            
        selected.append(rec)
        vendor_counter[v] += 1
        for tsw in rec["_target_sws"]:
            sw_name_counter[tsw] += 1
            
    return selected

def sample_balanced_negatives(examples: List[Dict[str, Any]], n_needed: int) -> List[Dict[str, Any]]:
    selected = []
    vendor_counter = Counter()
    seen_cves = set()
    
    rng = random.Random(SEED)
    rng.shuffle(examples)
    
    for rec in examples:
        if len(selected) >= n_needed: break
            
        v = rec["_primary_vendor"]
        cve = rec.get("cve_id")
        
        if cve in seen_cves: continue
        if vendor_counter[v] >= MAX_PER_VENDOR_NO_SW: continue
            
        selected.append(rec)
        vendor_counter[v] += 1
        seen_cves.add(cve)
        
    return selected

def main():
    # 1. Laden & Filtern (Bombenentschärfung)
    all_data = load_and_filter_data(INPUT)
    
    # 2. Trennen
    with_sw = [r for r in all_data if r["_has_target_sw"]]
    without_sw = [r for r in all_data if not r["_has_target_sw"]]
    
    print(f"[Stats] Clean Positives available: {len(with_sw)}")
    print(f"[Stats] Clean Negatives available: {len(without_sw)}")
    
    # 3. Sampling Positives (Anti-WordPress)
    print(f"[Step 1] Sampling Positives...")
    selected_sw = sample_balanced_positives(with_sw)
    print(f"         -> Selected {len(selected_sw)} positive examples.")
    
    # 4. Sampling Negatives (Rest auffüllen)
    needed = TOTAL_TARGET - len(selected_sw)
    print(f"[Step 2] Filling remainder with {needed} Negatives...")
    selected_nosw = sample_balanced_negatives(without_sw, needed)
    print(f"         -> Selected {len(selected_nosw)} negative examples.")
    
    # 5. Merge & Save
    final_list = selected_sw + selected_nosw
    random.shuffle(final_list)
    
    # Cleanup
    for rec in final_list:
        for k in ["_target_sws", "_primary_vendor", "_has_target_sw"]:
            rec.pop(k, None)
            
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for rec in final_list:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print(f"\n[Done] Saved {len(final_list)} CLEAN & BALANCED examples to {OUTPUT}")
    
    # --- FINAL CHECK ---
    print("\n=== Final Training Set Composition ===")
    tsw_counts = Counter()
    vendor_counts = Counter()
    
    for rec in final_list:
        msgs = rec.get("messages", [])
        comps = parse_components_from_messages(msgs)
        found_sw = False
        vendors = set()
        for c in comps:
            v = (c.get("vendor") or "").strip().lower()
            if v: vendors.add(v)
            tsw = (c.get("target_sw") or "").strip().lower()
            if is_valid_value(tsw):
                tsw_counts[tsw] += 1
                found_sw = True
        
        for v in vendors: vendor_counts[v] += 1
        
    print("\nTop 10 Target Softwares:")
    for k, v in tsw_counts.most_common(10):
        print(f"  {k:20s}: {v}")
        
    print("\nTop 10 Vendors:")
    for k, v in vendor_counts.most_common(10):
        print(f"  {k:20s}: {v}")

if __name__ == "__main__":
    main()