import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pandas as pd


# ===============================
#   Konfiguration
# ===============================

EXTRACTION_SYSTEM_PROMPT = """You are an information extraction assistant.
Given a vulnerability description, extract all vulnerable software components.
Return a JSON object with a single field "components", which is a list of objects.
Each component object MUST have the following fields:
- part (one of: a, o, h, * )
- vendor
- product
- target_sw (may be "*" if not specified)
- versionStartIncluding (string or empty)
- versionStartExcluding (string or empty)
- versionEndIncluding (string or empty)
- versionEndExcluding (string or empty)

Do NOT include any other fields.
Return ONLY valid JSON. No markdown, no commentary.
"""

# Raw inputs (from Phase 1)
PHASE1_DIR = Path("/kaggle/working/artifacts_phase1")
CPE_META_PATH = PHASE1_DIR / "cpe_meta.parquet"
GOLD_JSONL_PATH = PHASE1_DIR / "gold_by_cve.jsonl"

# Outputs (Phase 2)
PHASE2_DIR = Path("/kaggle/working/artifacts_phase2")
TRAIN_OUT_PATH = PHASE2_DIR / "extraction_train.jsonl"
EVAL_OUT_PATH = PHASE2_DIR / "extraction_eval.jsonl"

# Sampling / split
MAX_TRAIN_EXAMPLES = 100_000
EVAL_SIZE = 300
RANDOM_SEED = 42


# ===============================
#   Utilities
# ===============================

def load_cpe_meta(cpe_meta_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load cpe_meta.parquet and return lookup dict: cpe_uri -> {part, vendor, product, target_sw}
    """
    df = pd.read_parquet(cpe_meta_path)

    required_cols = {"cpe_uri", "part", "vendor", "product", "target_sw"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"cpe_meta.parquet is missing required columns: {missing}")

    df["cpe_uri"] = df["cpe_uri"].astype(str)

    lookup: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        uri = row["cpe_uri"]
        rec = {
            "part": str(row["part"]).lower().strip() if pd.notna(row["part"]) else "*",
            "vendor": str(row["vendor"]).lower().strip() if pd.notna(row["vendor"]) else "",
            "product": str(row["product"]).lower().strip() if pd.notna(row["product"]) else "",
            "target_sw": str(row["target_sw"]).lower().strip() if pd.notna(row["target_sw"]) else "",
        }
        lookup[uri] = rec

    return lookup


def iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_gold_components(
    gold_entry: Dict[str, Any],
    cpe_lookup: Dict[str, Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Convert gold_entry["vulnerable_components"] (which contains cpe_uri + version constraints)
    into list of component dicts matching the extraction schema.
    """
    comps_out: List[Dict[str, str]] = []

    vulns = gold_entry.get("vulnerable_components", [])
    for v in vulns:
        cpe_uri = str(v.get("cpe_uri", "")).strip()
        if not cpe_uri:
            continue
        meta = cpe_lookup.get(cpe_uri)
        if not meta:
            # skip unknown CPEs
            continue

        comp = {
            "part": meta.get("part", "*"),
            "vendor": meta.get("vendor", ""),
            "product": meta.get("product", ""),
            "target_sw": meta.get("target_sw", ""),
            "versionStartIncluding": str(v.get("versionStartIncluding") or ""),
            "versionStartExcluding": str(v.get("versionStartExcluding") or ""),
            "versionEndIncluding": str(v.get("versionEndIncluding") or ""),
            "versionEndExcluding": str(v.get("versionEndExcluding") or ""),
        }
        comps_out.append(comp)

    # Optional: dedupe exact dicts while preserving order
    seen = set()
    uniq = []
    for c in comps_out:
        key = tuple(sorted(c.items()))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    return uniq


def build_chat_example(description: str, gold_components: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Build a chat-style sample with system/user/assistant messages.
    """
    user_text = (description or "").strip()

    assistant_obj = {"components": gold_components}
    assistant_text = json.dumps(assistant_obj, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


def split_eval_vendor_balanced(
    all_examples: List[Dict[str, Any]],
    eval_size: int,
    random_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build a vendor-balanced eval set by sampling across vendors.
    Uses the FIRST component vendor in the gold label as the "vendor key".
    """
    rng = random.Random(random_seed)
    rng.shuffle(all_examples)

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for ex in all_examples:
        # parse vendor key from assistant JSON
        try:
            assistant_content = ex["messages"][2]["content"]
            obj = json.loads(assistant_content)
            comps = obj.get("components", [])
            vendor = ""
            if comps:
                vendor = str(comps[0].get("vendor", "")).strip().lower()
            if not vendor:
                vendor = "__no_vendor__"
        except Exception:
            vendor = "__parse_error__"

        buckets.setdefault(vendor, []).append(ex)

    # round-robin sample across buckets
    vendors = list(buckets.keys())
    rng.shuffle(vendors)

    eval_set = []
    while len(eval_set) < eval_size and vendors:
        progressed = False
        for v in list(vendors):
            if len(eval_set) >= eval_size:
                break
            if buckets[v]:
                eval_set.append(buckets[v].pop())
                progressed = True
            else:
                vendors.remove(v)
        if not progressed:
            break

    # remaining = train
    train_set = []
    for v, items in buckets.items():
        train_set.extend(items)

    return train_set, eval_set


def sample_train_set(
    train_examples: List[Dict[str, Any]],
    max_train: int,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Limit training set size while keeping it randomized.
    """
    if len(train_examples) <= max_train:
        return train_examples

    rng = random.Random(random_seed)
    rng.shuffle(train_examples)
    return train_examples[:max_train]


def save_jsonl(examples: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[Write] Saved {len(examples)} examples to {out_path}")


# ===============================
#   Main
# ===============================

def main():
    phase1_dir = Path("/kaggle/working/artifacts_phase1")
    phase2_dir = Path("/kaggle/working/artifacts_phase2")

    cpe_meta_path = phase1_dir / "cpe_meta.parquet"
    gold_jsonl_path = phase1_dir / "gold_by_cve.jsonl"

    train_out_path = phase2_dir / "extraction_train.jsonl"
    eval_out_path = phase2_dir / "extraction_eval.jsonl"

    print(f"[Load] cpe_meta from {cpe_meta_path}")
    cpe_lookup = load_cpe_meta(cpe_meta_path)

    print(f"[Load] gold JSONL from {gold_jsonl_path}")
    all_examples: List[Dict[str, Any]] = []
    n_in = 0
    n_skipped = 0

    for entry in iter_jsonl(gold_jsonl_path):
        n_in += 1
        desc = entry.get("description_en", "") or ""
        gold_comps = build_gold_components(entry, cpe_lookup)
        if not desc.strip():
            n_skipped += 1
            continue
        if not gold_comps:
            n_skipped += 1
            continue

        ex = build_chat_example(desc, gold_comps)
        all_examples.append(ex)

    print(f"[Stats] loaded CVEs: {n_in}")
    print(f"[Stats] usable examples: {len(all_examples)}")
    print(f"[Stats] skipped examples: {n_skipped}")

    print(f"[Split] building vendor-balanced eval set of size {EVAL_SIZE}")
    train_examples, eval_examples = split_eval_vendor_balanced(
        all_examples, eval_size=EVAL_SIZE, random_seed=RANDOM_SEED
    )

    print(f"[Split] train size: {len(train_examples)}, eval size: {len(eval_examples)}")

    print(f"[Sample] limiting train to max {MAX_TRAIN_EXAMPLES}")
    train_examples = sample_train_set(
        train_examples, max_train=MAX_TRAIN_EXAMPLES, random_seed=RANDOM_SEED,
    )
    print(f"[Sample] final train size: {len(train_examples)}")

    print("[Write] saving JSONL files ...")
    save_jsonl(train_examples, train_out_path)
    save_jsonl(eval_examples, eval_out_path)

    print("[Phase 2] Completed.")


if __name__ == "__main__":
    main()
