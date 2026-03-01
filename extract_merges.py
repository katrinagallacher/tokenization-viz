#!/usr/bin/env python3
"""
Extract merge data from a HuggingFace tokenizer.json for the BPE Merge Tree Visualizer.

Usage:
    python extract_merges.py <tokenizer.json> <output_name> [--label "Display Name"]

Examples:
    python extract_merges.py gemma2-tokenizer.json gemma2 --label "Gemma 2"
    python extract_merges.py qwen3-tokenizer.json  qwen3  --label "Qwen 3"

Output: ../data/<output_name>.json
"""

import json, sys, os, argparse


def detect_space_handling(data):
    """Detect how the tokenizer represents leading spaces in tokens.

    Returns:
      "sentencepiece" — spaces are replaced with ▁ by the normalizer
      "byte_level"    — spaces are byte-encoded as Ġ (GPT-2 style)
      "unknown"       — could not determine
    """
    norm = data.get("normalizer", {}) or {}
    pre  = data.get("pre_tokenizer", {}) or {}

    # Check normalizer chain for SentencePiece-style ▁ replacement
    normalizers = []
    if norm.get("type") == "Sequence":
        normalizers = norm.get("normalizers", [])
    elif norm.get("type"):
        normalizers = [norm]

    for step in normalizers:
        if step.get("type") == "Replace":
            pat = step.get("pattern", {})
            if pat.get("String") == " " and step.get("content") == "\u2581":
                return "sentencepiece"

    # Check for byte-level pre-tokenizer (GPT-2 / Qwen style)
    pretoks = []
    if pre.get("type") == "Sequence":
        pretoks = pre.get("pretokenizers", [])
    elif pre.get("type"):
        pretoks = [pre]

    for pt in pretoks:
        if pt.get("type") == "ByteLevel":
            return "byte_level"

    return "unknown"


def extract(path, name, label):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    model = data.get("model", {})

    # --- Merges ---
    merges_raw = model.get("merges", [])
    merges = []
    for m in merges_raw:
        # Two formats exist in the wild:
        #   Old: "left right" (single string, split on first space)
        #   New: ["left", "right"] (JSON array)
        if isinstance(m, str):
            parts = m.split(" ", 1)
            if len(parts) == 2:
                merges.append(parts)
        elif isinstance(m, list) and len(m) == 2:
            merges.append(m)

    # --- Vocab ---
    vocab = model.get("vocab", {})

    # --- Metadata ---
    space = detect_space_handling(data)

    meta = {
        "label": label or name,
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "space_handling": space,
    }

    return {"name": name, "meta": meta, "merges": merges, "vocab": vocab}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("tokenizer_json", help="Path to tokenizer.json")
    ap.add_argument("output_name", help="Short name, e.g. gemma2 or qwen3")
    ap.add_argument("--label", help="Human-readable label for the UI", default=None)
    args = ap.parse_args()

    print(f"Reading {args.tokenizer_json} ...")
    out = extract(args.tokenizer_json, args.output_name, args.label)

    m = out["meta"]
    print(f"  Model:          {m['label']}")
    print(f"  Vocab size:     {m['vocab_size']:,}")
    print(f"  Merges:         {m['num_merges']:,}")
    print(f"  Space handling: {m['space_handling']}")

    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{args.output_name}.json")

    # Write compact (no pretty-print) — saves ~50% vs pretty for large vocabs
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))

    sz = os.path.getsize(outpath)
    unit = "MB" if sz > 1e6 else "KB"
    val = sz / 1e6 if sz > 1e6 else sz / 1e3
    print(f"  Written:        {outpath}  ({val:.1f} {unit})")

    print("\nFirst 5 merges:")
    for i, (a, b) in enumerate(out["merges"][:5]):
        print(f"  #{i}: '{a}' + '{b}' -> '{a}{b}'")
    print(f"  ...")
    for i, (a, b) in enumerate(out["merges"][-3:], m["num_merges"] - 3):
        print(f"  #{i}: '{a}' + '{b}' -> '{a}{b}'")


if __name__ == "__main__":
    main()
