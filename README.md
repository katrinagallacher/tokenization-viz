# BPE Merge Tree Visualizer

Trace how BPE tokenizers produce specific tokenizations, step by step.

**[Live Demo →](https://YOUR_USERNAME.github.io/bpe-merge-viz/)**

Compare **Gemma 2** (SentencePiece BPE, 256K vocab) and **Qwen 3** (byte-level BPE, 151K vocab) side by side on compound words, showing where statistical merge ranks override morphological structure.

## What this shows

BPE tokenizers merge character pairs greedily by frequency rank. This creates situations where high-frequency subword merges fire before morphologically correct ones:

```
"bodyshell" → body + shell  ✓ (what humans expect)
"bodyshell" → bodys + hell  ✗ (what BPE might produce)
```

This happens because the `y + s` merge (from "days", "ways", "says" — rank ~1800) fires before `bod + y` (rank ~5000), locking in `ys` before the word boundary is established.

The visualizer traces each merge step with its rank, highlights what merged and what's about to merge next, and checks whether the final tokenization respects morphological boundaries.

## Setup

### 1. Clone this repo

```bash
git clone https://github.com/YOUR_USERNAME/bpe-merge-viz.git
cd bpe-merge-viz
```

### 2. Download tokenizer files

Both models distribute their tokenizer as `tokenizer.json` on HuggingFace:

**Gemma 2** (requires accepting Google's license on HuggingFace first):
```bash
# Option A: Direct download (needs HF token)
huggingface-cli download google/gemma-2-9b tokenizer.json --local-dir /tmp/gemma2

# Option B: Manual download from browser
# https://huggingface.co/google/gemma-2-9b/blob/main/tokenizer.json
```

**Qwen 3**:
```bash
# Option A: Direct download
huggingface-cli download Qwen/Qwen3-8B tokenizer.json --local-dir /tmp/qwen3

# Option B: Manual download from browser
# https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer.json
```

> **Note:** Any model in the same family shares the tokenizer. Gemma 2 2B/9B/27B all use the same `tokenizer.json`. Same for Qwen 3 0.6B/1.7B/4B/8B/14B/32B.

### 3. Extract merge data

```bash
pip install --quiet sentencepiece  # only needed if you want to cross-check

python scripts/extract_merges.py /tmp/gemma2/tokenizer.json gemma2 --label "Gemma 2"
python scripts/extract_merges.py /tmp/qwen3/tokenizer.json  qwen3  --label "Qwen 3"
```

This produces `data/gemma2.json` and `data/qwen3.json` — compact files containing the merge list, vocabulary, and metadata.

Expected output:
```
Reading /tmp/gemma2/tokenizer.json ...
  Model:          Gemma 2
  Vocab size:     256,128
  Merges:         ~250,000
  Space handling: sentencepiece
  Written:        data/gemma2.json  (~25 MB)
```

### 4. File size considerations

The extracted data files can be large (Gemma 2 ≈ 25MB, Qwen 3 ≈ 15MB) because they include the full merge list and vocabulary. GitHub Pages can serve these fine, but initial load will take a moment.

Options to reduce size:
- **Git LFS** (recommended): Track data files with Git LFS to keep the repo lightweight:
  ```bash
  git lfs install
  git lfs track "data/*.json"
  git add .gitattributes
  ```
- **CDN hosting**: Upload data files to GitHub Releases or a CDN and update the `load()` URLs in `index.html`.

### 5. Deploy to GitHub Pages

```bash
git add index.html scripts/ data/ README.md
git commit -m "BPE Merge Tree Visualizer"
git push origin main
```

Then go to **Settings → Pages → Deploy from branch** → select `main`, root `/`.

Your site will be live at `https://YOUR_USERNAME.github.io/bpe-merge-viz/`

## How it works

### The BPE replay algorithm

The visualizer implements BPE merge replay in JavaScript:

1. **Initialize** — Convert input text to a character sequence:
   - Gemma (SentencePiece): replace spaces with `▁`, split to characters
   - Qwen (byte-level): UTF-8 encode, map each byte through GPT-2's byte→unicode table

2. **Greedy merge** — Repeat until no more merges apply:
   - Scan all adjacent pairs, find the one with the lowest rank in the merge list
   - Merge it everywhere it occurs
   - Record the step: symbols, pair merged, rank

3. **Display** — Show the step-by-step tree with color coding:
   - 🟢 Green: just merged
   - 🟡 Yellow: about to merge in the next step

### Key difference: SentencePiece vs byte-level BPE

| | Gemma 2 (SentencePiece) | Qwen 3 (byte-level) |
|---|---|---|
| Space representation | `▁` (U+2581) | `Ġ` (U+0120) |
| Pre-tokenization | None — BPE merges can cross word boundaries | Regex splits text first — merges cannot cross word boundaries |
| Initial units | Unicode characters | UTF-8 bytes (mapped to printable unicode) |

This difference matters for compound words: Gemma's lack of pre-tokenization means merges like `s + hell` can bridge "bodyshell" incorrectly.

## Project structure

```
bpe-merge-viz/
├── index.html                  # Complete single-file web application
├── data/
│   ├── gemma2.json             # Extracted merge data (you generate this)
│   └── qwen3.json              # Extracted merge data (you generate this)
├── scripts/
│   └── extract_merges.py       # Extraction script
└── README.md
```

## Adding more models

The tool supports any BPE model that has a HuggingFace `tokenizer.json`:

1. Download the `tokenizer.json`
2. Run the extraction: `python scripts/extract_merges.py path/to/tokenizer.json modelname --label "Model Name"`
3. Add a third panel in `index.html` (copy an existing panel, change the IDs)
4. Add a `load("m3", "data/modelname.json")` call in the init function

Compatible models include: Llama 2/3, Mistral, DeepSeek, Phi-3, GPT-2, and any other BPE-based model.

Not compatible: WordPiece (BERT) or Unigram (T5) — these use different algorithms without merge lists.

## Research context

This tool is part of research investigating how BPE tokenization artifacts force language models to develop "repair circuits" in early layers. When statistical frequency causes morphologically incorrect token splits, models must spend computation disambiguating the resulting tokens — effectively performing detokenization repair.

The compound words in the word list were selected to showcase cases where merge rank ordering produces linguistically surprising tokenizations.

## License

MIT
