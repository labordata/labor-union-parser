# Labor Union Parser

Extract structured fields from labor union name strings by matching against
an OLMS gazetteer of ~44,000 filing records.

Given an input like `"SEIU Local 1199"`, the parser returns:
- `is_union`: True (detected as a union)
- `union_score`: 0.9997 (confidence score for union detection)
- `union_name`: SERVICE EMPLOYEES (matched union name from gazetteer)
- `desig_name`: LU (designation type — Local Union)
- `desig_num`: 1199 (local number)
- `prefix`: (designation prefix, if any)
- `suffix`: (designation suffix, if any)
- `f_num`: 509111 (OLMS filing number)
- `match_score`: -0.5042 (log-probability score of best gazetteer match)

## Installation

```bash
pip install labor-union-parser
```

## Usage

### Python API

```python
from labor_union_parser import Extractor

extractor = Extractor()
result = extractor.extract("SEIU Local 1199")
print(result)
# {'is_union': True, 'union_score': 0.9997, 'union_name': 'SERVICE EMPLOYEES',
#  'desig_name': 'LU', 'desig_num': '1199', 'prefix': '', 'suffix': '',
#  'f_num': '509111', 'match_score': '-0.5042'}
```

For batch processing, use `extract_batch` which processes texts in parallel for better throughput:

```python
from labor_union_parser import Extractor

extractor = Extractor()
results = extractor.extract_batch([
    "SEIU Local 1199",
    "Teamsters Local 705",
    "UAW Local 600",
])
# Returns list of result dicts, one per input text
```

The `batch_size` parameter controls how many texts are processed at once (default: 256). Larger batches are faster but use more memory:

```python
# Process 512 texts at a time
results = extractor.extract_batch(texts, batch_size=512)
```

For very large datasets, combine `extract_batch` with `itertools.batched` to process in chunks and avoid loading everything into memory:

```python
import itertools
from labor_union_parser import Extractor

extractor = Extractor()

# Stream through a large file, processing 1000 at a time
with open("union_names.txt") as f:
    for chunk in itertools.batched(f, 1000):
        texts = [line.strip() for line in chunk]
        for result in extractor.extract_batch(texts):
            print(result["union_name"], result["desig_num"])
```


### Command Line

```bash
# Process CSV file
labor-union-parser unions.csv -c union_name -o results.csv

# Process from stdin
echo "SEIU Local 1199" | labor-union-parser --no-header
# text,pred_is_union,pred_union_score,pred_union_name,pred_desig_name,pred_desig_num,pred_prefix,pred_suffix,pred_f_num,pred_match_score
# SEIU Local 1199,True,0.9997,SERVICE EMPLOYEES,LU,1199,,,509111,-0.5042
```

## Output Fields

| Field | Description |
|-------|-------------|
| `is_union` | Whether the text is detected as a union name |
| `union_score` | Similarity score to union centroid (0-1) |
| `union_name` | Matched parent union name (e.g., "SERVICE EMPLOYEES", "TEAMSTERS") |
| `desig_name` | Designation type (e.g., "LU" for Local Union, "JC" for Joint Council) |
| `desig_num` | Local/designation number (e.g., "1199") |
| `prefix` | Designation prefix, if any |
| `suffix` | Designation suffix, if any |
| `f_num` | OLMS filing number for the matched record |
| `match_score` | Log-probability score of best gazetteer match |

## Training

Training data and scripts are in `training/`. The pipeline is orchestrated by the root Makefile:

```bash
pip install -e ".[train]"   # Install training dependencies

make data                   # Download opdr.db, generate gazetteer and training data
make train                  # Train structured classifier and union detector
make evaluate               # Run evaluation scripts
make all                    # Full pipeline (data + train)
```

### Checked-in Data

- `training/data/labeled_data.csv` — labeled union name examples
- `training/data/nonunion_examples.csv` — non-union text examples
- `training/data/acronym_to_fullname.csv` — union acronym mappings

## Model Architecture

The model uses a two-stage pipeline:

```
Input: "SEIU Local 1199"
              │
              ▼
┌───────────────────────────────────────────────────┐
│  Tokenizer                                        │
│  tokens: ["SEIU", " ", "Local", " ", "1199"]      │
│  token_type: [word, space, word, space, number]   │
└───────────────────────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────────────┐
│  CharCNN                                          │
│                                                   │
│  For each token: chars → char embeddings →        │
│  parallel CNNs (1,2,3-grams) → max pool →         │
│  highway layer → 64-dim token embedding           │
│                                                   │
│  Typo-robust: "SEIU" ≈ "SIEU" ≈ "S.E.I.U."        │
└───────────────────────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────────────┐
│  Stage 1: Union Detection (Contrastive)           │
│                                                   │
│  Token embeddings + is_number embedding →         │
│  Cross-attention (learned query) → Projection →   │
│  Similarity to union centroid                     │
│                                                   │
│  score = 0.999 → is_union = True                  │
└───────────────────────────────────────────────────┘
              │
              ▼ (if is_union)
┌───────────────────────────────────────────────────┐
│  Stage 2: Structured Classifier + Gazetteer       │
│                                                   │
│  CharCNN per token → RoPE Transformer →           │
│  Per-field classification & pointer heads          │
│                                                   │
│  Sum log-probs across fields for each of          │
│  ~44K gazetteer records → best match              │
│                                                   │
│  Match: SERVICE EMPLOYEES LU 1199, f_num=509111   │
└───────────────────────────────────────────────────┘
              │
              ▼
Output: {is_union: True, union_name: "SERVICE EMPLOYEES",
         desig_name: "LU", desig_num: "1199", f_num: "509111", ...}
```

### CharCNN

Character-level CNN that computes token embeddings from characters.

- **Character embedding**: 16-dim lookup for ~50 chars (letters, digits, punctuation)
- **Parallel CNNs**: 1-gram (32 filters), 2-gram (64 filters), 3-gram (128 filters)
- **Pooling**: Max-pool over character dimension → 224-dim
- **Highway layer**: Gated transformation for non-linearity
- **Projection**: Linear layer → 64-dim token embedding
- **Typo-robust**: Similar spellings produce similar embeddings

### Stage 1: Union Detection

Contrastive learning to distinguish union names from non-union text.

- **Input**: CharCNN token embeddings + is_number embedding (8-dim)
- **Cross-attention**: Learned query attends over token sequence
- **Projection**: 2-layer MLP (72 → 128 → 64) with L2 normalization
- **Training**: One-class contrastive loss (union examples form positive pairs)
- **Inference**: Cosine similarity to learned union centroid
- **Threshold**: Similarity ≥ 0.5 → is_union = True

### Stage 2: Structured Classifier + Factored Scoring

A single forward pass through the classifier produces per-field probability
distributions. These are combined with a gazetteer of ~44K OLMS filing
records to find the best match — no pairwise comparisons needed.

**Classifier:**
- **Encoder**: CharCNN (20 tokens × 20 chars) → Transformer with RoPE (2 layers, 4 heads)
- **Classification heads**: `union_name`, `desig_name`, `f_num` — softmax over field vocabulary
- **Pointer heads**: `desig_num`, `prefix`, `suffix` — attention over input token positions

**Scoring:**

Each classification head produces a log-probability distribution over its
vocabulary (e.g., all known union names). Each pointer head produces a
log-probability distribution over input token positions (plus a "none"
position). For each gazetteer record, we look up the log-probability of
that record's field value under the corresponding head, then sum across
fields to get a total score. The highest-scoring record is the prediction.

The `f_num` head is treated separately because it directly predicts the
filing number, whereas the other fields (`union_name`, `desig_name`,
`desig_num`, etc.) are shared properties that many records have in common
and serve to narrow down the candidates. We blend the `f_num`
log-probability with the other fields' combined score using a per-record
weight: `score = (1 - w) * other_fields + w * f_num`. The weight `w`
ranges from 0.1 (for filing numbers unseen in training) to 0.6 (for
well-represented ones), scaling with `log(1 + n)` where `n` is the
training count.

### Performance

On held-out test data (7,160 labeled examples scored against the full 44K-record gazetteer):

| Metric | Score |
|--------|-------|
| Filing number accuracy (top-1) | 98.8% |
| Filing number accuracy (top-5) | 99.6% |
| Non-union text filtering accuracy | 96.3% |
| Non-union text filtering ROC-AUC | 0.995 |

Per-field classifier accuracy on test set:

| Field | Accuracy |
|-------|----------|
| `union_name` | 99.1% |
| `desig_name` | 99.3% |
| `desig_num` | 99.6% |
| `prefix` | 99.3% |
| `suffix` | 99.6% |
| `f_num` | 96.7% |
