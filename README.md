# Labor Union Parser

Extract structured fields from labor union name strings by matching against
an OLMS gazetteer of ~44,000 filing records.

Given an input like `"SEIU Local 1199"`, the parser returns:
- `is_union`: True (detected as a union)
- `union_score`: 0.9997 (confidence score for union detection)
- `union_name`: SERVICE EMPLOYEES (predicted parent union name)
- `desig_name`: LU (predicted designation type — Local Union)
- `desig_num`: 1199 (predicted local number)
- `prefix`: (predicted designation prefix, if any)
- `suffix`: (predicted designation suffix, if any)
- `f_num`: 509111 (OLMS filing number of best-scoring gazetteer record)
- `match_found`: True (whether the model found a confident gazetteer match)
- `match_score`: 0.4496 (probability of best gazetteer match)
- `conflicts`: [] (mismatches between predictions and matched record)

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
<!--[[[cog
import pprint
from labor_union_parser import Extractor

result = Extractor().extract("SEIU Local 1199")
for line in pprint.pformat(result, width=72).splitlines():
    cog.outl(f"# {line}")
]]]-->
# {'conflicts': [],
#  'desig_name': 'LU',
#  'desig_num': '1199',
#  'f_num': 516569,
#  'field_scores': {'desig_name': 0.8226500153541565,
#                   'desig_num': 0.9992030262947083,
#                   'f_num': 0.39868733286857605,
#                   'prefix': 0.9999760389328003,
#                   'suffix': 0.9995836615562439,
#                   'union_name': 0.999906063079834},
#  'is_union': True,
#  'match_found': True,
#  'match_score': 0.33174464106559753,
#  'prefix': '',
#  'suffix': '',
#  'union_name': 'SERVICE EMPLOYEES',
#  'union_score': 0.9997227191925049}
<!--[[[end]]]-->
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
<!--[[[cog
import subprocess
result = subprocess.run(
    'echo "SEIU Local 1199" | labor-union-parser --no-header',
    shell=True, capture_output=True, text=True
)
for line in result.stdout.strip().splitlines():
    cog.outl(line)
]]]-->
text,pred_is_union,pred_union_score,pred_union_name,pred_desig_name,pred_desig_num,pred_prefix,pred_suffix,pred_f_num,pred_match_found,pred_match_score,pred_conflicts,score_union_name,score_desig_name,score_f_num,score_desig_num,score_prefix,score_suffix
SEIU Local 1199,True,0.9997,SERVICE EMPLOYEES,LU,1199,,,516569,True,0.33174464106559753,,0.9999,0.8227,0.3987,0.9992,1.0000,0.9996
<!--[[[end]]]-->
```

## Output Fields

| Field | Description |
|-------|-------------|
| `is_union` | Whether the text is detected as a union name |
| `union_score` | Similarity score to union centroid (0-1) |
| `union_name` | Predicted parent union name (e.g., "SERVICE EMPLOYEES", "TEAMSTERS") |
| `desig_name` | Predicted designation type (e.g., "LU" for Local Union, "JC" for Joint Council) |
| `desig_num` | Predicted local/designation number (e.g., "1199") |
| `prefix` | Predicted designation prefix, if any |
| `suffix` | Predicted designation suffix, if any |
| `f_num` | OLMS filing number of the best-scoring gazetteer record |
| `match_found` | Whether the model found a confident gazetteer match (False when the learned null record outscores all real records) |
| `match_score` | Probability of best gazetteer match (0-1) |
| `conflicts` | List of conflict codes (see below) |
| `field_scores` | Per-field probabilities for the matched record (see below) |

### Conflict Codes

The `conflicts` list (Python API) or `pred_conflicts` column (CLI, pipe-delimited)
flags mismatches between the field predictions and the matched gazetteer record.
A non-empty conflicts list indicates the model's field-level predictions disagree
with the record selected by the gazetteer scoring, which may signal a bad match.

| Code | Description |
|------|-------------|
| `union_name_mismatch` | Predicted union name differs from the matched record. Strongest signal of a bad match. |
| `desig_name_mismatch` | Predicted designation type differs from the matched record (e.g., LU vs JC). |
| `desig_num_mismatch` | Predicted designation number differs from the matched record. |
| `prefix_mismatch` | Predicted prefix differs from the matched record. |
| `suffix_mismatch` | Predicted suffix differs from the matched record. |

The `field_scores` dict (Python API) or `score_*` columns (CLI) give the
classifier's confidence in its top prediction for each field.
Values close to 1.0 indicate the head is confident in its prediction;
lower values indicate uncertainty among multiple candidates.

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
              ▼ (always runs)
┌───────────────────────────────────────────────────┐
│  Stage 2: Structured Classifier + Gazetteer       │
│                                                   │
│  CharCNN per token → RoPE Transformer →           │
│  Per-field classification & pointer heads         │
│                                                   │
│  Learned linear combination of per-field          │
│  log-probs across ~44K gazetteer records          │
│                                                   │
│  Match: SERVICE EMPLOYEES LU 1199, f_num=509111   │
└───────────────────────────────────────────────────┘
              │
              ▼
Output: {is_union: True, union_name: "SERVICE EMPLOYEES",
         desig_name: "LU", desig_num: "1199", f_num: 509111, ...}
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
- **Threshold**: Similarity ≥ 0.9 → is_union = True

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
position). For each gazetteer record, we assemble a 12-feature vector:

- 6 **log-prob features**: the classifier's log-probability for each field
  value of that record (set to 0 if the value is unknown to the vocabulary
  or not found in the query tokens)
- 3 **unknown indicators**: 1 if the record's classification field value is
  not in the vocabulary, 0 otherwise
- 3 **not-found indicators**: 1 if the record's pointer field value is not
  present in the query tokens, 0 otherwise

A learned linear layer (12 weights + bias, trained with marginalized
cross-entropy over all correct records) scores each record. The
log-prob weights are less than 1.0, acting as correlation discounts
that correct for the naive independence assumption across fields. The
highest-scoring record is the prediction, and `match_score` is the
softmax probability of that record.

### Performance

<!--[[[cog
import sys; sys.path.insert(0, "training")
from evaluate import compute_test_metrics, SCORE_FIELDS

m = compute_test_metrics()

total_errors = m['wrong_matches'] + m['false_negatives'] + m['false_match_no_fnum'] + m['false_positives']
total_correct = m['n_scored'] - total_errors
accuracy = total_correct / m['n_scored']

cog.outl(f"End-to-end on held-out test data ({m['n_scored']:,} examples")
cog.outl("scored against the full 44K-record gazetteer):")
cog.outl("")
cog.outl("| Metric | Score |")
cog.outl("|--------|-------|")
cog.outl(f"| Accuracy | {accuracy:.1%} |")
cog.outl(f"| Wrong match (union, wrong f_num) | {m['wrong_matches']} |")
cog.outl(f"| False negatives (union missed) | {m['false_negatives']} |")
cog.outl("")
cog.outl(f"Per-field accuracy on test set ({m['n_is_union']:,} union examples with is_union=True):")
cog.outl("")
cog.outl("| Field | Accuracy |")
cog.outl("|-------|----------|")
for f in SCORE_FIELDS:
    if f == "f_num":
        continue
    cog.outl(f"| `{f}` | {m['field_accuracy'][f]:.1%} |")
]]]-->
End-to-end on held-out test data (7,687 examples
scored against the full 44K-record gazetteer):

| Metric | Score |
|--------|-------|
| Accuracy | 98.2% |
| Wrong match (union, wrong f_num) | 113 |
| False negatives (union missed) | 2 |

Per-field accuracy on test set (9,486 union examples with is_union=True):

| Field | Accuracy |
|-------|----------|
| `union_name` | 98.4% |
| `desig_name` | 99.0% |
| `desig_num` | 99.0% |
| `prefix` | 97.6% |
| `suffix` | 96.7% |
<!--[[[end]]]-->
