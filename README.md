# Labor Union Parser

Extract affiliation and local designation from labor union name strings.

Given an input like `"SEIU Local 1199"`, the parser returns:
- `is_union`: True (detected as a union)
- `union_score`: 0.999 (confidence score for union detection)
- `affiliation`: SEIU (Service Employees International Union)
- `affiliation_unrecognized`: False (True if affiliation couldn't be matched)
- `aff_score`: 0.997 (confidence score for affiliation)
- `designation`: 1199 (local number)

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
# {'is_union': True, 'union_score': 0.999, 'affiliation': 'SEIU',
#  'affiliation_unrecognized': False, 'designation': '1199', 'aff_score': 0.997}
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
            print(result["affiliation"], result["designation"])
```


### Filing Number Lookup

Look up OLMS filing numbers for a given affiliation and designation:

```python
from labor_union_parser import lookup_fnum

fnums = lookup_fnum("SEIU", "1199")
# [31847, 69557, 508557, ...]
```

### Command Line

```bash
# Process CSV file
labor-union-parser unions.csv -c union_name -o results.csv

# Process from stdin
echo "SEIU Local 1199" | labor-union-parser --no-header
# text,pred_is_union,pred_aff,pred_unknown,pred_desig,pred_union_score,pred_fnum,pred_fnum_multiple
# SEIU Local 1199,True,SEIU,False,1199,0.9992,"[31847, 69557, ...]",True
```

## Output Fields

| Field | Description |
|-------|-------------|
| `is_union` | Whether the text is detected as a union name |
| `union_score` | Similarity score to union centroid (0-1) |
| `affiliation` | Predicted affiliation abbreviation (e.g., "SEIU", "IBT") or `None` |
| `affiliation_unrecognized` | `True` if detected as union but affiliation unrecognized |
| `designation` | Extracted local number (e.g., "1199") or empty string |
| `aff_score` | Similarity to nearest affiliation centroid (higher = more confident) |

## Training

Training data is in `training/data/labeled_data.csv` with columns:
- `text`: Union name string
- `aff_abbr`: Affiliation abbreviation (e.g., "SEIU", "IBT", "UAW")
- `desig_num`: Local designation number

To retrain the model:

```bash
pip install -e ".[train]"  # Install training dependencies
python -m training.train              # Train all stages
python -m training.train --stage 1    # Train only union detector
python -m training.train --stage 2    # Train only affiliation classifier
python -m training.train --stage 3    # Train only designation extractor
```

## Model Architecture

The model uses a three-stage contrastive extraction pipeline:

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
│  CharCNN (shared across stages)                   │
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
│  Stage 2: Affiliation (Nearest Centroid)          │
│                                                   │
│  Token embeddings + is_number embedding →         │
│  Cross-attention (learned query) → Projection →   │
│  Similarity to affiliation centroids              │
│                                                   │
│  Nearest: SEIU (score = 0.997)                    │
└───────────────────────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────────────┐
│  Stage 3: Designation (Pointer Network)           │
│                                                   │
│  Token embeddings + Transformer encoder →         │
│  BiLSTM + affiliation embedding → pointer scores  │
│                                                   │
│  Points to: "1199"                                │
└───────────────────────────────────────────────────┘
              │
              ▼
Output: {is_union: True, union_score: 0.999, affiliation: "SEIU",
         affiliation_unrecognized: False, aff_score: 0.997, designation: "1199"}
```

### CharCNN

Character-level CNN that computes token embeddings, shared across all stages.

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

### Stage 2: Affiliation Classification

Nearest-centroid classification in contrastive embedding space.

- **Input**: CharCNN token embeddings + is_number embedding (8-dim)
- **Cross-attention**: Learned query attends over token sequence
- **Projection**: 2-layer MLP (72 → 128 → 64) with L2 normalization
- **Training**: Supervised contrastive loss (same-affiliation = positive pairs)
- **Inference**: Cosine similarity to each affiliation centroid
- **Threshold**: Best score < 0.80 → affiliation_unrecognized = True

### Stage 3: Designation Extraction

Pointer network that selects the correct local number token.

- **Input**: CharCNN token embeddings + special token embeddings (numbers, punct)
- **Context**: Transformer encoder (3 layers, 4 heads)
- **Selection**: BiLSTM + affiliation embedding → score each number token
- **Output**: Highest-scoring number token, or empty if no designation

### Performance

On labeled data (94,308 examples with known affiliations):

| Metric | All | Non-None Predictions |
|--------|-----|---------------------|
| Affiliation accuracy | 99.0% | 99.7% |
| Joint accuracy | 98.9% | 99.5% |

- Designation accuracy: 99.9%
- Only 0.7% of predictions return None (unrecognized affiliation)
