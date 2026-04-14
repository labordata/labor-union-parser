# Labor Union Parser

Match labor union name text to OLMS filing numbers using a factored
ArcFace model against a gazetteer of ~44,000 filing records.

Given an input like `"SEIU Local 1199"`, the parser returns:
- `is_union`: True (detected as a union)
- `union_score`: 0.992 (calibrated probability of being a union)
- `union_name`: SERVICE EMPLOYEES (predicted parent union name)
- `f_num`: 31847 (OLMS filing number of best-matching gazetteer record)
- `match_score`: 0.956 (softmax probability of best match)

Other record fields (designation type, local number, prefix, suffix)
are fully determined by the f_num and can be looked up from the
[OLMS gazetteer](https://github.com/labordata/opdr).

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
```

<!--[[[cog
import pprint
from labor_union_parser import Extractor

result = Extractor().extract("SEIU Local 1199")
lines = pprint.pformat(result, width=72).splitlines()
cog.outl("```")
for line in lines:
    cog.outl(f"# {line}")
cog.outl("```")
]]]-->
```
# {'f_num': 31847,
#  'is_union': True,
#  'match_score': 0.982312023639679,
#  'union_name': 'SERVICE EMPLOYEES',
#  'union_score': 0.8276892304420471}
```
<!--[[[end]]]-->

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
            print(result["f_num"], result["union_name"])
```


### Command Line

```bash
# Process CSV file
labor-union-parser unions.csv -c union_name -o results.csv

# Process from stdin
echo "SEIU Local 1199" | labor-union-parser --no-header
```

<!--[[[cog
import subprocess
result = subprocess.run(
    'echo "SEIU Local 1199" | labor-union-parser --no-header',
    shell=True, capture_output=True, text=True
)
cog.outl("```")
for line in result.stdout.strip().splitlines():
    cog.outl(line)
cog.outl("```")
]]]-->
```
text,pred_is_union,pred_union_score,pred_union_name,pred_f_num,pred_match_score
SEIU Local 1199,True,0.8277,SERVICE EMPLOYEES,31847,0.9823
```
<!--[[[end]]]-->

## Output Fields

| Field | Description |
|-------|-------------|
| `is_union` | Whether the text is detected as a union name |
| `union_score` | Calibrated probability of being a union (0-1, Platt-scaled) |
| `union_name` | Predicted parent union name from the shared classification head |
| `f_num` | OLMS filing number of the best-matching gazetteer record |
| `match_score` | Softmax probability of best gazetteer match (0-1) |

## Training

Training data and scripts are in `training/`. The pipeline is orchestrated by the root Makefile:

```bash
pip install -e ".[train]"   # Install training dependencies

make data                   # Download opdr.db, generate gazetteer and training data
make train                  # Train ArcFace classifier and union detector
make evaluate               # Run evaluation
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
│  tokens: ["seiu", "local", "1199"]                │
│  is_num: [False, False, True]                     │
│  + FastText char n-gram hashes + Bloom number IDs │
└───────────────────────────────────────────────────┘
              │
              ▼
┌───────────────────────────────────────────────────┐
│  Stage 1: Union Detection (Contrastive)           │
│                                                   │
│  FastText + Bloom + RoPE Transformer (2 layers)   │
│  → Mean pool → Projection → L2 normalize          │
│  → Cosine similarity to learned union prototype   │
│  → Platt scaling: sigmoid(a·sim + b)              │
│                                                   │
│  union_score = 0.99 → is_union = True             │
└───────────────────────────────────────────────────┘
              │
              ▼ (always runs)
┌───────────────────────────────────────────────────┐
│  Stage 2: Factored ArcFace Classifier             │
│                                                   │
│  FastText + Bloom + RoPE Transformer (3 layers)   │
│  → Mean pool → L2 normalize                       │
│                                                   │
│  Score against ~35K factored prototypes:           │
│  prototype = W_union + W_desig + bloom(num)       │
│            + W_prefix + W_suffix + W_fnum         │
│  (~17K trained + ~18K zero-shot from gazetteer)   │
│                                                   │
│  Match: SERVICE EMPLOYEES LU 1199 → f_num=31847   │
└───────────────────────────────────────────────────┘
              │
              ▼
Output: {is_union: True, union_name: "SERVICE EMPLOYEES",
         f_num: 31847, match_score: 0.96, ...}
```

### Stage 1: Union Detection

Contrastive learning to distinguish union names from non-union text.
Uses the same FastText+RoPE encoder architecture as Stage 2 (2 layers
instead of 3), trained with ArcFace angular margin against a learned
union prototype.

- **Encoder**: FastText + Bloom + RoPE Transformer (2 layers, 128-dim)
- **Pooling**: Masked mean pool → linear projection → L2 normalize (64-dim)
- **Training**: ArcFace contrastive loss with 20K F7 employer names as hard negatives
- **Calibration**: Platt scaling (sigmoid) for calibrated probability output
- **Inference**: Cosine similarity to learned prototype → Platt-scaled probability

### Stage 2: Factored ArcFace Classifier

A single forward pass through the encoder produces a query embedding.
This is scored against factored prototypes — one per gazetteer record —
via cosine similarity. No pairwise comparisons needed.

**Encoder:**
- **FastText embedding**: Vocabulary lookup + hashed character 3-6 gram average.
  Typo-robust: similar spellings share n-gram hashes.
- **Bloom number embedding**: Numbers hashed to 3 indices in a 4096-entry
  table, summed. Treats numbers as opaque identifiers.
- **RoPE Transformer**: 3 layers, 4 heads, 128-dim. Position-aware
  attention helps distinguish "district 10 local 66" from "district 66 local 10".
- **Pooling**: Masked mean pool → L2 normalize → 128-dim query embedding.

**Factored Prototypes:**

Each f_num's prototype is the sum of learned field embeddings:

```
prototype = W_union[u] + W_desig_name[d] + bloom(desig_num)
          + W_prefix[p] + W_suffix[s] + W_fnum[f]
```

This additive structure means the model learns separate representations
for each field. At inference, scoring is a single matrix multiply
against ~35K pre-computed prototype vectors (~17K trained classes +
~18K zero-shot from gazetteer with `W_fnum = 0`).

**Zero-shot prototypes:** For gazetteer f_nums without training data,
prototypes are built from field embeddings alone. During training,
these are included as frozen distractors in the ArcFace softmax,
teaching the model to distinguish trained classes from similar
zero-shot prototypes. W_fnum is L2-regularized to keep trained
prototypes close to their zero-shot versions.

**Union Head:**

An auxiliary classification head shares the `W_union` embedding weights
with the prototypes. During training, a disagree penalty ensures the
f_num predictions are consistent with the union head's prediction.
At inference, the union head provides the `union_name` output.

**CRF Tag Head (training only):**

A per-token CRF labels numbers as desig_num, prefix, or suffix using
constrained marginalization — we know the field values from the gazetteer
but not which tokens they correspond to, so the loss marginalizes over
all valid alignments (à la CTC). This teaches the encoder to represent
number roles without requiring ground truth token labels.

### Performance

<!--[[[cog
import sys; sys.path.insert(0, "training")
from evaluate import compute_test_metrics

m = compute_test_metrics()

total_errors = m['wrong_matches'] + m['false_negatives'] + m['false_positives']
total_correct = m['n_scored'] - total_errors
accuracy = total_correct / m['n_scored']

cog.outl(f"End-to-end on held-out test data ({m['n_scored']:,} examples")
cog.outl("scored against the full 44K-record gazetteer):")
cog.outl("")
cog.outl("| Metric | Score |")
cog.outl("|--------|-------|")
cog.outl(f"| Accuracy | {accuracy:.1%} |")
cog.outl(f"| f_num accuracy (union examples) | {m['fnum_accuracy']:.1%} ({m['fnum_correct']}/{m['fnum_total']}) |")
cog.outl(f"| f_num accuracy (in-vocab only) | {m['fnum_invocab_accuracy']:.1%} |")
cog.outl(f"| union_name accuracy | {m['union_accuracy']:.1%} ({m['union_correct']}/{m['union_total']}) |")
cog.outl(f"| Wrong match (union, wrong f_num) | {m['wrong_matches']} |")
cog.outl(f"| False negatives (union missed) | {m['false_negatives']} |")
cog.outl(f"| False positives (non-union matched) | {m['false_positives']} |")
]]]-->
End-to-end on held-out test data (8,691 examples
scored against the full 44K-record gazetteer):

| Metric | Score |
|--------|-------|
| Accuracy | 98.0% |
| f_num accuracy (union examples) | 98.6% (7519/7627) |
| f_num accuracy (in-vocab only) | 98.6% |
| union_name accuracy | 98.0% (9180/9364) |
| Wrong match (union, wrong f_num) | 108 |
| False negatives (union missed) | 24 |
| False positives (non-union matched) | 43 |
<!--[[[end]]]-->
