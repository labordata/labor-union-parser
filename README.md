# Labor Union Parser

Match labor union name text to [Office of Labor-Management Standards filing numbers](https://olmsapps.dol.gov/olpdr/).

## Installation

```console
pip install labor-union-parser
```

## Usage

### Python API

<!--[[[cog
import pprint
from labor_union_parser import Extractor

result = Extractor().extract("SEIU Local 1199")
lines = pprint.pformat(result, width=72).splitlines()
cog.outl("```python")
cog.outl("from labor_union_parser import Extractor")
cog.outl("")
cog.outl('extractor = Extractor()')
cog.outl('result = extractor.extract("SEIU Local 1199")')
cog.outl("print(result)")
for line in lines:
    cog.outl(f"# {line}")
cog.outl("```")
]]]-->
```python
from labor_union_parser import Extractor

extractor = Extractor()
result = extractor.extract("SEIU Local 1199")
print(result)
# {'f_num': 31847,
#  'f_num_score': 0.9500725865364075,
#  'is_union': True,
#  'is_union_score': 0.9268560409545898,
#  'union_name': 'SERVICE EMPLOYEES',
#  'union_name_score': 0.9972871541976929}
```
<!--[[[end]]]-->

For batch processing, use `extract_batch` which processes texts in parallel for better throughput:

<!--[[[cog
import pprint
from labor_union_parser import Extractor

results = Extractor().extract_batch([
    "SEIU Local 1199",
    "Teamsters Local 705",
    "UAW Local 600",
])
cog.outl("```python")
cog.outl("from labor_union_parser import Extractor")
cog.outl("")
cog.outl("extractor = Extractor()")
cog.outl("results = extractor.extract_batch([")
cog.outl('    "SEIU Local 1199",')
cog.outl('    "Teamsters Local 705",')
cog.outl('    "UAW Local 600",')
cog.outl("])")
for r in results:
    for line in pprint.pformat(r, width=72).splitlines():
        cog.outl(f"# {line}")
cog.outl("```")
]]]-->
```python
from labor_union_parser import Extractor

extractor = Extractor()
results = extractor.extract_batch([
    "SEIU Local 1199",
    "Teamsters Local 705",
    "UAW Local 600",
])
# {'f_num': 31847,
#  'f_num_score': 0.950072705745697,
#  'is_union': True,
#  'is_union_score': 0.9268560409545898,
#  'union_name': 'SERVICE EMPLOYEES',
#  'union_name_score': 0.9972871541976929}
# {'f_num': 43508,
#  'f_num_score': 0.9926707744598389,
#  'is_union': True,
#  'is_union_score': 0.9246779680252075,
#  'union_name': 'TEAMSTERS',
#  'union_name_score': 0.9981544613838196}
# {'f_num': 13030,
#  'f_num_score': 0.993687093257904,
#  'is_union': True,
#  'is_union_score': 0.8813596367835999,
#  'union_name': 'AUTO WORKERS AFL-CIO',
#  'union_name_score': 0.99698406457901}
```
<!--[[[end]]]-->

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

<!--[[[cog
import subprocess
result = subprocess.run(
    'echo "SEIU Local 1199" | labor-union-parser --no-header',
    shell=True, capture_output=True, text=True
)
cog.outl("```bash")
cog.outl("# Process CSV file")
cog.outl("labor-union-parser unions.csv -c union_name -o results.csv")
cog.outl("")
cog.outl("# Process from stdin")
cog.outl('echo "SEIU Local 1199" | labor-union-parser --no-header')
for line in result.stdout.strip().splitlines():
    cog.outl(line)
cog.outl("```")
]]]-->
```bash
# Process CSV file
labor-union-parser unions.csv -c union_name -o results.csv

# Process from stdin
echo "SEIU Local 1199" | labor-union-parser --no-header
```
<!--[[[end]]]-->

## Output Fields

| Field | Description |
|-------|-------------|
| `is_union` | Whether the text is detected as a union name |
| `is_union_score` | Calibrated probability of being a union (0-1, Platt-scaled) |
| `union_name` | Predicted parent union name from the shared classification head |
| `union_name_score` | Softmax probability of the predicted `union_name` (0-1) |
| `f_num` | OLMS filing number of the best-matching gazetteer record |
| `f_num_score` | Softmax probability of best gazetteer match (0-1) |

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
│  is_union_score = 0.99 → is_union = True          │
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
         f_num: 31847, f_num_score: 0.96, ...}
```

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
End-to-end on held-out test data (4,437 examples
scored against the full 44K-record gazetteer):

| Metric | Score |
|--------|-------|
| Accuracy | 97.8% |
| f_num accuracy (union examples) | 98.3% (3804/3868) |
| f_num accuracy (in-vocab only) | 98.3% |
| union_name accuracy | 97.8% (4665/4771) |
| Wrong match (union, wrong f_num) | 64 |
| False negatives (union missed) | 8 |
| False positives (non-union matched) | 27 |
<!--[[[end]]]-->
