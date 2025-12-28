# Labor Union Parser

Extract affiliation and local designation from labor union name strings.

Given an input like `"SEIU Local 1199"`, the parser returns:
- **Affiliation**: `SEIU` (Service Employees International Union)
- **Designation**: `1199` (local number)

## Installation

```bash
pip install -e .
```

## Usage

### Command Line

```bash
# Basic usage
labor-union-parser "SEIU Local 1199"
# Output:
# Affiliation: SEIU
# Designation: 1199

# JSON output
labor-union-parser "Teamsters Local 705" --json
# Output: {"affiliation": "IBT", "designation": "705"}
```

### Python API

```python
from labor_union_parser import extract

result = extract("UAW Local 600")
print(result)
# {'affiliation': 'UAW', 'designation': '600'}
```

For batch processing:

```python
from labor_union_parser import Extractor

extractor = Extractor()
results = extractor.extract_batch([
    "SEIU Local 1199",
    "Teamsters Local 705",
    "UAW Local 600",
])
# [{'affiliation': 'SEIU', 'designation': '1199'},
#  {'affiliation': 'IBT', 'designation': '705'},
#  {'affiliation': 'UAW', 'designation': '600'}]
```

For large datasets, use `extract_all` which yields results as a generator:

```python
from labor_union_parser import Extractor

extractor = Extractor()

# Process large list with progress bar
for result in extractor.extract_all(union_names, show_progress=True):
    print(result)

# Adjust batch size for memory/speed tradeoff
results = list(extractor.extract_all(union_names, batch_size=512))
```

## Training

Training data is in `training/data/labeled_data.csv` with columns:
- `text`: Union name string
- `aff_abbr`: Affiliation abbreviation (e.g., "SEIU", "IBT", "UAW")
- `desig_num`: Local designation number
- `split`: One of "train", "val", or "test"

To retrain the model:

```bash
pip install -e ".[train]"  # Install training dependencies
python training/train.py
```

The training script will:
1. Load data from `training/data/labeled_data.csv`
2. Train for 10 epochs with early stopping based on validation performance
3. Save the best model to `src/labor_union_parser/weights/bilstm_bio_crf.pt`

### Training Data Statistics

| Split | Examples |
|-------|----------|
| Train | 82,900   |
| Val   | 2,296    |
| Test  | 2,413    |

## Model Architecture

The model uses a multi-task BiLSTM-CRF architecture with shared token embeddings:

```
Input: "SEIU Local 1199"
         │
         ▼
┌─────────────────────┐
│  Token Embedding    │  (64-dim, shared)
│  [seiu, local, 1, 1, 9, 9]
└─────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌──────────┐
│ Conv  │  │ BiLSTM   │
│ + Pool│  │ (512-dim)│
└───┬───┘  └────┬─────┘
    │           │
    ▼           ▼
┌───────┐  ┌──────────────────┐
│ Aff   │  │ Concat with      │
│ Class │  │ Aff Embedding    │
└───┬───┘  └────────┬─────────┘
    │               │
    │               ▼
    │      ┌──────────────┐
    │      │ CRF Layer    │
    │      │ (BIO tagging)│
    │      └──────┬───────┘
    │             │
    ▼             ▼
Affiliation   Designation
  "SEIU"        "1199"
```

### Components

**Token Embedding (shared)**
- Vocabulary: ~2,500 tokens (words, individual digits, punctuation)
- Embedding dimension: 64

**Affiliation Classification Branch**
- 2-layer 1D convolution (kernel size 3)
- Global max pooling
- Linear classifier → affiliation label

**Designation Extraction Branch**
- Bidirectional LSTM (512 hidden units)
- Concatenated with affiliation embedding (64-dim) for gating
- Linear emission layer → BIO tag logits
- CRF layer for sequence decoding

**BIO Tagging Scheme**
- `O`: Outside designation span
- `B`: Beginning of designation (first digit)
- `I`: Inside designation (subsequent digits)

The CRF layer enforces valid tag sequences (e.g., I cannot follow O).

### Model Statistics

- Parameters: ~14M
- Inference: CPU (no GPU required)
- Model file: 14MB

### Performance

On held-out test set (2,413 examples):
- Affiliation accuracy: ~97%
- Designation accuracy: ~98%
