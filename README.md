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
3. Save the best model to `src/labor_union_parser/weights/char_cnn.pt`

### Training Data Statistics

| Split | Examples |
|-------|----------|
| Train | 82,900   |
| Val   | 2,296    |
| Test  | 2,413    |

## Model Architecture

The model uses a CharCNN architecture with pointer-based designation selection:

```
Input: "SEIU Local 1199"
         │
         ▼
┌─────────────────────────────┐
│  Tokenizer                  │
│  ["SEIU", " ", "Local", " ", "1199"]
│  token_type: [word, space, word, space, number]
└─────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌──────────┐  ┌──────────┐
│ CharCNN  │  │ Special  │
│ (words)  │  │ Embed    │
└────┬─────┘  └────┬─────┘
     └─────┬───────┘
           ▼
┌─────────────────────────────┐
│  Token Embeddings (64-dim)  │
│  + is_number feature (16-dim)│
└─────────────────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Self-Attention (4 heads)   │
└─────────────────────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌─────────┐  ┌──────────────┐
│ Set     │  │ BiLSTM       │
│ Attn    │  │ (512-dim)    │
│ Pooling │  └──────┬───────┘
└────┬────┘         │
     │              ▼
     │     ┌───────────────────┐
     │     │ + Aff Embedding   │
     │     │ Pointer Selection │
     │     └─────────┬─────────┘
     ▼               ▼
Affiliation     Designation
  "SEIU"          "1199"
```

### Components

**Character CNN (for word tokens)**
- Character embedding: 16-dim
- Multi-scale 1D convolutions (kernel sizes 2, 3, 4, 5)
- Max pooling → 64-dim token embedding
- Typo-robust: handles misspellings gracefully

**Special Token Embedding (for non-words)**
- Lookup table for numbers, punctuation, spaces
- 64-dim embeddings

**Token Features**
- `is_number`: Binary feature indicating numeric tokens
- Combined with token embedding via learned 16-dim feature embedding

**Self-Attention**
- Multi-head attention (4 heads) over token sequence
- Allows tokens to attend to each other for context

**Affiliation Classification**
- Set attention pooling: learned weighted sum of token representations
- Linear classifier → affiliation label

**Designation Selection (Pointer Network)**
- BiLSTM processes contextualized token embeddings
- Concatenates with predicted affiliation embedding
- Scores each numeric token position
- Includes learnable "null" score for no designation
- Selects highest-scoring number token

### Model Statistics

- Parameters: ~2.5M
- Inference: CPU or MPS (Apple Silicon)
- Model file: ~10MB

### Performance

On held-out test set (2,413 examples):
- Affiliation accuracy: ~97%
- Designation accuracy: ~98%
