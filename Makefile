.PHONY: all clean download train lookup

# Paths
DB_URL = https://github.com/labordata/opdr/releases/download/2021-05-31/opdr.db.zip
DB_ZIP = opdr.db.zip
DB_FILE = opdr.db
WEIGHTS_DIR = src/labor_union_parser/weights
FNUM_LOOKUP = $(WEIGHTS_DIR)/fnum_lookup.json
MODEL_WEIGHTS = $(WEIGHTS_DIR)/char_cnn.pt

all: $(FNUM_LOOKUP)

# Download and extract opdr.db
$(DB_FILE):
	curl -L -o $(DB_ZIP) $(DB_URL)
	unzip -o $(DB_ZIP)
	rm $(DB_ZIP)
	touch $(DB_FILE)

# Build fnum lookup from database
$(FNUM_LOOKUP): $(DB_FILE)
	python scripts/build_fnum_lookup.py

# Train the model
$(MODEL_WEIGHTS): training/data/labeled_data.csv
	python training/train.py

# Convenience targets
download: $(DB_FILE)

lookup: $(FNUM_LOOKUP)

train: $(MODEL_WEIGHTS)

clean:
	rm -f $(DB_FILE) $(DB_ZIP)
