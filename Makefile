.PHONY: all train

# Paths
DB_URL = https://github.com/labordata/opdr/releases/download/2021-05-31/opdr.db.zip
WEIGHTS_DIR = src/labor_union_parser/weights
MODEL_WEIGHTS = $(WEIGHTS_DIR)/char_cnn.pt

all: $(MODEL_WEIGHTS)


training/data/training_examples.json : training/data/unaff_synthetic.csv training/data/labeled_data.csv training/fnum_to_unit_identifier.csv
	python training/prepare_data.py

training/fnum_to_unit_identifier.csv : opdr.db
	python training/build_unit_identifiers.py

training/data/unaff_synthetic.csv : opdr.db training/data/acronym_to_fullname.csv
	python training/generate_unaff_synthetic.py

# Download and extract opdr.db
opdr.db: 
	curl -L -o opdr.db.zip $(DB_URL)
	unzip -o opdr.db.zip
	rm opdr.db.zip
	touch $@


# Train the model
$(MODEL_WEIGHTS): training/data/labeled_data.csv $(WEIGHTS_DIR)/fnum_lookup.json
	python training/train.py

train: $(MODEL_WEIGHTS)

