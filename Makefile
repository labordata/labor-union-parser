# Paths
DB_URL = https://github.com/labordata/opdr/releases/download/2021-05-31/opdr.db.zip
F7_URL = https://labordata.github.io/fmcs-f7/f7.db.zip
WEIGHTS_DIR = src/labor_union_parser/weights
DATA_DIR = training/data

.PHONY: all
all: weights readme

.PHONY: evaluate
evaluate: weights
	python training/evaluate.py

.PHONY: readme
readme: weights
	cog -r README.md

.PHONY: weights
weights: $(WEIGHTS_DIR)/arcface_classifier.pt \
         $(WEIGHTS_DIR)/union_detector.pt

.PHONY: train
train: $(WEIGHTS_DIR)/arcface_classifier.pt $(WEIGHTS_DIR)/union_detector.pt

# Bundle ArcFace classifier with gazetteer and fit temperatures
$(WEIGHTS_DIR)/arcface_classifier.pt : $(DATA_DIR)/arcface_classifier.ckpt \
                                       $(DATA_DIR)/gazetteer.json \
                                       $(DATA_DIR)/training_examples.json
	python training/bundle_arcface_classifier.py
	python training/fit_temperature.py

# Train ArcFace classifier (needs gazetteer for frozen OOV prototypes)
.SECONDARY: $(DATA_DIR)/arcface_classifier.ckpt
$(DATA_DIR)/arcface_classifier.ckpt : $(DATA_DIR)/training_examples.json \
                                      $(DATA_DIR)/gazetteer.json
	python training/train_arcface_classifier.py

# Train union detector (uses f7.db employer names as hard negatives)
$(WEIGHTS_DIR)/union_detector.pt : $(DATA_DIR)/training_examples.json \
                                   $(DATA_DIR)/f7.db
	python training/train_union_detector.py

.PHONY: data
data: $(DATA_DIR)/training_examples.json

$(DATA_DIR)/training_examples.json : $(DATA_DIR)/gazetteer.json \
                                     $(DATA_DIR)/vocabularies.json \
                                     $(DATA_DIR)/unaff_synthetic.csv \
                                     $(DATA_DIR)/labeled_data.csv
	python training/prepare_data.py

.SECONDARY: $(DATA_DIR)/vocabularies.json
$(DATA_DIR)/vocabularies.json : $(DATA_DIR)/gazetteer.json
	python training/generate_vocabularies.py

.SECONDARY: $(DATA_DIR)/unaff_synthetic.csv
$(DATA_DIR)/unaff_synthetic.csv : $(DATA_DIR)/opdr.db $(DATA_DIR)/acronym_to_fullname.csv
	python training/generate_unaff_synthetic.py

.SECONDARY: $(DATA_DIR)/gazetteer.json
$(DATA_DIR)/gazetteer.json : $(DATA_DIR)/opdr.db $(DATA_DIR)/fnum_to_unit_identifier.csv
	python training/generate_fnum_records.py

.SECONDARY: $(DATA_DIR)/fnum_to_unit_identifier.csv
$(DATA_DIR)/fnum_to_unit_identifier.csv : $(DATA_DIR)/opdr.db
	python training/build_unit_identifiers.py

# Download and extract opdr.db
$(DATA_DIR)/opdr.db:
	curl -L -o $(DATA_DIR)/opdr.db.zip $(DB_URL)
	unzip -o $(DATA_DIR)/opdr.db.zip -d $(DATA_DIR)
	rm $(DATA_DIR)/opdr.db.zip
	touch $@

# Download and extract f7.db
$(DATA_DIR)/f7.db:
	curl -L -o $(DATA_DIR)/f7.db.zip $(F7_URL)
	unzip -o $(DATA_DIR)/f7.db.zip -d $(DATA_DIR)
	rm $(DATA_DIR)/f7.db.zip
	touch $@

.PHONY: clean-training
clean-training:
	-rm -f $(DATA_DIR)/arcface_classifier.ckpt
	-rm -f $(WEIGHTS_DIR)/arcface_classifier.pt

.PHONY: clean
clean: clean-training
	-rm $(DATA_DIR)/vocabularies.json
	-rm $(DATA_DIR)/unaff_synthetic.csv
	-rm $(DATA_DIR)/training_examples.json
	-rm $(DATA_DIR)/gazetteer.json
	-rm $(DATA_DIR)/fnum_to_unit_identifier.csv
	-rm $(DATA_DIR)/opdr.db
	-rm $(DATA_DIR)/f7.db
