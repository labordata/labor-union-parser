# Paths
DB_URL = https://github.com/labordata/opdr/releases/download/2021-05-31/opdr.db.zip
WEIGHTS_DIR = src/labor_union_parser/weights
DATA_DIR = training/data

.PHONY: all
all: weights readme

.PHONY: evaluate
evaluate: weights
	python training/eval_factored_scoring.py
	python training/eval_union_detector.py
	python training/evaluate.py

.PHONY: readme
readme: weights
	cog -r README.md

.PHONY: weights
weights: $(WEIGHTS_DIR)/structured_classifier.pt \
         $(WEIGHTS_DIR)/union_detector.pt \
         $(WEIGHTS_DIR)/scoring_weights.pt

.PHONY: train
train: $(WEIGHTS_DIR)/structured_classifier.pt $(WEIGHTS_DIR)/union_detector.pt

PRECOMPUTED_SENTINEL = $(DATA_DIR)/precomputed_features/.done

# Bundle scoring weights for production
$(WEIGHTS_DIR)/scoring_weights.pt : $(DATA_DIR)/temperatures.json \
                                    $(WEIGHTS_DIR)/scoring_layer.pt
	python training/bundle_scoring_weights.py

# Train scoring layer (depends on precomputed features)
.SECONDARY: $(WEIGHTS_DIR)/scoring_layer.pt
$(WEIGHTS_DIR)/scoring_layer.pt : $(PRECOMPUTED_SENTINEL) \
                                  $(DATA_DIR)/training_examples.json
	python training/train_scoring_layer.py

# Precompute features (depends on bundled classifier + temperatures)
$(PRECOMPUTED_SENTINEL) : $(WEIGHTS_DIR)/structured_classifier.pt \
                          $(DATA_DIR)/temperatures.json \
                          $(DATA_DIR)/training_examples.json
	rm -rf $(DATA_DIR)/precomputed_features
	python training/precompute_features.py
	touch $@

# Fit per-head temperatures
.SECONDARY: $(DATA_DIR)/temperatures.json
$(DATA_DIR)/temperatures.json : $(WEIGHTS_DIR)/structured_classifier.pt
	cd training && python fit_temperatures.py

# Bundle trained model with gazetteer and fnum counts
$(WEIGHTS_DIR)/structured_classifier.pt : $(DATA_DIR)/structured_classifier.ckpt \
                                          $(DATA_DIR)/gazetteer.json \
                                          $(DATA_DIR)/training_examples.json
	python training/bundle_structured_classifier.py

# Train structured classifier (Lightning checkpoint)
.SECONDARY: $(DATA_DIR)/structured_classifier.ckpt
$(DATA_DIR)/structured_classifier.ckpt : $(DATA_DIR)/training_examples.json
	python training/train_structured_classifier.py

# Train union detector
$(WEIGHTS_DIR)/union_detector.pt : $(DATA_DIR)/training_examples.json
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

.PHONY: clean-training
clean-training:
	-rm $(DATA_DIR)/structured_classifier*.ckpt
	-rm -rf $(DATA_DIR)/precomputed_features
	-rm -rf $(DATA_DIR)/lightning_logs
	-rm -rf training/lightning_logs

.PHONY: clean
clean:
	-rm $(DATA_DIR)/vocabularies.json
	-rm $(DATA_DIR)/unaff_synthetic.csv
	-rm $(DATA_DIR)/training_examples.json
	-rm $(DATA_DIR)/gazetteer.json
	-rm $(DATA_DIR)/fnum_to_unit_identifier.csv
	-rm $(DATA_DIR)/opdr.db
