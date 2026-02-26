# Paths
DB_URL = https://github.com/labordata/opdr/releases/download/2021-05-31/opdr.db.zip
WEIGHTS_DIR = src/labor_union_parser/weights
DATA_DIR = training/data

.PHONY: all
all: weights

.PHONY: evaluate
evaluate: weights
	python training/eval_factored_scoring.py
	python training/eval_union_detector.py
	python training/evaluate.py

.PHONY: weights
weights: $(WEIGHTS_DIR)/structured_classifier.pt \
         $(WEIGHTS_DIR)/union_detector.pt \
         $(WEIGHTS_DIR)/scoring_weights.pt

.PHONY: train
train: $(WEIGHTS_DIR)/structured_classifier.pt $(WEIGHTS_DIR)/union_detector.pt

# Bundle scoring weights for production
$(WEIGHTS_DIR)/scoring_weights.pt : $(DATA_DIR)/temperatures.json \
                                    $(WEIGHTS_DIR)/scoring_layer.ckpt
	python training/bundle_scoring_weights.py

# Train scoring layer
.INTERMEDIATE: $(WEIGHTS_DIR)/scoring_layer.ckpt
$(WEIGHTS_DIR)/scoring_layer.ckpt : $(WEIGHTS_DIR)/structured_classifier.pt \
                                    $(DATA_DIR)/training_examples.json
	python training/train_scoring_layer.py

# Fit per-head temperatures
.INTERMEDIATE: $(DATA_DIR)/temperatures.json
$(DATA_DIR)/temperatures.json : $(WEIGHTS_DIR)/structured_classifier.pt
	cd training && python fit_temperatures.py

# Bundle trained model with gazetteer and fnum counts
$(WEIGHTS_DIR)/structured_classifier.pt : $(DATA_DIR)/structured_classifier.ckpt \
                                          $(DATA_DIR)/gazetteer.json \
                                          $(DATA_DIR)/training_examples.json
	python training/bundle_structured_classifier.py

# Train structured classifier (Lightning checkpoint)
.INTERMEDIATE: $(DATA_DIR)/structured_classifier.ckpt
$(DATA_DIR)/structured_classifier.ckpt : $(DATA_DIR)/training_examples.json
	python training/train_structured_classifier.py

# Train union detector
$(WEIGHTS_DIR)/union_detector.pt : $(DATA_DIR)/labeled_data.csv \
                                   $(DATA_DIR)/nonunion_examples.csv
	python training/train_union_detector.py

.PHONY: data
data: $(DATA_DIR)/training_examples.json

$(DATA_DIR)/training_examples.json : $(DATA_DIR)/gazetteer.json \
                                     $(DATA_DIR)/vocabularies.json \
                                     $(DATA_DIR)/unaff_synthetic.csv \
                                     $(DATA_DIR)/labeled_data.csv
	python training/prepare_data.py

.INTERMEDIATE: $(DATA_DIR)/vocabularies.json
$(DATA_DIR)/vocabularies.json : $(DATA_DIR)/gazetteer.json
	python training/generate_vocabularies.py

.INTERMEDIATE: $(DATA_DIR)/unaff_synthetic.csv
$(DATA_DIR)/unaff_synthetic.csv : $(DATA_DIR)/opdr.db $(DATA_DIR)/acronym_to_fullname.csv
	python training/generate_unaff_synthetic.py

.INTERMEDIATE: $(DATA_DIR)/gazetteer.json
$(DATA_DIR)/gazetteer.json : $(DATA_DIR)/opdr.db $(DATA_DIR)/fnum_to_unit_identifier.csv
	python training/generate_fnum_records.py

.INTERMEDIATE: $(DATA_DIR)/fnum_to_unit_identifier.csv
$(DATA_DIR)/fnum_to_unit_identifier.csv : $(DATA_DIR)/opdr.db
	python training/build_unit_identifiers.py

# Download and extract opdr.db
$(DATA_DIR)/opdr.db:
	curl -L -o $(DATA_DIR)/opdr.db.zip $(DB_URL)
	unzip -o $(DATA_DIR)/opdr.db.zip -d $(DATA_DIR)
	rm $(DATA_DIR)/opdr.db.zip
	touch $@

.PHONY: clean
clean:
	-rm $(DATA_DIR)/vocabularies.json
	-rm $(DATA_DIR)/unaff_synthetic.csv
	-rm $(DATA_DIR)/training_examples.json
	-rm $(DATA_DIR)/gazetteer.json
	-rm $(DATA_DIR)/fnum_to_unit_identifier.csv
	-rm $(DATA_DIR)/opdr.db
