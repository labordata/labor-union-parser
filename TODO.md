# Learned Scoring Layer TODO

Replace the hand-tuned gazetteer scoring with a learned linear scoring layer trained end-to-end on the factored classifier outputs.

## 1. Trainable scoring layer

- Save learned weights from `train_scoring_layer.py` to a checkpoint/JSON
- The scoring layer is `nn.Linear(13, 1, bias=False)` trained with pairwise hinge loss + L2 regularization on combined val+test
- 13 features: 6 per-field log-probs, 3 unknown indicators, 3 not-found indicators, log(1+fnum_count)
- Current best: 88 test errors (vs 73 hand-tuned baseline) — gap likely from non-linear fnum blending the hand-tuned approach uses

## 2. Simplify pointer head fallbacks

- The pointer heads (desig_num, prefix, suffix) currently have hand-tuned `POINTER_NOT_FOUND_LOG_PROB` fallback values baked into `scoring.py`
- With the learned scoring layer, not-found cases are handled by the indicator features and their learned penalties, so the fallback log-probs can be removed
- The pointer log-prob feature is zeroed out when not found; the `notfound_*` indicator carries the penalty instead
- Simplify `_score_gazetteer` in `extractor.py` and `scoring.py` accordingly

## 3. Prune unused vocabulary classes from classification heads

- union_name and f_num heads include classes that never appear in training data
- These inflate softmax size, slow down inference, and add trainable parameters that never get meaningful gradients
- Audit vocab sizes vs actual training counts; remove classes with zero training examples
- Mainly a speed/size optimization — fewer parameters to train, smaller softmax at inference
- Check that pruning doesn't break the gazetteer lookup (records referencing pruned classes become "unknown")

## 4. Integrate into extraction pipeline and Makefile

- Load learned scoring weights in `extractor.py` `_score_gazetteer`
- Replace the hand-tuned per-record fnum weighting and field summation with feature construction + linear layer forward pass
- Update `bundle_structured_classifier.py` to include scoring weights in the production checkpoint
- Add `train_scoring_layer.py` step to the Makefile training pipeline (runs after classifier training)
- Update `eval_factored_scoring.py` to support `--use-learned-weights` mode for apples-to-apples comparison
