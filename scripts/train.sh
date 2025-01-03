python scripts/train.py \
--data-name "wyscout_offense_only" \
--data-module "wyscout" \
--config configs/model.jsonnet \
--mapping configs/label2events_seq2event_offense_only.json \
--exp-name "test" \
--name "train_run" \
--epochs 200 \
--gradient-accumulation-steps 1 \
--class-weight-type "exponential" \
--beta 0.9 \
--accelerator "gpu" \
--devices 1 \
--prediction-method "predictor" \
--strategy "auto" \
--ignore-tokens "Change of possession" "[UNK]" "[PAD]" "Goal"

python scripts/train.py \
--data-name "wyscout_offense_only" \
--data-module "wyscout_sequence" \
--config configs/sample_model.jsonnet \
--mapping configs/label2events_seq2event_offense_only.json \
--exp-name "test" \
--name "train_run" \
--epochs 100 \
--gradient-accumulation-steps 1 \
--class-weight-type "exponential" \
--beta 0.9 \
--accelerator "gpu" \
--devices 1 \
--prediction-method "sequence" \
--strategy "auto" \
--ignore-tokens "Change of possession" "[UNK]" "[PAD]"

python scripts/evaluate.py \
--data-name "wyscout_offense_only" \
--config configs/model.jsonnet \
--mapping configs/label2events_seq2event_offense_only.json \
--run-name "train_run" \
--class-weight-type "exponential" \
--beta 0.9 \
--prediction-method "predictor" \
--ignore-tokens "Change of possession" "[UNK]" "[PAD]" "Goal"

python scripts/evaluate.py \
--data-name "wyscout_offense_only" \
--config configs/sample_model.jsonnet \
--mapping configs/label2events_seq2event_offense_only.json \
--run-name "test_run" \
--class-weight-type "exponential" \
--beta 0.9 \
--prediction-method "sequence" \
--ignore-tokens "Change of possession" "[UNK]" "[PAD]"