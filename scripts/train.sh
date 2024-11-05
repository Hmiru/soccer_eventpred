python scripts/train.py \
--data-name "wyscout_offense_only" \
--data-module "wyscout_sequence" \
--config configs/sequence_model.jsonnet \
--mapping configs/label2events_seq2event_offense_only.json \
--exp-name "test" \
--name "train_run" \
--epochs 30 \
--gradient-accumulation-steps 1 \
--class-weight-type "exponential" \
--beta 0.9 \
--num-workers 8 \
--accelerator "gpu" \
--devices 1 \
--prediction-method "sequence" \
--strategy "auto" \
--ignore-tokens "Change of possession" "[UNK]" "[PAD]" "Goal"

python scripts/evaluate.py \
--data-name "wyscout_offense_only" \
--data-module "wyscout_sequence" \
--config configs/sequence_model.jsonnet \
--mapping configs/label2events_seq2event_offense_only.json \
--run-name "train_run" \
--class-weight-type "exponential" \
--beta 0.9 \
--prediction-method "sequence" \
--ignore-tokens "Change of possession" "[UNK]" "[PAD]" "Goal"