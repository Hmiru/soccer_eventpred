# Soccer Event Prediction

## Introduction

This repository contains code for the paper "Leveraging Player Embeddings for Soccer Event Prediction".

## Getting Started

### Environment

Type `make setup` to create a virtual environment and install all dependencies.

### Data

Raw data provided by wyscout is supposed to be placed in `data/wyscout` directory.
The directory structure should be as follows:

```
data/wyscout
└── raw
    ├── events
    │   ├── events_England.json
    │   ├── events_France.json
    │   ├── events_Germany.json
    │   ├── events_Italy.json
    │   ├── events_Spain.json
    │   └── events_World_Cup.json
    └──mappings
        ├── players.json
        ├── tags2name.csv
        └── teams.json
```
[여기](https://github.com/koenvo/wyscout-soccer-match-event-dataset) 에서 데이터를 받아오실 수 있습니다.
A large part of the event preprocessing code is borrowed from [seq2Event](https://github.com/statsonthecloud/Soccer-SEQ2Event).

```
poetry run python scripts/preprocess_data.py teams \
    --input_path data/wyscout_offense_only/raw/mappings/teams.json \
    --output_path data/wyscout_offense_only/preprocessed/mappings/id2team.json

poetry run python scripts/preprocess_data.py players \
    --input_path data/wyscout_offense_only/raw/mappings/players.json \
    --output_path data/wyscout_offense_only/preprocessed/mappings/id2player.json

poetry run python scripts/preprocess_data.py tags \
    --input_path data/wyscout_offense_only/raw/mappings/tags2name.csv \
    --output_path data/wyscout_offense_only/preprocessed/mappings/tagid2name.json

poetry run python scripts/preprocess_data.py events \
    --input_dir data/wyscout_offense_only/raw/events \
    --output_dir data/wyscout_offense_only/preprocessed/events \
    --mappings_dir data/wyscout_offense_only/preprocessed/mappings \
    --targets "events_Spain.json" \
    --offense_only true

poetry run python scripts/preprocess_data.py split \
    --df_pickle_path data/wyscout_offense_only/preprocessed/events/all_preprocessed.pkl \
    --output_dir data/wyscout_offense_only/preprocessed/events \
    --random_state 42
```

### Modeling

```
python ../scripts/train.py --data-name "wyscout_offense_only" \
--config ../configs/sequence_model.jsonnet \
--mapping ../configs/label2events_seq2event_offense_only.json \
--exp-name "test" \
--name "LaLiga_sequence_all_ignored" --epochs 20 --gradient-accumulation-steps 4 \
--class-weight-type "exponential" \
--beta 0.9 --accelerator "gpu" \
--data-module "wyscout_sequence" --devices 1 \
--prediction-method "sequence" \
--strategy "auto" \
--ignore-tokens "Change of possession" "Goal" "[UNK]" "[PAD]" 

```

## Evaluation

```
python ../scripts/evaluate.py --data-name "wyscout_offense_only" \
--config ../configs/sequence_model.jsonnet \
--mapping ../configs/label2events_seq2event_offense_only.json \
--run-name "LaLiga_sequence_all_ignored" --class-weight-type "exponential" \
--beta 0.9 --data-module "wyscout_sequence" \
--prediction-method "sequence" \
--ignore-tokens "Change of possession" "Goal" "[UNK]" "[PAD]"  
```

## Confusion matrix
```
python scripts/confusion_matrix.py \
    --run-name "LaLiga_sequence_all_ignored"
```
