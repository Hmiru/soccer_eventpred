import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

import pytorch_lightning as pl
from _jsonnet import evaluate_file
from sklearn.metrics import accuracy_score
import xgboost as xgb

from soccer_eventpred.data.sources.source import SoccerDataSource
from soccer_eventpred.env import OUTPUT_DIR, PROJECT_DIR
from soccer_eventpred.models.event_predictor import EventPredictor
from soccer_eventpred.modules.class_weight import ClassWeightBase
from soccer_eventpred.modules.datamodule.soccer_datamodule import SoccerDataModule
from soccer_eventpred.util import load_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def normalize_labels(labels, offset):
    """라벨을 0부터 시작하도록 정규화"""
    return labels - offset

def train_xgboost(train_features, train_labels, valid_features, valid_labels):
    # 고유 클래스 확인
    unique_train_labels = set(train_labels)
    unique_valid_labels = set(valid_labels)
    print(f"Unique train labels: {unique_train_labels}")
    print(f"Unique valid labels: {unique_valid_labels}")

    # 클래스 개수 확인
    num_classes = len(unique_train_labels)
    print(f"Number of classes: {num_classes}")

    # 라벨 값 범위 확인
    for label in train_labels:
        if label < 0 or label >= num_classes:
            print(f"Invalid label in train_labels: {label}")
    for label in valid_labels:
        if label < 0 or label >= num_classes:
            print(f"Invalid label in valid_labels: {label}")

    # 데이터셋 생성
    try:
        dtrain = xgb.DMatrix(train_features, label=train_labels)
        dvalid = xgb.DMatrix(valid_features, label=valid_labels)
    except Exception as e:
        print("Error in creating DMatrix:")
        print(f"train_features shape: {train_features.shape}")
        print(f"train_labels shape: {train_labels.shape}")
        print(f"valid_features shape: {valid_features.shape}")
        print(f"valid_labels shape: {valid_labels.shape}")
        raise e

    # XGBoost 파라미터 설정
    params = {
        "objective": "multi:softmax",  # 다중 클래스 분류
        "num_class": num_classes,  # 클래스 수
        "eval_metric": "mlogloss",  # 평가 지표
        "eta": 0.1,  # 학습률
        "max_depth": 6,  # 최대 깊이
        "subsample": 0.8,  # 데이터 샘플링 비율
        "colsample_bytree": 0.8,  # 특징 샘플링 비율
        "seed": 42,  # 랜덤 시드
    }

    eval_list = [(dtrain, "train"), (dvalid, "valid")]

    print("Starting XGBoost training...")
    try:
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,  # 부스팅 반복 횟수
            evals=eval_list,
            early_stopping_rounds=10,  # 조기 종료 설정
            verbose_eval=True,
        )
    except Exception as e:
        print("Error during training:")
        raise e

    # 검증 데이터 평가
    try:
        valid_preds = model.predict(dvalid)
        accuracy = accuracy_score(valid_labels, valid_preds)
        print(f"Validation Accuracy: {accuracy:.4f}")
    except Exception as e:
        print("Error during evaluation:")
        raise e

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-source", type=str, default="wyscout_offense_only")
    parser.add_argument("--data-module", type=str, default="wyscout_single")
    parser.add_argument("--data-name", type=str, default="wyscout_offense_only")
    parser.add_argument(
        "-c",
        "--config",
        type=lambda p: Path(p).absolute(),
        required=True,
        help="path to config file",
    )
    parser.add_argument(
        "--prediction-method",
        type=str,
        default="sequence",
        help="prediction method to use",
    )
    parser.add_argument("-n", "--name", type=str, help="name of the run", required=True)
    parser.add_argument(
        "-e",
        "--exp-name",
        type=str,
        default="default",
        help="name of the experiment in mlflow",
    )
    parser.add_argument(
        "-m",
        "--mapping",
        type=str,
        default=None,
        help="mapping of labels to events if any",
    )
    parser.add_argument(
        "--use-xgboost",
        action="store_true",
        help="Use XGBoost for training instead of the PyTorch model",
    )
    parser.add_argument(
        "--devices", type=int, default=-1, help="devices to use for training"
    )
    parser.add_argument(
        "--strategy", type=str, default=None, help="training strategy to use"
    )
    parser.add_argument(
        "--accelerator", type=str, default="gpu", help="accelerator to use for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of epochs to train for"
    )
    parser.add_argument(
        "--num-workers", type=int, default=8, help="number of workers for dataloader"
    )
    parser.add_argument("-g", "--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--class-weight-type", type=str, default=None)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--loss-function", type=str, default="cross_entropy_loss")
    parser.add_argument("--focal-loss-gamma", type=float, default=2.0)
    parser.add_argument("--val-check-interval", type=float, default=1.0)
    parser.add_argument("--ignore-tokens", type=str, default=None, nargs="+")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--encoder_dim", type=int, default=None)
    parser.add_argument("--sequence_encoder_dim", type=int, default=None)

    args = parser.parse_args()
    if args.encoder_dim is not None and args.sequence_encoder_dim is not None:
        params = json.loads(
            evaluate_file(
                str(args.config),
                ext_vars={
                    "encoder_dim": str(args.encoder_dim),
                    "sequence_encoder_dim": str(args.sequence_encoder_dim),
                },
            )
        )
    else:
        params = json.loads(evaluate_file(str(args.config)))
    if args.learning_rate is not None:
        params["optimizer"]["lr"] = args.learning_rate
    params_copy = deepcopy(params)

    logger.info("Loading data")

    # SoccerDataSource: 전처리 및 split이 완료된 데이터셋을 로드
    # train.jsonl, dev.jsonl, test.jsonl: [competition, wyscout_match_id, wyscout_team_id_1, wyscout_team_id_2, player_list_1, player_list_2, events]
    train_datasource = SoccerDataSource.from_params(
        params_={
            "type": args.data_source,
            "data_name": args.data_name,
            "subset": "train.jsonl",
        }
    )
    val_datasource = SoccerDataSource.from_params(
        params_={
            "type": args.data_source,
            "data_name": args.data_name,
            "subset": "dev.jsonl",
        }
    )
    test_datasource = SoccerDataSource.from_params(
        params_={
            "type": args.data_source,
            "data_name": args.data_name,
            "subset": "test.jsonl",
        }
    )

    # 통합할 이벤트 Dictionary: 각 주요 이벤트(label)마다 관련된 세부 이벤트들이 그룹으로 묶여 있음
    label2events = load_json(args.mapping) if args.mapping is not None else None

    # wyscout_single_event_datamodule(or WyScoutSequenceDataModule) 클래스를 인스턴스화
    # 이 모듈은 데이터셋 로딩, 배치 구성, 이벤트 및 팀/선수 관련 vocab 빌드 기능 등을 포함.
    datamodule = SoccerDataModule.from_params(
        params_={
            "type": args.data_module,
        },
        train_datasource=train_datasource,
        val_datasource=val_datasource,
        test_datasource=test_datasource,

        sequence_length=params["sequence_length"],
        ignore_tokens = args.ignore_tokens,
        label2events=label2events,

        batch_size=params["batch_size"],
        num_workers=args.num_workers,
    )

    logger.info("Preparing datamodule...")
    datamodule.build_vocab()

    logger.info("Preparing data...")
    datamodule.prepare_data()

    logger.info(f"Training dataset size: {len(datamodule._train_dataset)}")
    logger.info(f"Validation dataset size: {len(datamodule._val_dataset) if datamodule._val_dataset else 0}")
    logger.info(f"Test dataset size: {len(datamodule._test_dataset) if datamodule._test_dataset else 0}")

    logger.info("Calculating class weights...")
    if args.class_weight_type is not None:
        if args.class_weight_type == "exponential":
            class_weight_fn = ClassWeightBase.from_params(
                params_={
                    "type": args.class_weight_type,
                    "beta": args.beta,
                }
            )
        else:
            class_weight_fn = ClassWeightBase.from_params(
                params_={
                    "type": args.class_weight_type,
                }
            )
        class_weight = class_weight_fn.calculate(
            dataset=datamodule._train_dataset,
            num_classes=datamodule.vocab.size("events"),
            ignore_indices=[
                int(datamodule.vocab.get(token, namespace="events"))
                for token in args.ignore_tokens
            ]
            if args.ignore_tokens is not None
            else None,
            class_counts=datamodule.event_counts,
        )
    else:
        class_weight = None

    for token in args.ignore_tokens:
        print(f"token={token}: {datamodule.vocab.get(token, namespace='events')}")
    print(f"class_weight = {class_weight}")

    # loss function
    if args.loss_function == "cross_entropy_loss":
        loss_function = {
            "type": "torch::CrossEntropyLoss",
        }
    elif args.loss_function == "focal_loss":
        loss_function = {"type": "FocalLoss", "gamma": args.focal_loss_gamma}
    else:
        raise ValueError(f"Loss function {args.loss_function} not supported")

    # model
    if args.prediction_method == "single":
        model_config = {
            "type": args.prediction_method,
            "seq2vec_encoder": params["seq2vec_encoder"],
        }
    elif args.prediction_method == "single_with_performer":
        model_config = {
            "type": args.prediction_method,
            "seq2vec_encoder": params["seq2vec_encoder"],
        }
    elif args.prediction_method == "sequence":
        model_config = {
            "type": args.prediction_method,
            "seq2seq_encoder": params["seq2seq_encoder"],
        }
    else:
        raise ValueError(f"Invalid model: {args.prediction_method}")
    if args.use_xgboost:
        logger.info("Training XGBoost model...")
        train_features, train_labels, valid_features, valid_labels, test_features, test_labels = datamodule.get_xgboost_features()
        label_offset = min(train_labels.min(), valid_labels.min())
        train_labels = normalize_labels(train_labels, label_offset)
        valid_labels = normalize_labels(valid_labels, label_offset)
        test_labels = normalize_labels(test_labels, label_offset)

        try:
            xgboost_model = train_xgboost(train_features, train_labels, valid_features, valid_labels)

            # 테스트 데이터로 평가
            logger.info("Evaluating XGBoost model on test data...")
            dtest = xgb.DMatrix(test_features, label=test_labels)
            test_preds = xgboost_model.predict(dtest)
            test_accuracy = accuracy_score(test_labels, test_preds)
            logger.info(f"Test Accuracy: {test_accuracy:.4f}")

        except Exception as e:
            logger.error("Error during XGBoost training or evaluation:")
            raise e

    else:
        model = EventPredictor.from_params(
            params_=model_config,
            time_encoder=params["time_encoder"],
            team_encoder=params["team_encoder"],
            event_encoder=params["event_encoder"],
            x_axis_encoder=params["x_axis_encoder"],
            y_axis_encoder=params["y_axis_encoder"],
            datamodule=datamodule,
            optimizer=params["optimizer"],
            loss_function=loss_function,
            player_encoder=params["player_encoder"] if "player_encoder" in params else None,
            scheduler=params["scheduler"] if "scheduler" in params else None,
            class_weight=class_weight,
            ignore_tokens= args.ignore_tokens
        )
        output_dir = OUTPUT_DIR / args.name
        chackpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="valid_loss",
            dirpath=output_dir,
            mode="min",
            save_top_k=1,
        )
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor="valid_loss",
            patience=20,
            mode="min",
            min_delta=1e-10,
        )
        mlflow_logger = pl.loggers.MLFlowLogger(
            experiment_name=args.exp_name,
            run_name=args.name,
            save_dir=str(PROJECT_DIR / "mlruns"),
        )
        mlflow_logger.log_hyperparams(params_copy)
        trainer = pl.Trainer(
            accelerator=args.accelerator,
            strategy=args.strategy,
            devices=args.devices,
            callbacks=[chackpoint_callback, early_stopping_callback],
            logger=[mlflow_logger],
            deterministic=True,
            max_epochs=args.epochs,
            accumulate_grad_batches=args.gradient_accumulation_steps,
            val_check_interval=args.val_check_interval,
            detect_anomaly=True,
        )
        trainer.fit(model, datamodule=datamodule)
