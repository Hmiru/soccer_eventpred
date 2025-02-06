import xgboost as xgb
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
)
import os
from soccer_eventpred.modules.datamodule.xgboost_datamodule import XGBoostDataModule
import json
import logging
import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostPredictor:
    def __init__(self, params_path="../configs/xgboost_params.json"):
        self.params = self.load_params(params_path)
        self.model = None  


    @staticmethod
    def load_params(json_path):
        """JSON 파일에서 XGBoost 하이퍼파라미터 로드"""
        try:
            with open(json_path, "r") as f:
                params = json.load(f)
            logger.info(f"Loaded parameters from {json_path}")
            return params
        except Exception as e:
            logger.error("Error loading JSON config file:", e)
            raise e

    def prepare_data(self, datamodule):
        logger.info("Checking training dataset...")
        datamodule.prepare_data()
        logger.info("Training XGBoost model...")
        xgboost_datamodule = XGBoostDataModule(datamodule)
        train_features, train_labels, valid_features, valid_labels, test_features, test_labels = xgboost_datamodule.get_xgboost_features()  


        original_train_labels = train_labels.copy()
        original_valid_labels = valid_labels.copy()
        original_test_labels = test_labels.copy()

        label_offset = min(train_labels.min(), valid_labels.min(), test_labels.min())

        train_labels = train_labels-label_offset
        valid_labels = valid_labels-label_offset
        test_labels = test_labels-label_offset

        num_classes = len(set(train_labels))
        print(f"Number of classes: {num_classes}")
        
        # 라벨 매핑 생성 (offset 적용 전 → 후)
        label_mapping = {orig: new for orig, new in zip(original_train_labels, train_labels)}

        unique_labels = sorted(set(train_labels))
        # datamodule의 vocab에서 라벨 이름을 가져오는 코드
        label_names = [datamodule.vocab.get(label, namespace="events") for label in unique_labels]

        # 라벨 숫자와 해당되는 이벤트 출력
        print("\n[Label Mapping]")
        for orig, new in label_mapping.items():
            event_name = datamodule.vocab.get(orig, namespace="events")
            print(f"Original Label {orig} → Offset Label {new} : {event_name}")


        self.params["num_class"] = num_classes
        return train_features, train_labels, valid_features, valid_labels, test_features, test_labels

    def train(self, train_features, train_labels, valid_features, valid_labels):
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
        eval_list=[(dtrain, "train"), (dvalid, "valid")]
        logger.info("Training XGBoost model...")
        self.model=xgb.train(
            self.params,
            dtrain,
            num_boost_round=100,
            evals=eval_list,
            early_stopping_rounds=10,
            verbose_eval=10,
        )    
        logger.info("Training complete.")

    def evaluate(self, test_features, test_labels):    
        try:
            dtest=xgb.DMatrix(test_features, label=test_labels)
            test_preds=self.model.predict(dtest)    
            
            accuracy = accuracy_score(test_labels, test_preds)
            logger.info(f"Test Accuracy: {accuracy:.4f}")

            # 상세 성능 출력
            logger.info("Classification Report:")
            logger.info("\n" + classification_report(test_labels, test_preds))

            # 혼동 행렬 시각화
            self.plot_confusion_matrix(test_labels, test_preds)

        except Exception as e:
            logger.error("Error during evaluation:")
            raise e

    def plot_confusion_matrix(self, y_true, y_pred, save_dir="./confusion_matrices"):
            """혼동 행렬을 저장하는 함수"""
            cm = confusion_matrix(y_true, y_pred)
            
            # 현재 시각을 기반으로 파일명 생성 (예: confusion_matrix_20240206_143015.png)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"confusion_matrix_{timestamp}.png"

            # 저장할 디렉토리가 없으면 생성
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, filename)

            # 혼동 행렬 시각화
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")

            # 이미지 파일 저장
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # 메모리 해제