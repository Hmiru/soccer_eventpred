import numpy as np
import torch
import pandas as pd
from soccer_eventpred.modules.datamodule.wyscout_sequence_datamodule import WyScoutSequenceDataModule
from soccer_eventpred.data.dataclass import Batch

class XGBoostDataModule:
    def __init__(self, datamodule: WyScoutSequenceDataModule):
        self.datamodule = datamodule
        self.first_match = True  # 첫 번째 배치 여부 플래그

    def batch_to_numpy(self, batch: Batch):
        print(f"\n[INFO] Converting batch to numpy, batch size: {len(batch.event_times)}")

        # 사용될 feature 목록
        feature_names = [
            "event_times",
            "team_ids",
            "event_ids",
            "player_ids",
            "start_pos_x",
            "start_pos_y"
        ]

        # 각 피처별 디버깅
        def debug_tensor(name, tensor):
            print(f"\n{name} shape: {tensor.shape}")
            print(f"{name} (first 5 rows): \n{tensor[:5].cpu().numpy()}")

        for name in feature_names:
            debug_tensor(name, getattr(batch, name))

        features = torch.cat(
            [getattr(batch, name) for name in feature_names],
            dim=1,
        ).cpu().numpy()

        labels = batch.labels.cpu().numpy()

        # 첫 번째 배치일 때만 저장
        if self.first_match:
            self._save_to_csv(features, labels, feature_names, "first_batch_features_labels.csv")
            self.first_match = False  # 첫 번째 배치 처리 후 플래그 변경

        return features.reshape(features.shape[0], -1), labels

    def get_xgboost_features(self):
        print("Extracting features for XGBoost...")

        train_features, train_labels = self._extract_features_labels(self.datamodule.train_dataloader())
        print("Training set loaded. Shape:", train_features.shape, train_labels.shape)

        valid_features, valid_labels = self._extract_features_labels(self.datamodule.val_dataloader())
        print("Validation set loaded. Shape:", valid_features.shape, valid_labels.shape)

        test_features, test_labels = self._extract_features_labels(self.datamodule.test_dataloader())
        print("Test set loaded. Shape:", test_features.shape, test_labels.shape)

        return train_features, train_labels, valid_features, valid_labels, test_features, test_labels

    def _extract_features_labels(self, dataloader):
        all_features = []
        all_labels = []

        for i, batch in enumerate(dataloader):
            print(f"\n[INFO] Processing batch {i + 1}...")
            features, labels = self.batch_to_numpy(batch)
            all_features.append(features)
            all_labels.append(labels)

        all_features = np.vstack(all_features)
        all_labels = np.hstack(all_labels)

        print(f"\n[INFO] Final feature shape: {all_features.shape}")
        print(f"Final label shape: {all_labels.shape}")

        return all_features, all_labels

    def _save_to_csv(self, features, labels, feature_names, filename="features_labels.csv"):
        # 각 feature 이름에 대해 _1_frame~_40_frame 추가
        num_frames = features.shape[1] // len(feature_names)  # 프레임 개수 자동 계산
        expanded_columns = [f"{name}_{i+1}_frame" for name in feature_names for i in range(num_frames)]

        df = pd.DataFrame(features, columns=expanded_columns)
        df['label'] = labels
        df.to_csv(filename, index=False)

        print(f"\n[INFO] First batch data saved to {filename}")
