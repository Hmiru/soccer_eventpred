import numpy as np
import torch
from soccer_eventpred.modules.datamodule.wyscout_sequence_datamodule import WyScoutSequenceDataModule
from soccer_eventpred.data.dataclass import Batch


class XGBoostDataModule:
    def __init__(self, datamodule: WyScoutSequenceDataModule):
        self.datamodule = datamodule

    def batch_to_numpy(self, batch: Batch):
        print(f"Converting batch to numpy, batch size: {len(batch.event_times)}")

        features = torch.cat(
            [
                batch.event_times,
                batch.team_ids,
                batch.event_ids,
                batch.player_ids,
                batch.start_pos_x,
                batch.start_pos_y,
                batch.end_pos_x,
                batch.end_pos_y,
            ],
            dim=1,
        ).cpu().numpy()

        labels = batch.labels.cpu().numpy()
        return features.reshape(features.shape[0], -1), labels

    def get_xgboost_features(self):
        train_features, train_labels = self._extract_features_labels(self.datamodule.train_dataloader())
        valid_features, valid_labels = self._extract_features_labels(self.datamodule.val_dataloader())
        test_features, test_labels = self._extract_features_labels(self.datamodule.test_dataloader())
        return train_features, train_labels, valid_features, valid_labels, test_features, test_labels

    def _extract_features_labels(self, dataloader):
        all_features = []
        all_labels = []

        for batch in dataloader:
            features, labels = self.batch_to_numpy(batch)
            all_features.append(features)
            all_labels.append(labels)

        all_features = np.vstack(all_features)
        all_labels = np.hstack(all_labels)

        return all_features, all_labels
