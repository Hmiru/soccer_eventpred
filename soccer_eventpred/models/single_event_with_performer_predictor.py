from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tango.integrations.torch import LRScheduler, Optimizer
from typing import List, Tuple
from soccer_eventpred.data.dataclass import SingleEventBatch
from soccer_eventpred.data.vocabulary import PAD_TOKEN
from soccer_eventpred.models.event_predictor import EventPredictor
from soccer_eventpred.modules.datamodule.soccer_datamodule import SoccerDataModule
from soccer_eventpred.modules.seq2vec_encoder.seq2vec_encoder import Seq2VecEncoder
from soccer_eventpred.modules.token_embedder.token_embedder import TokenEmbedder
from soccer_eventpred.torch.loss_function import LossFunction
from soccer_eventpred.torch.metrics.classification import (
    get_classification_full_metrics,
)


@EventPredictor.register("single_with_performer")
class SingleEventWithPerformerPredictor(EventPredictor):
    def __init__(
        self,
        time_encoder: Dict[str, Any],
        team_encoder: Dict[str, Any],
        event_encoder: Dict[str, Any],
        x_axis_encoder: Dict[str, Any],
        y_axis_encoder: Dict[str, Any],
        seq2vec_encoder: Dict[str, Any],
        datamodule: SoccerDataModule,
        optimizer: Dict[str, Any],
        loss_function: Dict[str, Any],
        player_encoder: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Dict[str, Any]] = None,
        class_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self._time_encoder = TokenEmbedder.from_params(params_=time_encoder)
        self._team_encoder = TokenEmbedder.from_params(
            params_=team_encoder,
            num_embeddings=datamodule.vocab.size("teams"),
            padding_idx=datamodule.vocab.get(PAD_TOKEN, namespace="teams"),
        )
        self._event_encoder = TokenEmbedder.from_params(
            params_=event_encoder,
            num_embeddings=datamodule.vocab.size("events"),
            padding_idx=datamodule.vocab.get(PAD_TOKEN, namespace="events"),
        )
        self._x_axis_encoder = TokenEmbedder.from_params(params_=x_axis_encoder)
        self._y_axis_encoder = TokenEmbedder.from_params(params_=y_axis_encoder)
        self._seq2vec_encoder = Seq2VecEncoder.from_params(params_=seq2vec_encoder)
        if player_encoder is not None:
            self._player_encoder = TokenEmbedder.from_params(
                params_=player_encoder,
                num_embeddings=datamodule.vocab.size("players"),
                padding_idx=datamodule.vocab.get(PAD_TOKEN, namespace="players"),
            )
        else:
            self._player_encoder = None
        self._event_projection = nn.Linear(
            self._seq2vec_encoder.get_output_dim(), self._event_encoder.get_input_dim()
        )
        self._datamodule = datamodule
        self._num_classes = self._datamodule.vocab.size("events")
        self._pad_idx = self._datamodule.vocab.get(PAD_TOKEN, namespace="events")
        self._optimizer_config = optimizer
        self._scheduler_config = scheduler
        self.train_metrics = get_classification_full_metrics(
            num_classes=self._num_classes,
            average_method="macro",
            ignore_index=self._pad_idx,
            prefix="train_",
        )
        self.valid_metrics = get_classification_full_metrics(
            num_classes=self._num_classes,
            average_method="macro",
            ignore_index=self._pad_idx,
            prefix="valid_",
        )
        self.test_metrics = get_classification_full_metrics(
            num_classes=self._num_classes,
            average_method="macro",
            ignore_index=self._pad_idx,
            prefix="test_",
        )
        self._class_weight = class_weight
        self.loss_fn = LossFunction.from_params(
            params_=loss_function, weight=self._class_weight
        )
        self.save_hyperparameters(
            "time_encoder",
            "team_encoder",
            "event_encoder",
            "x_axis_encoder",
            "y_axis_encoder",
            "seq2vec_encoder",
            "optimizer",
            "player_encoder",
            "scheduler",
            "class_weight",
        )

    def forward(self, batch: SingleEventBatch) -> Any:

        print("Time Encoder Output:", self._time_encoder(batch.event_times).shape)
        print("Team Encoder Output:", self._team_encoder(batch.team_ids).shape)
        print("Event Encoder Output:", self._event_encoder(batch.event_ids).shape)


        if self._player_encoder is not None:
            print("Player Encoder Output:", self._player_encoder(batch.player_ids).shape)
            print("X Axis Encoder Output:", self._x_axis_encoder(batch.start_pos_x).shape)
            print("Y Axis Encoder Output:", self._y_axis_encoder(batch.start_pos_y).shape)
            embeddings = self._seq2vec_encoder(
                inputs=torch.cat(
                    (
                        self._time_encoder(batch.event_times),
                        self._team_encoder(batch.team_ids),
                        self._event_encoder(
                            torch.cat(
                                [
                                    batch.event_ids[:, :-1],
                                    torch.full(
                                        size=(batch.event_ids.shape[0], 1),
                                        fill_value=self._datamodule.vocab.get(
                                            PAD_TOKEN, namespace="events"
                                        ),
                                        dtype=torch.long,
                                        device=batch.event_ids.device,
                                    ),
                                ],
                                dim=1,
                            )
                        ),
                        self._player_encoder(batch.player_ids) if self._player_encoder is not None else None,
                        torch.cat(
                            (self._x_axis_encoder(batch.start_pos_x), self._y_axis_encoder(batch.end_pos_x)),
                            dim=2
                        ),
                        torch.cat(
                            (self._x_axis_encoder(batch.start_pos_y), self._y_axis_encoder(batch.end_pos_y)),
                            dim=2
                        ),
                    ),
                    dim=2,
                ),
                mask=batch.mask,
            )
        else:
            embeddings = self._seq2vec_encoder(
                inputs=torch.cat(
                    (
                        self._time_encoder(batch.event_times),
                        self._team_encoder(batch.team_ids),
                        self._event_encoder(
                            torch.cat(
                                [
                                    batch.event_ids[:, :-1],
                                    torch.full(
                                        size=(batch.event_ids.shape[0], 1),
                                        fill_value=self._datamodule.vocab.get(
                                            PAD_TOKEN, namespace="events"
                                        ),
                                        dtype=torch.long,
                                        device=batch.event_ids.device,
                                    ),
                                ],
                                dim=1,
                            )
                        ),
                        self._x_axis_encoder(batch.start_pos_x),
                        self._y_axis_encoder(batch.start_pos_y),
                    ),
                    dim=2,
                ),
                mask=batch.mask,
            )
        assert embeddings.dim() == 2
        assert embeddings.shape[0] == batch.event_ids.shape[0]
        assert embeddings.shape[1] == self._seq2vec_encoder.get_output_dim()
        embeddings = torch.tanh(embeddings)
        output = self._event_projection(embeddings)
        print("Batch Team IDs:", batch.team_ids.shape)
        print("Batch Event IDs:", batch.event_ids.shape)
        print("Batch Player IDs:", batch.player_ids.shape)
        print("Batch Start Pos X:", batch.start_pos_x.shape)
        print("Batch Start Pos Y:", batch.start_pos_y.shape)
        print("Batch End Pos X:", batch.end_pos_x.shape)
        print("Batch End Pos Y:", batch.end_pos_y.shape)
        return output

    def training_step(
        self, batch: SingleEventBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        output = self.forward(batch)
        assert output.shape[0] == batch.event_ids.shape[0]
        assert output.shape[1] == self._num_classes

        loss = self.loss_fn(F.softmax(output, dim=1), batch.labels)
        loss = torch.mean(loss)
        pred = torch.argmax(F.softmax(output, dim=1), dim=1)
        self.train_metrics(pred, batch.labels)
        self.log("train_loss", loss)
        self.log_dict(self.train_metrics)  # type: ignore
        return loss

    def validation_step(
        self, batch: SingleEventBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        output = self.forward(batch)
        assert output.shape[0] == batch.event_ids.shape[0]
        assert output.shape[1] == self._num_classes
        assert batch.labels.shape[0] == batch.event_ids.shape[0]

        loss = self.loss_fn(F.softmax(output, dim=1), batch.labels)
        loss = torch.mean(loss)
        pred = torch.argmax(F.softmax(output, dim=1), dim=1)
        self.valid_metrics(pred, batch.labels)
        self.log("valid_loss", loss)
        self.log_dict(self.valid_metrics)  # type: ignore
        return loss

    def test_step(
        self, batch: SingleEventBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        output = self.forward(batch)
        assert output.shape[0] == batch.event_ids.shape[0]
        assert output.shape[1] == self._num_classes
        assert batch.labels.shape[0] == batch.event_ids.shape[0]

        loss = self.loss_fn(F.softmax(output, dim=1), batch.labels)
        loss = torch.mean(loss)
        pred = torch.argmax(F.softmax(output, dim=1), dim=1)
        self.test_metrics(pred, batch.labels)
        self.log("test_loss", loss)
        self.log_dict(self.test_metrics)  # type: ignore
        return loss

    def predict_step(
        self, batch: SingleEventBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        output = self.forward(batch)
        pred = torch.argmax(F.softmax(output, dim=1), dim=1)
        return pred

    def predict(self, batch: SingleEventBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.forward(batch)
        pred = torch.argmax(F.softmax(output, dim=1), dim=1)
        gold = batch.labels
        return gold, pred

    def configure_optimizers(
        self,
    ) -> Optimizer | Tuple[List[Optimizer], List[LRScheduler]]:
        self._optimizer = Optimizer.from_params(
            params_=self._optimizer_config, params=self.parameters()
        )
        if self._scheduler_config is None:
            return self._optimizer
        self._scheduler = LRScheduler.from_params(
            params_=self._scheduler_config, optimizer=self._optimizer
        )
        return [self._optimizer], [self._scheduler]
