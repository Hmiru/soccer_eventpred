from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tango.integrations.torch import LRScheduler, Optimizer

from soccer_eventpred.data.dataclass import Batch
from soccer_eventpred.data.vocabulary import PAD_TOKEN
from soccer_eventpred.models.event_predictor import EventPredictor
from soccer_eventpred.modules.datamodule.soccer_datamodule import SoccerDataModule
from soccer_eventpred.modules.seq2seq_encoder.seq2seq_encoder import Seq2SeqEncoder
from soccer_eventpred.modules.token_embedder.token_embedder import TokenEmbedder
from soccer_eventpred.torch.loss_function import LossFunction
from soccer_eventpred.torch.metrics.classification import (
    get_classification_full_metrics,
)


@EventPredictor.register("predictor")
class WyScoutEventPredictor(EventPredictor):
    def __init__(
        self,
        time_embedding: Dict[str, Any],
        team_embedding: Dict[str, Any],
        event_embedding: Dict[str, Any],
        x_axis_embedding: Dict[str, Any],
        y_axis_embedding: Dict[str, Any],
        encoder: Dict[str, Any],
        datamodule: SoccerDataModule,
        optimizer: Dict[str, Any],
        loss_function: Dict[str, Any],
        player_embedding: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Dict[str, Any]] = None,
        class_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self._time_embedding = TokenEmbedder.from_params(params_=time_embedding)
        self._team_embedding = TokenEmbedder.from_params(
            params_=team_embedding,
            num_embeddings=datamodule.vocab.size("teams"),
            padding_idx=datamodule.vocab.get(PAD_TOKEN, namespace="teams"),
        )
        self._event_embedding = TokenEmbedder.from_params(
            params_=event_embedding,
            num_embeddings=datamodule.vocab.size("events"),
            padding_idx=datamodule.vocab.get(PAD_TOKEN, namespace="events"),
        )
        self._x_axis_embedding = TokenEmbedder.from_params(params_=x_axis_embedding)
        self._y_axis_embedding = TokenEmbedder.from_params(params_=y_axis_embedding)
        self._encoder = Seq2SeqEncoder.from_params(params_=encoder)

        if player_embedding is not None:
            self._player_embedding = TokenEmbedder.from_params(
                params_=player_embedding,
                num_embeddings=datamodule.vocab.size("players"),
                padding_idx=datamodule.vocab.get(PAD_TOKEN, namespace="players"),
            )
        else:
            self._player_embedding = None

        self._event_projection = nn.Linear(
            self._encoder.get_output_dim(), self._event_embedding.get_input_dim()
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
            params_=loss_function,
            weight=self._class_weight,
            ignore_index=self._pad_idx,
            reduction="none",
        )

        self.save_hyperparameters(
            "time_embedding",
            "team_embedding",
            "event_embedding",
            "x_axis_embedding",
            "y_axis_embedding",
            "encoder",
            "optimizer",
            "player_embedding",
            "scheduler",
            "class_weight",
        )

    def forward(self, batch: Batch) -> Any:
        if self._player_embedding is not None:
            embeddings = self._encoder(
                # last 39 events are used as input
                inputs=torch.cat(
                    (
                        self._time_embedding(batch.event_times[:, :]),
                        self._team_embedding(batch.team_ids[:, :]),
                        self._event_embedding(batch.event_ids[:, :]),
                        self._player_embedding(batch.player_ids[:, :]),
                        self._x_axis_embedding(batch.start_pos_x[:, :]),
                        self._y_axis_embedding(batch.start_pos_y[:, :]),
                        self._x_axis_embedding(batch.end_pos_x[:, :]),
                        self._y_axis_embedding(batch.end_pos_y[:, :]),
                    ),
                    dim=2, #concatenate along the last dimension; embedding dim
                ),
                mask=batch.mask[:, :],
            )
        else:
            embeddings = self._encoder(
                inputs=torch.cat(
                    (
                        self._time_embedding(batch.event_times[:, :]),
                        self._team_embedding(batch.team_ids[:, :]),
                        self._event_embedding(batch.event_ids[:, :]),
                        self._x_axis_embedding(batch.start_pos_x[:, :]),
                        self._y_axis_embedding(batch.start_pos_y[:, :]),
                        self._x_axis_embedding(batch.end_pos_x[:, :]),
                        self._y_axis_embedding(batch.end_pos_y[:, :]),
                    ),
                    dim=2,#same as above
                ),
                mask=batch.mask[:, :],#mask for all events except the last one
            )

        embeddings = torch.tanh(embeddings) # bs x seq_len x embed
        output = self._event_projection(embeddings) # bs x seq_len x vocab

        return output

    def training_step(self, batch: Batch, batch_idx: int) -> Any:
        output = self.forward(batch)
        targets = batch.labels 

        loss = self.loss_fn(
            F.softmax(output[:, -1], dim=1),  # Apply softmax to the last time step
            targets
        )# calculate loss for the last event

        #loss *= batch.mask[:, -1]
        loss = loss.sum() / batch.mask[:, -1:].sum()

        pred = torch.argmax(F.softmax(output[:, -1], dim=1), dim=1)# event that has the highest probability
        self.train_metrics(pred, targets)
        self.log("train_loss", loss)
        self.log_dict(self.train_metrics)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> Any:
        output = self.forward(batch)

        # Correct target for the last event
        targets = batch.event_ids[:, -1]

        # Use prediction for the last event (40th)
        loss = self.loss_fn(
            F.softmax(output[:, -1], dim=1),
            targets
        )

        # Apply mask for the last event (차원을 맞춰줌)
        loss *= batch.mask[:, -1]

        loss = loss.sum() / batch.mask[:, -1].sum()

        # Calculate predictions for the last event
        pred = torch.argmax(F.softmax(output[:, -1], dim=1), dim=1)


        self.valid_metrics(pred, targets)
        self.log("valid_loss", loss)
        self.log_dict(self.valid_metrics)  # type: ignore
        return loss

    def test_step(self, batch: Batch, batch_idx: int) -> Any:
        output = self.forward(batch)

        # Correct target for the last event
        targets = batch.event_ids[:, -1]

        # Use prediction for the last event (40th)
        loss = self.loss_fn(
            F.softmax(output[:, -1], dim=1),
            targets
        )


        loss *= batch.mask[:, -1]
        loss = loss.sum() / batch.mask[:, -1].sum()

        # Calculate predictions for the last event
        pred = torch.argmax(F.softmax(output[:, -1], dim=1), dim=1)  # dim=1로 수정


        self.test_metrics(pred, targets)
        self.log("test_loss", loss)
        self.log_dict(self.test_metrics)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output = self.forward(batch)
        # Calculate predictions for the last event
        pred = torch.argmax(F.softmax(output[:, -1], dim=1), dim=1)  # dim=1로 수정
        return pred

    def predict(self, batch):
        output = self.forward(batch)

        pred = torch.argmax(F.softmax(output[:, -1], dim=1), dim=1)  # dim=1로 수정

        # Correct target for the last event
        gold = batch.event_ids[:, -1]
        return gold, pred

    def configure_optimizers(self):
        self._optimizer = Optimizer.from_params(
            params_=self._optimizer_config, params=self.parameters()
        )
        if self._scheduler_config is not None:
            self._scheduler = LRScheduler.from_params(
                params_=self._scheduler_config, optimizer=self._optimizer
            )
        return [self._optimizer], [self._scheduler]
