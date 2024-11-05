from typing import Any, List, Dict, Optional

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


@EventPredictor.register("sequence")
class WyScoutSequenceEventPredictor(EventPredictor):
    def __init__(
        self,
        time_encoder: Dict[str, Any],
        team_encoder: Dict[str, Any],
        event_encoder: Dict[str, Any],
        x_axis_encoder: Dict[str, Any],
        y_axis_encoder: Dict[str, Any],
        seq2seq_encoder: Dict[str, Any],
        datamodule: SoccerDataModule,
        optimizer: Dict[str, Any],
        loss_function: Dict[str, Any],
        player_encoder: Optional[Dict[str, Any]] = None,
        scheduler: Optional[Dict[str, Any]] = None,
        class_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self._datamodule = datamodule
        self._num_classes = self._datamodule.vocab.size("events")
        self.num_teams  = self._datamodule.vocab.size("teams")
        self.num_players  = self._datamodule.vocab.size("players")

        self._pad_idx = self._datamodule.vocab.get(PAD_TOKEN, namespace="events")
        
        self._optimizer_config = optimizer
        self._scheduler_config = scheduler
        self._class_weight = class_weight

        self._time_encoder = TokenEmbedder.from_params(params_=time_encoder)
        self._team_encoder = TokenEmbedder.from_params(
            params_=team_encoder,
            num_embeddings=self.num_teams,
            padding_idx=datamodule.vocab.get(PAD_TOKEN, namespace="teams"),
        )
        self._event_encoder = TokenEmbedder.from_params(
            params_=event_encoder,
            num_embeddings=self._num_classes, # 총 이벤트 개수
            padding_idx=self._pad_idx,
        )
        self._x_axis_encoder = TokenEmbedder.from_params(params_=x_axis_encoder)
        self._y_axis_encoder = TokenEmbedder.from_params(params_=y_axis_encoder)
        self._seq2seq_encoder = Seq2SeqEncoder.from_params(params_=seq2seq_encoder)

        if player_encoder is not None:
            self._player_encoder = TokenEmbedder.from_params(
                params_=player_encoder,
                num_embeddings=self.num_players,
                padding_idx=datamodule.vocab.get(PAD_TOKEN, namespace="players"),
            )
        else:
            self._player_encoder = None
        self._event_projection = nn.Linear(
            self._seq2seq_encoder.get_output_dim(), self._event_encoder.get_input_dim()
        )

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

        # 특정정 클래스(target)을 손실 계산에서 제외하는 역할 
        # 모델이 예측한 결과와 정답 레이블 사이의 손실을 계산할 때, ignore_index로 설정된 값은 무시되어 그 값에 대해 손실이 계산되지 않음
        self.loss_fn = LossFunction.from_params(
            params_=loss_function,
            weight=self._class_weight,
            ignore_index=self._pad_idx,
            reduction="none",
        )

        self.save_hyperparameters(
            "time_encoder",
            "team_encoder",
            "event_encoder",
            "x_axis_encoder",
            "y_axis_encoder",
            "seq2seq_encoder",
            "optimizer",
            "player_encoder",
            "scheduler",
            "class_weight",
        )

    def forward(self, batch: Batch) -> Any:
        inputs=torch.cat(
                    (
                        self._time_encoder(batch.event_times[:, :-1]),#all events except the last one
                        self._team_encoder(batch.team_ids[:, :-1]),
                        self._event_encoder(batch.event_ids[:, :-1]),
                        self._player_encoder(batch.player_ids[:, :-1]),
                        self._x_axis_encoder(batch.start_pos_x[:, :-1]),
                        self._y_axis_encoder(batch.start_pos_y[:, :-1]),
                        self._x_axis_encoder(batch.end_pos_x[:, :-1]),
                        self._y_axis_encoder(batch.end_pos_y[:, :-1]),
                    ),
                    dim=2, #concatenate along the last dimension; embedding dim
                )

        if self._player_encoder is not None:
            # embeddings: bs x seq x embed_dim
            embeddings = self._seq2seq_encoder(
                # last 39 events are used as input
                # inputs: bs x seq x all_features(8x10)
                inputs=torch.cat(
                    (
                        self._time_encoder(batch.event_times[:, :-1]),#all events except the last one
                        self._team_encoder(batch.team_ids[:, :-1]),
                        self._event_encoder(batch.event_ids[:, :-1]),
                        self._player_encoder(batch.player_ids[:, :-1]),
                        self._x_axis_encoder(batch.start_pos_x[:, :-1]),
                        self._y_axis_encoder(batch.start_pos_y[:, :-1]),
                        self._x_axis_encoder(batch.end_pos_x[:, :-1]),
                        self._y_axis_encoder(batch.end_pos_y[:, :-1]),
                    ),
                    dim=2, #concatenate along the last dimension; embedding dim
                ),
                mask=batch.mask[:, :-1],
            )
        else:
            embeddings = self._seq2seq_encoder(
                inputs=torch.cat(
                    (
                        self._time_encoder(batch.event_times[:, :-1]),
                        self._team_encoder(batch.team_ids[:, :-1]),
                        self._event_encoder(batch.event_ids[:, :-1]),
                        self._x_axis_encoder(batch.start_pos_x[:, :-1]),
                        self._y_axis_encoder(batch.start_pos_y[:, :-1]),
                        self._x_axis_encoder(batch.end_pos_x[:, :-1]),
                        self._y_axis_encoder(batch.end_pos_y[:, :-1]),
                    ),
                    dim=2,#same as above
                ),
                mask=batch.mask[:, :-1],#mask for all events except the last one
            )

        embeddings = torch.tanh(embeddings) # bs x seq x embed_dim -> bs x seq x embed_dim
        output = self._event_projection(embeddings) # s x seq x embed_dim - > bs x seq x vocab_size(이벤트 총 개수)
        return output

    def training_step(self, batch: Batch, batch_idx: int) -> Any:
        # batch -> bs x self.sequence_length+1 x fs
        # len(batch) = batch_size
        # len(batch.event_ids[0]) = self.sequence_length + 1 
        output = self.forward(batch)              # bs x seq x vocab size(이벤트 총 개수)
        output = F.softmax(output[:, -1], dim=-1) # bs x vocab size(이벤트 총 개수)
        assert output.shape[0] == batch.event_ids.shape[0] # bs
        assert output.shape[1] == self._num_classes        # vocab size

        targets = batch.event_ids[:, -1] #last event as the target, len(targets) = batch_size x 1
        loss = self.loss_fn(output,targets) # calculate loss for the last event, bs x 1
        loss = torch.mean(loss)

        # Apply mask for the last event
        # mask: F(input_t) - Output_t+1간의 loss를 계산할 때, Mask(t+1)=True인 경우에만 활용을 하겠다. 즉, Ouput_t+1이 padding이 아닐 때만 활용
        # loss *= batch.mask[:, -1] #mask for the last event
        # loss = loss.sum() / batch.mask[:, -1:].sum()

        pred = torch.argmax(output, dim=1)# event that has the highest probability
 
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
    
