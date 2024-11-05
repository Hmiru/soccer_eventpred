from typing import Optional

import torch
import torch.nn as nn
from tango.common import Registrable


class LossFunction(nn.Module, Registrable):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@LossFunction.register("FocalLoss")
class FocalLoss(LossFunction):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.ignore_index
        # 특정정 클래스(target)을 손실 계산에서 제외하는 역할 
        # 모델이 예측한 결과와 정답 레이블 사이의 손실을 계산할 때, ignore_index로 설정된 값은 무시되어 그 값에 대해 손실이 계산되지 않음
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.weight, ignore_index=self.ignore_index, reduction="none"
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce_loss(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# Register all loss functions
for name, cls in torch.nn.modules.loss.__dict__.items():
    if (
        isinstance(cls, type)
        and (
            issubclass(cls, torch.nn.modules.loss._Loss)
            or issubclass(cls, torch.nn.modules.loss._WeightedLoss)
        )
        and not cls == torch.nn.modules.loss._Loss
        and not cls == torch.nn.modules.loss._WeightedLoss
    ):
        LossFunction.register("torch::" + name)(cls)
