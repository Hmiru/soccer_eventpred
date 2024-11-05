from typing import Any, Iterator, List

from soccer_eventpred.data.dataclass import WyScoutEvent, WyScoutEventSequence
from soccer_eventpred.data.sources.source import SoccerDataSource
from soccer_eventpred.env import DATA_DIR
from soccer_eventpred.util import load_jsonlines

import pandas as pd
@SoccerDataSource.register("wyscout_offense_only")
class WyScoutDataSource(SoccerDataSource):
    def __init__(self, data_name: str = "wyscout_offense_only", subset: str = "train.jsonl") -> None:
        self._datasource = load_jsonlines(
            DATA_DIR / data_name / "preprocessed/events" / subset
        )
        self._data: List[WyScoutEventSequence] = []
        self._build_data()

    def _build_data(self):
        # _datasource: train/dev/test에 저장된 preprecess data
        # data: 경기정보
        # len(self._data): 총 경기수
        # self._data = [{competition:EPL,...}, {competition: LaLiga..}, ..., {competition: K-league}]
        for data in self._datasource:
            events = [WyScoutEvent(**event) for event in data.pop("events")]
            self._data.append(
                WyScoutEventSequence(
                    events=events,
                    **data,
                )
            )

    def collect(self) -> Iterator[Any]:
        yield from self._data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> WyScoutEventSequence:
        return self._data[idx]
