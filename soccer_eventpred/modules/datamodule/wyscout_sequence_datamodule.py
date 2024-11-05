from typing import Dict, List, Optional, cast

import torch

from soccer_eventpred.data.dataclass import Batch, Instance
from soccer_eventpred.data.vocabulary import PAD_TOKEN, UNK_TOKEN, Vocabulary
from soccer_eventpred.modules.datamodule.soccer_datamodule import SoccerDataModule
from soccer_eventpred.modules.datamodule.soccer_dataset import SoccerEventDataset


@SoccerDataModule.register("wyscout_sequence")
class WyScoutSequenceDataModule(SoccerDataModule):
    def __init__(
        self,
        train_datasource,
        val_datasource=None,
        test_datasource=None,

        sequence_length=39,
        ignore_tokens: Optional[List[str]] = None,
        label2events: Optional[Dict[str, List[str]]] = None,
        vocab: Optional[Vocabulary] = None,

        batch_size=32,
        num_workers=0,
    ):
        super().__init__()
        self._train_dataset = SoccerEventDataset()
        self._val_dataset = SoccerEventDataset() if val_datasource else None
        self._test_dataset = SoccerEventDataset() if test_datasource else None
        self._train_datasource = train_datasource
        self._val_datasource = val_datasource
        self._test_datasource = test_datasource

        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.ignore_tokens = ignore_tokens
        self.vocab = vocab or Vocabulary()

        self._label2events = label2events
        if self._label2events is not None:
            self._event2label = {}
            for label, events in self._label2events.items():
                for event in events:
                    self._event2label[event] = label
        else:
            self._event2label = None
        self.event_counts = {}

    def prepare_data(self):
        '''
        각 데이터 소스를 기반으로 데이터셋을 준비합니다. LightningDataModule에서 사용하는 표준 함수.
        '''
        if not self.vocab.size("events"):
            self.build_vocab()
        
        self._prepare_data(self._train_dataset, self._train_datasource)
        print(f"train_dataset = {len(self._train_dataset)}")

        if self._val_datasource is not None:
            self._prepare_data(self._val_dataset, self._val_datasource)
            print(f"val_dataset = {len(self._val_dataset)}")

        if self._test_datasource is not None:
            self._prepare_data(self._test_dataset, self._test_datasource)
            print(f"test_dataset = {len(self._test_dataset)}")

    def _prepare_data(self, dataset, data_source):
        '''
        Function to create data instances using raw data(data_source).
        
        - dataset: List where instances for train/dev/test are stored.
        - len(dataset): Total number of instances created.
        - len(dataset[0]): Number of (event) data points within a single instance.
        '''
        # data_source: WyScoutDataSource
        # data_source.collect(): self._data 내에 있는 각 요소(경기 별)를 순차적으로 반환하는 제너레이터
        for match in data_source.collect():
            # match: 특정 경기 정보
            instance = self._prepare_instance(match)
            dataset.add(instance) # extend(list([Instance_1, Instance_2,....,Instance_N]))
       
    def build_vocab(self, matches=None):
        if (
            self.vocab.size("teams")
            and self.vocab.size("events")
            and self.vocab.size("players")
        ):
            return
        
        # 각 vocab에 UNK_TOKEN과 PAD_TOKEN을 기본으로 추가
        for category in ["teams", "events", "players"]:
            self.vocab.add(UNK_TOKEN, category)
            self.vocab.add(PAD_TOKEN, category)

        if matches is None:
            matches = self._train_datasource.collect()

        for match in matches:
            for event in match.events:
                self.vocab.add(event.team_name, "teams")
                self.vocab.add(event.player_name, "players")
                if self._event2label is not None:
                    event_id = self.vocab.add(self._event2label[event.comb_event_name], "events")
                    self.event_counts[event_id] = self.event_counts.get(event_id, 0) + 1
                else:
                    event_id = self.vocab.add(event.comb_event_name, "events")
                    self.event_counts[event_id] = self.event_counts.get(event_id, 0) + 1

        # ignore_tokens에 대해 event_counts를 0으로 설정하여 weighted loss 계산 시 제외
        for token in self.ignore_tokens:
            self.event_counts[self.vocab.get(token, "events")] = 0

        # event_counts: 학습 시 weighted loss 부여를 위함
        self.event_counts = [
            elem[1] for elem in sorted(self.event_counts.items(), key=lambda x: x[0])
        ]

    def _prepare_instance(self, match):
        '''
        instance: Represents input and output for each sequence segment: input: 39 events (t-39 to t-1) & output: 1 event (t-th event)
        len(instances): Total number of segments created per match, each of (self.sequence_length + 1) events.
        len(instances[0]): Number of events in the 0th instance, which is (self.sequence_length + 1).
        '''

        instances = []
        # Represents the data grouped in units of (self.sequence_length + 1) events per sliding window
        # len(match.events): 특정 경기의 이벤트 총 수
        for start_idx in range(0, len(match.events)-self.sequence_length+1):
            end_idx = min(start_idx + self.sequence_length, len(match.events))

            # 마지막 이벤트가 ignore_index에 있는 경우, 현재 슬라이딩 윈도우를 건너뜀
            label_token = self.vocab.get(match.events[end_idx - 1].comb_event_name, "events")
            if label_token in self.ignore_tokens:
                continue

            # Generate data in (self.sequence_length+1) units: 
            # input (39 events: t-39 to t-1) + output (1 event: t-th event)
            event_times = [event.scaled_event_time for event in match.events[start_idx:end_idx]] # len(event_times) = self.sequence_length+1
            team_ids = [self.vocab.get(event.team_name, "teams") for event in match.events[start_idx:end_idx]]
            start_pos_x = [event.start_pos_x for event in match.events[start_idx:end_idx]]
            start_pos_y = [event.start_pos_y for event in match.events[start_idx:end_idx]]
            end_pos_x = [event.end_pos_x for event in match.events[start_idx:end_idx]]
            end_pos_y = [event.end_pos_y for event in match.events[start_idx:end_idx]]
            
            if self._event2label is not None:
                event_ids = [
                    self.vocab.get(self._event2label[event.comb_event_name], "events")
                    for event in match.events[start_idx:end_idx]
                ]
            else:
                event_ids = [
                    self.vocab.get(event.comb_event_name, "events")
                    for event in match.events[start_idx:end_idx]
                ]

            player_ids = [
                self.vocab.get(event.player_name, "players") for event in match.events[start_idx:end_idx]
            ]

            instances.append(Instance(
                event_times,
                team_ids,
                event_ids,
                player_ids,
                start_pos_x,
                start_pos_y,
                end_pos_x,
                end_pos_y,
            ))
        
        return instances

    def setup(self, stage: str) -> None:
        ...

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return self.build_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self.build_dataloader(self._val_dataset, shuffle=False)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return self.build_dataloader(self._test_dataset, shuffle=False)

    def build_dataloader(
        self, dataset, batch_size=32, shuffle=False, num_workers=0
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            collate_fn=self.batch_collator,
            num_workers=self.num_workers,
        )
    
    def batch_collator(self, instances: List[Instance]) -> Batch:
        '''
        :param instances: List which is a batch of instances: collate_fn을 사용하여 같은 배치 안에 길이가 가장 긴 input에 맞춰 다른 input들에 PAD_TOKEN을 부여한다
        :return: Batch object which is a batch of tensors
        '''

        '''
        make empty tensors of size (total_windows, max_length) for each attribute
        The "+1" in self.sequence_length + 1 accounts for the inclusion of both features and labels
        '''
        max_length = max(len(instance.event_ids) for instance in instances)

        event_times = cast(
            torch.LongTensor,
            torch.full((len(instances), max_length), 120, dtype=torch.long),
        )
        team_ids = cast(
            torch.LongTensor,
            torch.full(
                (len(instances), max_length),
                self.vocab.get(PAD_TOKEN, "teams"),
                dtype=torch.long,
            ),
        )
        event_ids = cast(
            torch.LongTensor,
            torch.full(
                (len(instances), max_length),
                self.vocab.get(PAD_TOKEN, "events"),
                dtype=torch.long,
            ),
        )
        player_ids = cast(
            torch.LongTensor,
            torch.full(
                (len(instances), max_length),
                self.vocab.get(PAD_TOKEN, "players"),
                dtype=torch.long,
            ),
        )
        start_pos_x = cast(
            torch.LongTensor,
            torch.full((len(instances), max_length), 101, dtype=torch.long),
        )
        start_pos_y = cast(
            torch.LongTensor,
            torch.full((len(instances), max_length), 101, dtype=torch.long),
        )
        end_pos_x = cast(
            torch.LongTensor,
            torch.full((len(instances), max_length), 101, dtype=torch.long),
        )
        end_pos_y = cast(
            torch.LongTensor,
            torch.full((len(instances), max_length), 101, dtype=torch.long),
        )
        mask = cast(
            torch.BoolTensor,
            torch.zeros((len(instances), max_length), dtype=torch.bool),
        )

        for i, instance in enumerate(instances):
            event_times[i, : len(instance.event_times)] = torch.tensor(
                instance.event_times, dtype=torch.long
            )
            team_ids[i, : len(instance.team_ids)] = torch.tensor(
                instance.team_ids, dtype=torch.long
            )
            event_ids[i, : len(instance.event_ids)] = torch.tensor(
                instance.event_ids, dtype=torch.long
            )
            player_ids[i, : len(instance.player_ids)] = torch.tensor(
                instance.player_ids, dtype=torch.long
            )
            start_pos_x[i, : len(instance.start_pos_x)] = torch.tensor(
                instance.start_pos_x, dtype=torch.long
            )
            start_pos_y[i, : len(instance.start_pos_y)] = torch.tensor(
                instance.start_pos_y, dtype=torch.long
            )
            end_pos_x[i, : len(instance.end_pos_x)] = torch.tensor(
                instance.end_pos_x, dtype=torch.long
            )
            end_pos_y[i, : len(instance.end_pos_y)] = torch.tensor(
                instance.end_pos_y, dtype=torch.long
            )
            mask[i, : len(instance.event_ids)] = True 

        return Batch(
            event_times=event_times,
            team_ids=team_ids,
            event_ids=event_ids,
            player_ids=player_ids,
            start_pos_x=start_pos_x,
            start_pos_y=start_pos_y,
            end_pos_x=end_pos_x,
            end_pos_y=end_pos_y,
            mask=mask,
        )