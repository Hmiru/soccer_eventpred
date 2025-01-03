from typing import Dict, List, Optional, cast

import torch

from soccer_eventpred.data.dataclass import Batch, Instance
from soccer_eventpred.data.vocabulary import PAD_TOKEN, UNK_TOKEN, Vocabulary
from soccer_eventpred.modules.datamodule.soccer_datamodule import SoccerDataModule
from soccer_eventpred.modules.datamodule.soccer_dataset import SoccerEventDataset

@SoccerDataModule.register("wyscout")
class WyScoutDataModule(SoccerDataModule):
    def __init__(
        self,
        train_datasource,
        val_datasource=None,
        test_datasource=None,
        batch_size=32,
        num_workers=8,
        label2events: Optional[Dict[str, List[str]]] = None,
        vocab: Optional[Vocabulary] = None,
    ):

        super().__init__()
        self._train_dataset = SoccerEventDataset()
        self._val_dataset = SoccerEventDataset() if val_datasource else None
        self._test_dataset = SoccerEventDataset() if test_datasource else None
        self._train_datasource = train_datasource
        self._val_datasource = val_datasource
        self._test_datasource = test_datasource
        self.batch_size = batch_size
        self.num_workers = num_workers
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
        if not self.vocab.size("events"):
            self.build_vocab()

        self._prepare_data(self._train_dataset, self._train_datasource)

        if self._val_datasource is not None:
            self._prepare_data(self._val_dataset, self._val_datasource)

        if self._test_datasource is not None:
            self._prepare_data(self._test_dataset, self._test_datasource)

    def _prepare_data(self, dataset, data_source):
        for match in data_source.collect(): # 경기별 이벤트데이터를 인스턴스 형태로 생성
            instance = self._prepare_instance(match) # 이벤트데이터의 인스터스 생성. len(instance)=학습에 활용되는 모든 이벤트 개수

            if isinstance(instance, dict):
                for key in instance.keys():
                    dataset.add(instance[key])
            else:
                dataset.add(instance)

    def build_vocab(self, matches=None):
        if (
            self.vocab.size("teams")
            and self.vocab.size("events")
            and self.vocab.size("players")
        ):
            return
        self.vocab.add(UNK_TOKEN, "teams") # UNK_TOKEN: 0
        self.vocab.add(PAD_TOKEN, "teams") # PAD_TOKEN: 1
        self.vocab.add(UNK_TOKEN, "events")
        self.vocab.add(PAD_TOKEN, "events")
        self.vocab.add(UNK_TOKEN, "players")
        self.vocab.add(PAD_TOKEN, "players")

        if matches is None:
            matches = self._train_datasource.collect()
        for match in matches:
            for event in match.events:
                self.vocab.add(event.team_name, "teams")
                self.vocab.add(event.player_name, "players")
                if self._event2label is not None:
                    event_id = self.vocab.add(
                        self._event2label[event.comb_event_name], "events"
                    )
                    self.event_counts[event_id] = self.event_counts.get(event_id, 0) + 1
                else:
                    event_id = self.vocab.add(event.comb_event_name, "events")
                    self.event_counts[event_id] = self.event_counts.get(event_id, 0) + 1

        
        self.event_counts[self.vocab.get(UNK_TOKEN, "events")] = 0
        self.event_counts[self.vocab.get(PAD_TOKEN, "events")] = 0
        self.event_counts = [
            elem[1] for elem in sorted(self.event_counts.items(), key=lambda x: x[0])
        ]

    def _prepare_instance(self, match):
        offensive_sequence = True
        if offensive_sequence:
            team_names = set([event.team_name for event in match.events if event.team_name != '[UNK]']) # Cop토큰의 Team Name은 UNK Token
            
            if len(team_names) <= 1:
                raise ValueError(f"경기 내 이벤트 데이터셋의 팀 정보가 오직 하나이다: {team_names}")
            
            instance_by_team = {}
            for team_name in team_names:
                event_times = [event.scaled_event_time for event in match.events 
                               if event.team_name in ['[UNK]', team_name]]
                team_ids = [self.vocab.get(event.team_name, "teams") for event in match.events 
                            if event.team_name in ['[UNK]', team_name]]

                start_pos_x = [event.start_pos_x for event in match.events 
                               if event.team_name in ['[UNK]', team_name]]
                start_pos_y = [event.start_pos_y for event in match.events 
                               if event.team_name in ['[UNK]', team_name]]
                end_pos_x = [event.end_pos_x for event in match.events 
                             if event.team_name in ['[UNK]', team_name]]
                end_pos_y = [event.end_pos_y for event in match.events 
                             if event.team_name in ['[UNK]', team_name]]

                if self._event2label is not None:
                    event_ids = [self.vocab.get(self._event2label[event.comb_event_name], "events") for event in match.events 
                                 if event.team_name in ['[UNK]', team_name]]
                else:
                    event_ids = [self.vocab.get(event.comb_event_name, "events")for event in match.events 
                                 if event.team_name in ['[UNK]', team_name]]

                player_ids = [self.vocab.get(event.player_name, "players") for event in match.events 
                              if event.team_name in ['[UNK]', team_name]]

                instance_by_team[team_name] = Instance(event_times, team_ids, event_ids, player_ids, 
                                                       start_pos_x, start_pos_y, end_pos_x, end_pos_y)

            return instance_by_team
        else:
            event_times = [event.scaled_event_time for event in match.events]
            team_ids = [self.vocab.get(event.team_name, "teams") for event in match.events]
            start_pos_x = [event.start_pos_x for event in match.events]
            start_pos_y = [event.start_pos_y for event in match.events]
            end_pos_x = [event.end_pos_x for event in match.events]
            end_pos_y = [event.end_pos_y for event in match.events]
            if self._event2label is not None:
                event_ids = [self.vocab.get(self._event2label[event.comb_event_name], "events")for event in match.events]
            else:
                event_ids = [self.vocab.get(event.comb_event_name, "events")for event in match.events]

            player_ids = [self.vocab.get(event.player_name, "players") for event in match.events]

            return Instance(
                event_times,
                team_ids,
                event_ids,
                player_ids,
                start_pos_x,
                start_pos_y,
                end_pos_x,
                end_pos_y,
            )

    def setup(self, stage: str) -> None:
        ...

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        print("train DataSet: ", len(self._train_dataset))
        return self.build_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        print("valid DataSet: ", len(self._train_dataset))
        return self.build_dataloader(self._val_dataset)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        print("test DataSet: ", len(self._train_dataset))
        return self.build_dataloader(self._test_dataset)

    def build_dataloader(
        self, dataset, batch_size=None, shuffle=False, num_workers=0
    ) -> torch.utils.data.DataLoader:

        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=shuffle,
            collate_fn=self.batch_collator,
            num_workers=self.num_workers,
        )

    def transfer_batch_to_device(
        self, batch: Batch, device, dataloader_idx: int
    ) -> Batch:
        return Batch(
            event_times=batch.event_times.to(device),
            team_ids=batch.team_ids.to(device),
            event_ids=batch.event_ids.to(device),
            player_ids=batch.player_ids.to(device),
            start_pos_x=batch.start_pos_x.to(device),
            start_pos_y=batch.start_pos_y.to(device),
            end_pos_x=batch.end_pos_x.to(device),
            end_pos_y=batch.end_pos_y.to(device),
            mask=batch.mask.to(device),
            labels=batch.labels.to(device),
        )

    def batch_collator(self, instances: List[Instance]) -> Batch:
        '''
        입력으로 받은 instances에서 각 instance에 대해, step size가 1인 슬라이딩 윈도우 기법을 사용하여 최대 40개의 이벤트 시퀀스를 추출합니다. 
        각 `instance` 내에서 생성되는 윈도우의 개수는 window_size = len(instance) - 40 + 1 입니다.

        :param instances: List which is a batch of instances
        :return: Batch object which is a batch of tensors
        '''

        MAX_LENGTH = 39
        DEFAULT_EVENT_TIME=120
        DEFAULT_POSITION=101
        EXCEPT_EVENT_IDS = {0, 3, 7} # PAD TOKEN = ignore_token이므로 loss function시 제거됨

        windows_per_instance=[len(instance.event_ids)-MAX_LENGTH for instance in instances]
        total_windows=sum(windows_per_instance)

        '''
        make empty tensors of size (total_windows, max_length) for each attribute
        '''
        event_times = torch.full((total_windows, MAX_LENGTH), DEFAULT_EVENT_TIME, dtype=torch.long)
        team_ids = torch.full((total_windows, MAX_LENGTH), self.vocab.get(PAD_TOKEN, "teams"),dtype=torch.long)
        event_ids = torch.full((total_windows, MAX_LENGTH), self.vocab.get(PAD_TOKEN, "events"),dtype=torch.long)
        player_ids = torch.full((total_windows, MAX_LENGTH), self.vocab.get(PAD_TOKEN, "players"), dtype=torch.long)

        start_pos_x = torch.full((total_windows, MAX_LENGTH), DEFAULT_POSITION, dtype=torch.long)
        start_pos_y = torch.full((total_windows, MAX_LENGTH), DEFAULT_POSITION, dtype=torch.long)
        end_pos_x = torch.full((total_windows, MAX_LENGTH), DEFAULT_POSITION, dtype=torch.long)
        end_pos_y = torch.full((total_windows, MAX_LENGTH), DEFAULT_POSITION, dtype=torch.long)

        # mask: loss function에 계산될 데이터셋 여부 -> loss *= mask
        # if mask = False, not calculate loss function
        mask = torch.zeros((total_windows, MAX_LENGTH), dtype=torch.bool) 
        labels = torch.full((total_windows,), self.vocab.get(PAD_TOKEN, "events"),dtype=torch.long)

        window_idx=0
        # instances: batch dataset
        for _, instance in enumerate(instances):
            seq_length=len(instance.event_ids) # length of the sequence
            
            # 슬라이딩 윈도우
            for start_idx in range(0, seq_length-MAX_LENGTH):
                end_idx = start_idx + MAX_LENGTH # input: start_idx~end_idx = 40 = input(39) + output(1)

                # len(instance.event_times[start_idx:end_idx]) = 40 -> 리스트(List)이므로 end_idx는 포함되지 않음
                event_times[window_idx, : ] = torch.tensor(instance.event_times[start_idx:end_idx], dtype=torch.long)
                team_ids[window_idx, : ] = torch.tensor(instance.team_ids[start_idx:end_idx], dtype=torch.long)
                event_ids[window_idx, : ] = torch.tensor(instance.event_ids[start_idx:end_idx], dtype=torch.long)
                player_ids[window_idx, : ] = torch.tensor(instance.player_ids[start_idx:end_idx], dtype=torch.long)
                start_pos_x[window_idx, : ] = torch.tensor(instance.start_pos_x[start_idx:end_idx], dtype=torch.long)
                start_pos_y[window_idx, : ] = torch.tensor(instance.start_pos_y[start_idx:end_idx], dtype=torch.long)
                end_pos_x[window_idx, : ] = torch.tensor(instance.end_pos_x[start_idx:end_idx], dtype=torch.long)
                end_pos_y[window_idx, : ] = torch.tensor(instance.end_pos_y[start_idx:end_idx], dtype=torch.long)
                mask[window_idx, : ] = True # 입력의 길이를 표현하는 변수(pack_padded_sequence) 

                labels[window_idx] = instance.event_ids[end_idx]

                if labels[window_idx] in EXCEPT_EVENT_IDS:
                    labels[window_idx] = self.vocab.get(PAD_TOKEN, "events")

                window_idx+=1
            
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
            labels=labels
        )