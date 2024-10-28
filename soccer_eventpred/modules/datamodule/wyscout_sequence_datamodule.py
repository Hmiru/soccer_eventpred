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
        batch_size=32,
        num_workers=0,
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
        for match in data_source.collect():
            instance = self._prepare_instance(match)
            dataset.add(instance)

    # def build_teams_vocab(self, matches=None) -> None:
    #     self.vocab.add(UNK_TOKEN, "teams")
    #     self.vocab.add(PAD_TOKEN, "teams")
    #     if matches is None:
    #         matches = self._train_datasource.collect()
    #     for match in matches:
    #         for event in match.events:
    #             self.vocab.add(event.team_name, "teams")

    # def build_events_vocab(self, matches=None) -> None:
    #     self.vocab.add(UNK_TOKEN, "events")
    #     self.vocab.add(PAD_TOKEN, "events")
    #     if matches is None:
    #         matches = self._train_datasource.collect()
    #     for match in matches:
    #         for event in match.events:
    #             if self._event2label is not None:
    #                 self.vocab.add(self._event2label[event.comb_event_name], "events")
    #             else:
    #                 self.vocab.add(event.comb_event_name, "events")

    # def build_players_vocab(self, matches=None) -> None:
    #     self.vocab.add(UNK_TOKEN, "players")
    #     self.vocab.add(PAD_TOKEN, "players")
    #     if matches is None:
    #         matches = self._train_datasource.collect()
    #     for match in matches:
    #         for event in match.events:
    #             self.vocab.add(event.player_name, "players")

    def build_vocab(self, matches=None):
        if (
            self.vocab.size("teams")
            and self.vocab.size("events")
            and self.vocab.size("players")
        ):
            return
        self.vocab.add(UNK_TOKEN, "teams")
        self.vocab.add(PAD_TOKEN, "teams")
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
        event_times = [event.scaled_event_time for event in match.events]
        team_ids = [self.vocab.get(event.team_name, "teams") for event in match.events]
        start_pos_x = [event.start_pos_x for event in match.events]
        start_pos_y = [event.start_pos_y for event in match.events]
        end_pos_x = [event.end_pos_x for event in match.events]
        end_pos_y = [event.end_pos_y for event in match.events]
        if self._event2label is not None:
            event_ids = [
                self.vocab.get(self._event2label[event.comb_event_name], "events")
                for event in match.events
            ]
        else:
            event_ids = [
                self.vocab.get(event.comb_event_name, "events")
                for event in match.events
            ]

        player_ids = [
            self.vocab.get(event.player_name, "players") for event in match.events
        ]
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
        return self.build_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return self.build_dataloader(self._val_dataset)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
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
        )

    def batch_collator(self, instances: List[Instance]) -> Batch:
        '''

        :param instances: List which is a batch of instances
        :return: Batch object which is a batch of tensors
        '''


        max_length=40
        windows_per_instance=[]
        for instance in instances:
            num_windows=max(1, len(instance.event_ids)-max_length+1)
            windows_per_instance.append(num_windows) # number of windows for each instance
        total_windows=sum(windows_per_instance)
        '''
        make empty tensors of size (total_windows, max_length) for each attribute
        '''
        event_times = cast(
            torch.LongTensor,
            torch.full(
                (total_windows, max_length),
                120, dtype=torch.long),
        )
        team_ids = cast(
            torch.LongTensor,
            torch.full(
                (total_windows, max_length),
                self.vocab.get(PAD_TOKEN, "teams"),
                dtype=torch.long,
            ),
        )
        event_ids = cast(
            torch.LongTensor,
            torch.full(
                (total_windows, max_length),
                self.vocab.get(PAD_TOKEN, "events"),
                dtype=torch.long,
            ),
        )
        player_ids = cast(
            torch.LongTensor,
            torch.full(
                (total_windows, max_length),
                self.vocab.get(PAD_TOKEN, "players"),
                dtype=torch.long,
            ),
        )
        start_pos_x = cast(
            torch.LongTensor,
            torch.full((total_windows, max_length),
                       101, dtype=torch.long),
        )
        start_pos_y = cast(
            torch.LongTensor,
            torch.full((total_windows, max_length),
                       101, dtype=torch.long),
        )
        end_pos_x = cast(
            torch.LongTensor,
            torch.full((total_windows, max_length),
                       101, dtype=torch.long),
        )
        end_pos_y = cast(
            torch.LongTensor,
            torch.full((total_windows, max_length),
                       101, dtype=torch.long),
        )
        mask = cast(
            torch.BoolTensor,
            torch.zeros((total_windows, max_length),
            dtype=torch.bool),
        )

        window_idx=0
        for instance in instances:
            sequence_length=len(instance.event_ids) # length of the sequence
            for start_idx in range(0, sequence_length-max_length+1):
                end_idx = start_idx + max_length
                '''
                fill the tensors with the values of the current window
                '''
                event_times[window_idx, : max_length] = torch.tensor(instance.event_times[start_idx:end_idx], dtype=torch.long)
                team_ids[window_idx, : max_length] = torch.tensor(instance.team_ids[start_idx:end_idx], dtype=torch.long)
                event_ids[window_idx, : max_length] = torch.tensor(instance.event_ids[start_idx:end_idx], dtype=torch.long)
                player_ids[window_idx, : max_length] = torch.tensor(instance.player_ids[start_idx:end_idx], dtype=torch.long)
                start_pos_x[window_idx, : max_length] = torch.tensor(instance.start_pos_x[start_idx:end_idx], dtype=torch.long)
                start_pos_y[window_idx, : max_length] = torch.tensor(instance.start_pos_y[start_idx:end_idx], dtype=torch.long)
                end_pos_x[window_idx, : max_length] = torch.tensor(instance.end_pos_x[start_idx:end_idx], dtype=torch.long)
                end_pos_y[window_idx, : max_length] = torch.tensor(instance.end_pos_y[start_idx:end_idx], dtype=torch.long)
                mask[window_idx, : max_length] = True
                window_idx+=1 # increment window index

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







        # for i, instance in enumerate(instances):
        # # slice the last 40 events
        #     last_events = instance.event_ids[-40:] if len(instance.event_ids) >= 40 else instance.event_ids
        #
        #     if len(last_events) == 0:
        #         print(f"Warning: Instance {i} has zero-length sequence!")
        #
        #     event_times[i, : len(last_events)] = torch.tensor(instance.event_times[-len(last_events):], dtype=torch.long)
        #     team_ids[i, : len(last_events)] = torch.tensor(instance.team_ids[-len(last_events):], dtype=torch.long)
        #     event_ids[i, : len(last_events)] = torch.tensor(last_events, dtype=torch.long)
        #     player_ids[i, : len(last_events)] = torch.tensor(instance.player_ids[-len(last_events):], dtype=torch.long)
        #     start_pos_x[i, : len(last_events)] = torch.tensor(instance.start_pos_x[-len(last_events):], dtype=torch.long)
        #     start_pos_y[i, : len(last_events)] = torch.tensor(instance.start_pos_y[-len(last_events):], dtype=torch.long)
        #     end_pos_x[i, : len(last_events)] = torch.tensor(instance.end_pos_x[-len(last_events):], dtype=torch.long)
        #     end_pos_y[i, : len(last_events)] = torch.tensor(instance.end_pos_y[-len(last_events):], dtype=torch.long)
        #     mask[i, : len(last_events)] = True
        #
        #     # event_times[i, : len(instance.event_times)] = torch.tensor(
        #     #     instance.event_times, dtype=torch.long
        #     # )
        #     # team_ids[i, : len(instance.team_ids)] = torch.tensor(
        #     #     instance.team_ids, dtype=torch.long
        #     # )
        #     # event_ids[i, : len(instance.event_ids)] = torch.tensor(
        #     #     instance.event_ids, dtype=torch.long
        #     # )
        #     # player_ids[i, : len(instance.player_ids)] = torch.tensor(
        #     #     instance.player_ids, dtype=torch.long
        #     # )
        #     # start_pos_x[i, : len(instance.start_pos_x)] = torch.tensor(
        #     #     instance.start_pos_x, dtype=torch.long
        #     # )
        #     # start_pos_y[i, : len(instance.start_pos_y)] = torch.tensor(
        #     #     instance.start_pos_y, dtype=torch.long
        #     # )
        #     # end_pos_x[i, : len(instance.end_pos_x)] = torch.tensor(
        #     #     instance.end_pos_x, dtype=torch.long
        #     # )
        #     # end_pos_y[i, : len(instance.end_pos_y)] = torch.tensor(
        #     #     instance.end_pos_y, dtype=torch.long
        #     # )
        #     # mask[i, : len(instance.event_ids)] = True
        #
        # return Batch(
        #     event_times=event_times,
        #     team_ids=team_ids,
        #     event_ids=event_ids,
        #     player_ids=player_ids,
        #     start_pos_x=start_pos_x,
        #     start_pos_y=start_pos_y,
        #     end_pos_x=end_pos_x,
        #     end_pos_y=end_pos_y,
        #     mask=mask,
        # )
