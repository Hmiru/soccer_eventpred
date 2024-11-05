from soccer_eventpred.util import load_jsonlines
import pandas as pd
import json
df=pd.read_json("train.jsonl", lines=True)
# max columns to display
# pd.set_option('display.max_columns', None)
event_data=[
    {"comb_event_name": event['comb_event_name'], "team_name": event['team_name']}
        for event in df.loc[0, 'events']
]
events_df=pd.DataFrame(event_data)
#max rows to display
pd.set_option('display.max_rows', None)
print(events_df)