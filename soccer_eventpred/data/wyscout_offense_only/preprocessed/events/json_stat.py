from soccer_eventpred.util import load_jsonlines

train_data=load_jsonlines("train.jsonl")
print(len(train_data))