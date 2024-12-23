import json
import os


# jsonl 파일 경로
base=os.getcwd()
file_path = os.path.join(os.path.dirname(__file__), "test.jsonl")

# 데이터셋 크기 측정을 위한 변수 초기화
total_matches = 0
total_events = 0
events_per_match = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        total_matches += 1
        match_data = json.loads(line.strip())
        
        # 각 매치에서 이벤트 수 추출
        events = match_data.get("events", [])
        num_events = len(events)
        total_events += num_events
        events_per_match.append(num_events)

# 결과 출력
print(f"총 매치 수: {total_matches}")
print(f"총 이벤트 수: {total_events}")
print(f"매치당 평균 이벤트 수: {total_events / total_matches:.2f}")
print(f"최대 이벤트 수: {max(events_per_match)}")
print(f"최소 이벤트 수: {min(events_per_match)}")
