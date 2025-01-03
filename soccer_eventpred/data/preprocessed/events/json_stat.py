import pandas as pd
data = "all_preprocessed.pkl"
df = pd.read_pickle(data)
df=df.reset_index(drop=True)
pd.set_option("display.max_columns", 150)
df = df.head(25)

# 현재 사용 중인 df 설정
df['changed'] = df['wyscout_team_id'] != df['wyscout_team_id'].shift(1)
df.loc[0, 'changed'] = False  # 첫 번째 행은 변화를 감지하지 않도록 설정
change_indices = df[df['changed']].index.tolist()
print(df)
print(change_indices)
new_rows = []
prev_index = 0
#
# # 변화 인덱스마다 반복
for index in change_indices:
    # 변화 전까지의 행들을 추가
    new_rows.append(df.iloc[prev_index:index])

    # 'change_poss' 행을 생성
    new_row = pd.DataFrame({
       'competition': [df.loc[prev_index, 'competition']],
       'wyscout_match_id': [df.loc[prev_index, 'wyscout_match_id']],
       'match_period': [None],
       'event_time_period': [None],
        'event_time': [None],
        'scaled_event_time': [None],
        'wyscout_team_id': [None],
        'team_name': [None],
        'comb_event_name': ['change_poss'],
        'start_pos_x': [None],
        'start_pos_y': [None],
        'end_pos_x': [None],
        'end_pos_y': [None],
        'wyscout_player_id': [None],
        'player_name': [None],
        'tags': [None]
    })

    # 새로운 'change_poss' 행 추가
    new_rows.append(new_row)

    # 인덱스 업데이트
    prev_index = index

# 마지막으로 남은 부분 추가
new_rows.append(df.iloc[prev_index:])

# 최종 데이터프레임 생성
df_final = pd.concat(new_rows, ignore_index=True)

# 결과 출력
print(df_final)


import pandas as pd
#
# import pandas as pd
#
# # Example DataFrame with event data, including both offensive and defensive events
# data = {
#     'wyscout_team_id': [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1],
#     'team_name': ['Team A', 'Team A', 'Team B', 'Team B', 'Team A', 'Team A', 'Team B', 'Team B', 'Team A', 'Team A',
#                   'Team B', 'Team B', 'Team A'],
#     'comb_event_name': [
#         'Pass', 'Duel_Air duel', 'Duel_Air duel', 'Shot', 'Duel_Ground defending duel',
#         'Pass', 'Save attempt_Save attempt', 'Goal', 'Interruption_Whistle', 'Pass',
#         'Foul_Foul', 'Save attempt_Reflexes', 'Pass'
#     ]
# }
#
# # Create the DataFrame
# df = pd.DataFrame(data)
#
# # DEFENSIVE_COMB_EVENTS 목록
# DEFENSIVE_COMB_EVENTS = [
#     "Duel_Air duel",
#     "Duel_Ground defending duel",
#     "Duel_Ground loose ball duel",
#     "Interruption_Ball out of the field",
#     "Interruption_Whistle",
#     "Foul_Foul",
#     "Foul_Hand foul",
#     "Foul_Late card foul",
#     "Foul_Out of game foul",
#     "Foul_Protest",
#     "Foul_Simulation",
#     "Foul_Time lost foul",
#     "Foul_Violent Foul",
#     "Offside_",
#     "Goalkeeper leaving line_Goalkeeper leaving line",
#     "Save attempt_Reflexes",
#     "Save attempt_Save attempt"
# ]
#
#
# def insert_CoP(df: pd.DataFrame) -> pd.DataFrame:
#     # 팀이 바뀌는 지점 계산
#     df['team_changed'] = df['wyscout_team_id'] != df['wyscout_team_id'].shift(1)
#
#     new_rows = []
#     prev_index = 0
#
#     for index, row in df.iterrows():
#         # 기존 행 추가
#         new_rows.append(row)
#
#         # 팀이 바뀌고 수비적 이벤트가 있을 때 CoP 삽입
#         if index > 0 and df.loc[index, 'team_changed']:
#             prev_event = df.loc[index - 1, 'comb_event_name']
#             if prev_event in DEFENSIVE_COMB_EVENTS:
#                 # 'CoP' 행 생성
#                 new_row = {
#                     'wyscout_team_id': None,
#                     'team_name': None,
#                     'comb_event_name': 'CoP'
#                 }
#                 new_rows.append(new_row)
#
#     # 새로운 데이터프레임으로 변환
#     df_final = pd.DataFrame(new_rows)
#     df_final = df_final.drop(columns='team_changed')
#
#     return df_final
#
#
# # 함수 실행
# df_final = insert_CoP(df)
#
# # 결과 확인
# print(df_final)
