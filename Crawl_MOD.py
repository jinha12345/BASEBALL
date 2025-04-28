import requests
import pandas as pd
import numpy as np
import re
import pickle
import json

def KBO_Crawl(GAME_ID):
    data = {'MIME Type': 'application/x-www-form-urlencoded; charset=UTF-8', 
        'leId': 1,
        'srId': 0,
        'seasonId': GAME_ID[:4],
        'gameId': GAME_ID
       } 
    url_base_sc = 'https://www.koreabaseball.com/ws/Schedule.asmx/GetScoreBoardScroll'
    url_base_bs = 'https://www.koreabaseball.com/ws/Schedule.asmx/GetBoxScoreScroll'
    res_sc = requests.post(url_base_sc, data = data)
    res_bs = requests.post(url_base_bs, data = data)
    a = res_sc.json()
    GameInfo = {'GAME_ID' : a['G_ID'], 'GAME_DATE' : a['G_DT'], 'SEASON_ID' : a['SEASON_ID'], 'HOME_NAME' : a['HOME_NM'], 'HOME_ID' : a['HOME_ID'], 'AWAY_NAME' : a['AWAY_NM'], 'AWAY_ID' : a['AWAY_ID'], 'STADIUM' : a['S_NM'], 'CROWD' : a['CROWD_CN'], 'START_TIME' : a['START_TM'], 'END_TIME' : a['END_TM'], 'USE_TIME' : a['USE_TM'], 'FULL_HOME_NAME' : a['FULL_HOME_NM'], 'FULL_AWAY_NAME' : a['FULL_AWAY_NM'], 'INNING' : a['maxInning']}

    table1_sc = json.loads(res_sc.json()['table1'].replace('\r\n',''))
    table2_sc = json.loads(res_sc.json()['table2'].replace('\r\n',''))
    table3_sc = json.loads(res_sc.json()['table3'].replace('\r\n',''))

    table_ETC_bs = json.loads(res_bs.json()['tableEtc'].replace('\r\n',''))

    table_Hitter_bs = res_bs.json()['arrHitter']
    table_Pitcher_bs = res_bs.json()['arrPitcher']

    table1_1_Hitter_bs = json.loads(table_Hitter_bs[0]['table1'].replace('\r\n',''))
    table1_2_Hitter_bs = json.loads(table_Hitter_bs[0]['table2'].replace('\r\n',''))
    table1_3_Hitter_bs = json.loads(table_Hitter_bs[0]['table3'].replace('\r\n',''))
    table2_1_Hitter_bs = json.loads(table_Hitter_bs[1]['table1'].replace('\r\n',''))
    table2_2_Hitter_bs = json.loads(table_Hitter_bs[1]['table2'].replace('\r\n',''))
    table2_3_Hitter_bs = json.loads(table_Hitter_bs[1]['table3'].replace('\r\n',''))

    table1_Pitcher_bs = json.loads(table_Pitcher_bs[0]['table'].replace('\r\n',''))
    table2_Pitcher_bs = json.loads(table_Pitcher_bs[1]['table'].replace('\r\n',''))
    # Extract the rows from the data structure
    rows_data = table1_1_Hitter_bs['rows']

    # Create lists to store the extracted data
    numbers = []
    positions = []
    names = []

    # Extract data from each row
    for row_item in rows_data:
        row = row_item['row']
        if len(row) >= 3:
            numbers.append(row[0]['Text'])
            positions.append(row[1]['Text'])
            names.append(row[2]['Text'])

    # Create the DataFrame
    df = pd.DataFrame({
        '번호': numbers,
        '포지션': positions,
        '선수명': names
    })

    # For display formatting (optional)
    pd.set_option('display.unicode.east_asian_width', True)

    df_1 = df

    # Extract the data from the structure
    headers_data = table1_2_Hitter_bs['headers'][0]['row']
    rows_data = table1_2_Hitter_bs['rows']

    # Extract column names (inning numbers)
    columns = [header['Text'] for header in headers_data]

    # Create an empty DataFrame with the correct dimensions
    df = pd.DataFrame(index=range(len(rows_data)), columns=columns)

    # Fill the DataFrame with the at-bat results
    for i, row_item in enumerate(rows_data):
        row = row_item['row']
        for j, cell in enumerate(row):
            # Check if the cell has meaningful content (not &nbsp;)
            if cell['Text'] and cell['Text'] != '&nbsp;':
                df.iloc[i, j] = cell['Text']
            else:
                df.iloc[i, j] = None  # Set empty cells to None/null

    # Reset the index to start from 0
    df.reset_index(drop=True, inplace=True)

    # For display formatting (optional)
    pd.set_option('display.unicode.east_asian_width', True)

    df_2 = df

    # Create DataFrame from the rows data
    rows = table1_3_Hitter_bs['rows']
    data = []

    for row_dict in rows:
        row = row_dict['row']
        data.append([cell['Text'] for cell in row])

    # Use the Korean headers from the image
    headers = ['타수', '안타', '타점', '득점', '타율']

    # Create DataFrame
    df = pd.DataFrame(data, columns=headers)

    # Convert numeric columns to appropriate data types
    df['타수'] = pd.to_numeric(df['타수'])
    df['안타'] = pd.to_numeric(df['안타'])
    df['타점'] = pd.to_numeric(df['타점'])
    df['득점'] = pd.to_numeric(df['득점'])
    df['타율'] = pd.to_numeric(df['타율'])

    # Add total row from tfoot if needed
    if 'tfoot' in table1_3_Hitter_bs and table1_3_Hitter_bs['tfoot']:
        tfoot_row = table1_3_Hitter_bs['tfoot'][0]['row']
        totals = [cell['Text'] for cell in tfoot_row]
        # You can add this as a row with a special index or handle it separately
        # df.loc['Total'] = totals

    # Display the DataFrame
    df_3 = df

    Team1_Hitter = pd.concat([df_1,df_2,df_3],axis=1)

    # Extract the rows from the data structure
    rows_data = table2_1_Hitter_bs['rows']

    # Create lists to store the extracted data
    numbers = []
    positions = []
    names = []

    # Extract data from each row
    for row_item in rows_data:
        row = row_item['row']
        if len(row) >= 3:
            numbers.append(row[0]['Text'])
            positions.append(row[1]['Text'])
            names.append(row[2]['Text'])

    # Create the DataFrame
    df = pd.DataFrame({
        '번호': numbers,
        '포지션': positions,
        '선수명': names
    })

    # For display formatting (optional)
    pd.set_option('display.unicode.east_asian_width', True)

    df_1 = df

    # Extract the data from the structure
    headers_data = table2_2_Hitter_bs['headers'][0]['row']
    rows_data = table2_2_Hitter_bs['rows']

    # Extract column names (inning numbers)
    columns = [header['Text'] for header in headers_data]

    # Create an empty DataFrame with the correct dimensions
    df = pd.DataFrame(index=range(len(rows_data)), columns=columns)

    # Fill the DataFrame with the at-bat results
    for i, row_item in enumerate(rows_data):
        row = row_item['row']
        for j, cell in enumerate(row):
            # Check if the cell has meaningful content (not &nbsp;)
            if cell['Text'] and cell['Text'] != '&nbsp;':
                df.iloc[i, j] = cell['Text']
            else:
                df.iloc[i, j] = None  # Set empty cells to None/null

    # Reset the index to start from 0
    df.reset_index(drop=True, inplace=True)

    # For display formatting (optional)
    pd.set_option('display.unicode.east_asian_width', True)

    df_2 = df

    # Create DataFrame from the rows data
    rows = table2_3_Hitter_bs['rows']
    data = []

    for row_dict in rows:
        row = row_dict['row']
        data.append([cell['Text'] for cell in row])

    # Use the Korean headers from the image
    headers = ['타수', '안타', '타점', '득점', '타율']

    # Create DataFrame
    df = pd.DataFrame(data, columns=headers)

    # Convert numeric columns to appropriate data types
    df['타수'] = pd.to_numeric(df['타수'])
    df['안타'] = pd.to_numeric(df['안타'])
    df['타점'] = pd.to_numeric(df['타점'])
    df['득점'] = pd.to_numeric(df['득점'])
    df['타율'] = pd.to_numeric(df['타율'])

    # Add total row from tfoot if needed
    if 'tfoot' in table1_3_Hitter_bs and table1_3_Hitter_bs['tfoot']:
        tfoot_row = table1_3_Hitter_bs['tfoot'][0]['row']
        totals = [cell['Text'] for cell in tfoot_row]
        # You can add this as a row with a special index or handle it separately
        # df.loc['Total'] = totals

    # Display the DataFrame
    df_3 = df

    Team2_Hitter = pd.concat([df_1,df_2,df_3],axis=1)

    # Get the headers from the dictionary
    headers = []
    for cell in table1_Pitcher_bs['headers'][0]['row']:
        headers.append(cell['Text'])

    # Get the data from the rows
    data = []
    for row_dict in table1_Pitcher_bs['rows']:
        row = []
        for cell in row_dict['row']:
            # Clean up any HTML entities (like &nbsp;)
            cell_text = cell['Text']
            if cell_text == '&nbsp;':
                cell_text = ''
            row.append(cell_text)
        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data, columns=headers)

    # Convert columns to appropriate data types where needed
    numeric_columns = ['승', '패', '세', '이닝', '타자', '투구수', '타수', '피안타', '홈런', '4사구', '삼진', '실점', '자책']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert 평균자책점 column to float
    if '평균자책점' in df.columns:
        df['평균자책점'] = pd.to_numeric(df['평균자책점'], errors='coerce')

    # If you need to handle the footer row separately
    # (For creating totals or handling them specially)
    if 'tfoot' in table1_Pitcher_bs and table1_Pitcher_bs['tfoot']:
        footer_row = []
        for cell in table1_Pitcher_bs['tfoot'][0]['row']:
            footer_row.append(cell['Text'])
        # Add footer as a new row if needed
        # df.loc['TOTAL'] = footer_row

    # Print the DataFrame
    Team1_Pitcher = df

    # Get the headers from the dictionary
    headers = []
    for cell in table2_Pitcher_bs['headers'][0]['row']:
        headers.append(cell['Text'])

    # Get the data from the rows
    data = []
    for row_dict in table2_Pitcher_bs['rows']:
        row = []
        for cell in row_dict['row']:
            # Clean up any HTML entities (like &nbsp;)
            cell_text = cell['Text']
            if cell_text == '&nbsp;':
                cell_text = ''
            row.append(cell_text)
        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data, columns=headers)

    # Convert columns to appropriate data types where needed
    numeric_columns = ['승', '패', '세', '이닝', '타자', '투구수', '타수', '피안타', '홈런', '4사구', '삼진', '실점', '자책']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert 평균자책점 column to float
    if '평균자책점' in df.columns:
        df['평균자책점'] = pd.to_numeric(df['평균자책점'], errors='coerce')

    # If you need to handle the footer row separately
    # (For creating totals or handling them specially)
    if 'tfoot' in table2_Pitcher_bs and table2_Pitcher_bs['tfoot']:
        footer_row = []
        for cell in table2_Pitcher_bs['tfoot'][0]['row']:
            footer_row.append(cell['Text'])
        # Add footer as a new row if needed
        # df.loc['TOTAL'] = footer_row

    # Print the DataFrame
    Team2_Pitcher = df

    # Function to clean text (remove extra whitespace and newlines)
    def clean_text(text):
        if isinstance(text, str):
            return ' '.join(text.strip().replace('\r\n', ' ').split())
        return text

    # Extract data from the dictionary
    rows = []
    for row_dict in table_ETC_bs['rows']:
        category = row_dict['row'][0]['Text']
        value = clean_text(row_dict['row'][1]['Text'])
        rows.append([category, value])

    # Create DataFrame
    df = pd.DataFrame(rows, columns=['구분', '내용'])

    # Display the DataFrame
    ETC = df

    # 주어진 데이터
    data = table1_sc
    # 필요한 데이터 추출
    result_data = []

    for row in data['rows']:
        win_lose = row['row'][0]['Text']
        
        # 팀 코드 추출 (initial_XX.png에서 XX 부분)
        team_info = row['row'][1]['Text']
        team_code_match = re.search(r'initial_([A-Z]+)\.png', team_info)
        team_code = team_code_match.group(1) if team_code_match else None
        
        # 전적 추출 (승패무 정보)
        record_match = re.search(r'(\d+)승\s*(\d+)패\s*(\d+)무', team_info)
        win_count = record_match.group(1) if record_match else None
        lose_count = record_match.group(2) if record_match else None
        draw_count = record_match.group(3) if record_match else None
        record = f"{win_count}승 {lose_count}패 {draw_count}무" if win_count and lose_count and draw_count else None
        
        result_data.append({
            '승패여부': win_lose,
            '팀명': team_code,
            '전적': record
        })

    # 데이터프레임 생성
    df_1 = pd.DataFrame(result_data)

    # Your JSON data - assuming it's stored in a variable called table2_sc
    # Extract column headers
    columns = []
    for cell in table2_sc['headers'][0]['row']:
        columns.append(cell['Text'])

    # Extract row data
    data = []
    for row_data in table2_sc['rows']:
        row_values = []
        for cell in row_data['row']:
            row_values.append(cell['Text'])
        data.append(row_values)

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)

    # If you want to convert data types (since these are likely numeric except for the '-')
    # Replace '-' with NaN and convert applicable columns to numeric
    df = df.replace('-', pd.NA)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])

    df_2 = df

    # Extract data from table3_sc
    data = []
    for row_dict in table3_sc['rows']:
        row_data = []
        for cell in row_dict['row']:
            row_data.append(cell['Text'])
        data.append(row_data)

    # Create DataFrame with column names
    df = pd.DataFrame(data, columns=['R', 'H', 'E', 'B'])

    # Convert text values to numeric
    df = df.apply(pd.to_numeric)

    # Display the DataFrame
    df_3 = df

    Score = pd.concat([df_1,df_2,df_3],axis=1)

    All_Info = {'GameInfo' : GameInfo, 'Score' : Score, 'ETC' : ETC, 'Team1_Hitter' : Team1_Hitter, 'Team2_Hitter' : Team2_Hitter, 'Team1_Pitcher' : Team1_Pitcher, 'Team2_Pitcher' : Team2_Pitcher}

    with open('./DATA/' + GameInfo['GAME_ID'] + ".pickle","wb") as f:
        pickle.dump(All_Info, f)