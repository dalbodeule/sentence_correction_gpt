from typing import List, Dict

import json
import pandas as pd
import os
import re

replaced = r"^\"|\"$"

def read_json_file(file_path: str) -> pd.DataFrame:
    """
    JSON 파일을 읽어와 파싱한 데이터를 리스트로 반환하는 함수
    :param file_path: 읽어올 JSON 파일의 경로
    :return: JSON 파일을 파싱한 리스트
    """

    df = pd.DataFrame(columns=['id', 'text', 'corrected'])

    # JSON 파일 읽어오기
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        data: Dict = json.load(f)
        df.loc[len(df)] = {'id': file_path, 'text': data.get('ko'), 'corrected':  re.sub(replaced,'', str(data.get('corrected'))).strip()}
    
    return df

def get_json_files(folder_path):
    json_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files