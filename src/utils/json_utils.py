import json
import os
from pathlib import Path

def load_json(path):
    """
    JSON 파일을 읽어서 파이썬 객체로 반환

    Parameters
    ----------
    path : str or Path
        JSON 파일 경로

    Returns
    -------
    dict or list or None
        JSON 데이터. 파일이 없으면 None 반환
    """
    from pathlib import Path
    import os

    path = Path(path)

    if not os.path.exists(path):
        return None

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, path):
    """
    파이썬 객체를 JSON 파일로 저장

    Parameters
    ----------
    data : dict or list
        저장할 데이터
    path : str or Path
        저장할 파일 경로
    """
    from pathlib import Path

    path = Path(path)

    # 디렉토리가 없으면 생성
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)