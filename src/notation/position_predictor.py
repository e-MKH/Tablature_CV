"""
기타 포지션 예측 모듈

MIDI 음정과 기타 포지션(현, 프렛) 간의 변환 및 예측을 담당합니다.
비디오 분석으로 포지션을 찾지 못한 경우, 이전 포지션과 물리적 거리를
고려하여 최적의 포지션을 예측합니다.
"""

from typing import Dict, List, Optional
from src.config.constants import (
    STRING_OFFSETS,
    MIN_FRET,
    MAX_FRET,
    STRING_MOVE_WEIGHT
)


def get_video_midi(string_name: str, fret_num: int) -> int:
    """
    기타 포지션(현, 프렛)을 MIDI 번호로 변환

    Parameters
    ----------
    string_name : str
        현 이름 (예: 'E', 'A', 'D', 'G')
    fret_num : int
        프렛 번호 (0 = 개방현, 1 = 1프렛, ...)

    Returns
    -------
    midi_note : int
        MIDI 번호 (0~127). 잘못된 현 이름이면 -1 반환

    Examples
    --------
    >>> get_video_midi('E', 0)  # E 개방현
    28
    >>> get_video_midi('A', 5)  # A 5프렛
    38
    >>> get_video_midi('X', 0)  # 잘못된 현
    -1

    Notes
    -----
    베이스 기타 개방현 MIDI 번호:
    - E (가장 낮은 현): 28
    - A: 33
    - D: 38
    - G (가장 높은 현): 43
    """
    if string_name not in STRING_OFFSETS:
        return -1

    return STRING_OFFSETS[string_name] + fret_num


def get_possible_positions(midi_note: int) -> List[Dict[str, any]]:
    """
    MIDI 번호에 대응하는 모든 가능한 기타 포지션 반환

    하나의 음(MIDI 번호)은 기타에서 여러 위치로 연주 가능합니다.
    예: MIDI 38은 E-10프렛, A-5프렛, D-0프렛(개방현) 모두 가능

    Parameters
    ----------
    midi_note : int
        찾고자 하는 MIDI 번호

    Returns
    -------
    possibilities : List[Dict[str, any]]
        가능한 포지션 리스트. 각 항목은 {'string': str, 'fret': int} 형태
        불가능한 MIDI 번호면 빈 리스트 반환

    Examples
    --------
    >>> get_possible_positions(38)
    [
        {'string': 'E', 'fret': 10},
        {'string': 'A', 'fret': 5},
        {'string': 'D', 'fret': 0}
    ]
    >>> get_possible_positions(28)  # E 개방현
    [{'string': 'E', 'fret': 0}]

    Notes
    -----
    - 프렛 범위: 0 ~ 22 (MAX_FRET)
    - 모든 현에 대해 계산 후 유효한 것만 반환
    """
    possibilities = []

    for string_name, string_midi in STRING_OFFSETS.items():
        fret = midi_note - string_midi

        # 유효한 프렛 범위 체크
        if MIN_FRET <= fret <= MAX_FRET:
            possibilities.append({
                'string': string_name,
                'fret': fret
            })

    return possibilities


def predict_best_position(
        target_midi: int,
        last_pos: Optional[Dict[str, any]] = None
) -> Dict[str, any]:
    """
    이전 포지션을 고려하여 최적의 기타 포지션 예측

    비디오 분석으로 현재 포지션을 찾지 못한 경우,
    이전 손가락 위치에서 가장 가까운 위치를 예측합니다.

    예측 기준:
    1. 이전 포지션이 없으면 → 가장 낮은 프렛 선택 (연주 시작 위치)
    2. 이전 포지션이 있으면 → 손가락 이동 거리가 최소인 위치 선택
       - 프렛 간 이동: 거리 1
       - 현 간 이동: 거리 5 (STRING_MOVE_WEIGHT)

    Parameters
    ----------
    target_midi : int
        찾고자 하는 MIDI 번호
    last_pos : Dict[str, any], optional
        이전 포지션. {'string': str, 'fret': int} 형태
        None이면 초기 상태로 간주

    Returns
    -------
    best_position : Dict[str, any]
        예측된 최적 포지션 {'string': str, 'fret': int|str}
        불가능한 MIDI면 {'string': '?', 'fret': '?'} 반환

    Examples
    --------
    >>> # 이전 포지션 없음 → 가장 낮은 프렛 선택
    >>> predict_best_position(33)
    {'string': 'E', 'fret': 5}

    >>> # 이전 포지션: A-5프렛 → A-7프렛 (같은 현, 2프렛 이동)
    >>> predict_best_position(35, {'string': 'A', 'fret': 5})
    {'string': 'A', 'fret': 7}

    >>> # 이전 포지션: D-3프렛 → D-5프렛 (같은 현, 2프렛 이동)
    >>> # E-15프렛도 가능하지만 거리가 더 멈
    >>> predict_best_position(43, {'string': 'D', 'fret': 3})
    {'string': 'D', 'fret': 5}

    Notes
    -----
    거리 계산 공식:
    distance = |fret_diff| + |string_diff| × STRING_MOVE_WEIGHT

    현 간 이동을 프렛 이동보다 5배 더 비용이 크다고 간주하는 이유:
    - 사람의 손가락 이동 패턴 상 같은 현에서 프렛 이동이 더 자연스러움
    - 현을 바꾸려면 손목/팔꿈치 각도도 조정해야 함
    """
    # 1. 가능한 포지션 찾기
    candidates = get_possible_positions(target_midi)

    # 불가능한 MIDI 번호
    if not candidates:
        return {'string': '?', 'fret': '?'}

    # 2. 이전 포지션이 없으면 → 가장 낮은 프렛 선택
    if not last_pos:
        # 프렛 번호 기준으로 정렬 (낮은 프렛 = 연주 시작 위치)
        candidates.sort(key=lambda x: x['fret'])
        return candidates[0]

    # 3. 이전 포지션이 있으면 → 거리 기반 선택

    # 현 이름 → 인덱스 매핑
    string_map = {'E': 0, 'A': 1, 'D': 2, 'G': 3}

    # 이전 위치 인덱스
    last_string_idx = string_map.get(last_pos['string'], 0)
    last_fret = last_pos['fret']

    # 최소 거리 탐색
    best_candidate = None
    min_distance = float('inf')

    for candidate in candidates:
        curr_string_idx = string_map.get(candidate['string'], 0)
        curr_fret = candidate['fret']

        # 거리 계산: 프렛 이동 + 현 이동 × 가중치
        fret_diff = abs(curr_fret - last_fret)
        string_diff = abs(curr_string_idx - last_string_idx)

        distance = fret_diff + string_diff * STRING_MOVE_WEIGHT

        if distance < min_distance:
            min_distance = distance
            best_candidate = candidate

    return best_candidate


def get_midi_note_name(midi_note: int) -> str:
    """
    MIDI 번호를 음이름으로 변환 (디버깅/로깅용)

    Parameters
    ----------
    midi_note : int
        MIDI 번호 (0~127)

    Returns
    -------
    note_name : str
        음이름 (예: 'E1', 'A2', 'D#3')

    Examples
    --------
    >>> get_midi_note_name(28)
    'E1'
    >>> get_midi_note_name(33)
    'A1'
    >>> get_midi_note_name(38)
    'D2'
    """
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']

    octave = (midi_note // 12) - 1
    note = note_names[midi_note % 12]

    return f"{note}{octave}"


def validate_position(string_name: str, fret_num: int) -> bool:
    """
    포지션의 유효성 검증

    Parameters
    ----------
    string_name : str
        현 이름
    fret_num : int
        프렛 번호

    Returns
    -------
    is_valid : bool
        유효한 포지션이면 True

    Examples
    --------
    >>> validate_position('E', 5)
    True
    >>> validate_position('X', 0)  # 잘못된 현
    False
    >>> validate_position('E', 25)  # 범위 초과
    False
    """
    # 현 이름 체크
    if string_name not in STRING_OFFSETS:
        return False

    # 프렛 범위 체크
    if not (MIN_FRET <= fret_num <= MAX_FRET):
        return False

    return True