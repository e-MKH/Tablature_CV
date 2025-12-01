"""
Geometry and Coordinate Mapping Module

기타 넥의 기하학적 계산 및 좌표 매핑 함수들
"""

import numpy as np
from ..config.vision_config import (
    STRING_NAMES,
    STRING_MARGIN,
    MIDI_BASE_PITCHES
)


def get_geometry(corners):
    """
    기타 넥의 기하학적 정보 계산

    Args:
        corners: 4개의 코너 좌표 (4x2 numpy array)
                [nut_top, nut_bottom, body_top, body_bottom]

    Returns:
        tuple: (nut_center, unit_vec, neck_len)
            - nut_center: 넛 중심점
            - unit_vec: 넥 방향 단위 벡터
            - neck_len: 넥 길이
    """
    lt, lb, rt, rb = corners

    # 넛(nut) 중심과 바디(body) 중심 계산
    nut_center = (lt + lb) / 2.0
    body_center = (rt + rb) / 2.0

    # 넥 벡터 계산
    neck_vector = body_center - nut_center
    neck_len = np.linalg.norm(neck_vector)

    # 길이가 0이면 예외 처리
    if neck_len == 0:
        neck_len = 1

    # 단위 벡터 계산
    unit_vec = neck_vector / neck_len

    return nut_center, unit_vec, neck_len


def project_ratio(point, origin, unit_vec, total_len):
    """
    포인트를 넥 방향으로 투영하여 비율 계산

    Args:
        point: 투영할 포인트 (x, y)
        origin: 원점 (넛 중심)
        unit_vec: 넥 방향 단위 벡터
        total_len: 넥 전체 길이

    Returns:
        float: 넛으로부터의 거리 비율 (0.0 ~ 1.0+)
    """
    vec = point - origin
    dist = np.dot(vec, unit_vec)
    return dist / total_len


def get_strings_lines(corners):
    """
    4개 현(string)의 라인 좌표 계산

    Args:
        corners: 4개의 코너 좌표 (4x2 numpy array)

    Returns:
        tuple: (string_lines, divider_lines)
            - string_lines: 4개 현의 시작/끝 좌표 리스트
            - divider_lines: 3개 구분선의 시작/끝 좌표 리스트
    """
    lt, lb, rt, rb = corners

    # 왼쪽과 오른쪽 벡터
    left_vec = lb - lt
    right_vec = rb - rt

    # 현 위치 계산 (상하 여백 포함)
    MARGIN = STRING_MARGIN
    string_lines = []

    for i in range(4):
        # 각 현의 비율 계산 (0번=E현, 3번=G현)
        r = MARGIN + (i * (1 - 2 * MARGIN) / 3.0)

        # 시작점과 끝점
        start = lt + left_vec * r
        end = rt + right_vec * r

        string_lines.append((
            (int(start[0]), int(start[1])),
            (int(end[0]), int(end[1]))
        ))

    # 현 사이 구분선 계산
    divider_lines = []
    for i in range(3):
        r1 = MARGIN + (i * (1 - 2 * MARGIN) / 3.0)
        r2 = MARGIN + ((i + 1) * (1 - 2 * MARGIN) / 3.0)
        mid = (r1 + r2) / 2.0

        start = lt + left_vec * mid
        end = rt + right_vec * mid

        divider_lines.append((
            (int(start[0]), int(start[1])),
            (int(end[0]), int(end[1]))
        ))

    return string_lines, divider_lines


def get_midi_note(string_name, fret_number):
    """
    현과 프렛 번호로 MIDI 음높이 계산

    Args:
        string_name: 현 이름 ("E", "A", "D", "G")
        fret_number: 프렛 번호 (0=개방현, 1=1프렛, ...)

    Returns:
        int: MIDI 노트 번호
    """
    base_pitch = MIDI_BASE_PITCHES.get(string_name, 0)
    return base_pitch + fret_number


def detect_string(finger_pos, string_lines, threshold=25):
    """
    손가락 위치에서 가장 가까운 현 찾기

    Args:
        finger_pos: 손가락 좌표 (x, y)
        string_lines: 현 라인 리스트
        threshold: 허용 거리 임계값 (픽셀)

    Returns:
        int: 현 인덱스 (0~3) 또는 -1 (감지 실패)
    """
    cx, cy = finger_pos
    detected_string = -1
    min_dist = 999

    for si, ((sx1, sy1), (sx2, sy2)) in enumerate(string_lines):
        # 수직선 방지
        if sx2 == sx1:
            continue

        # 손가락 x 좌표에서 현의 y 좌표 계산 (선형 보간)
        line_y = sy1 + (sy2 - sy1) * ((cx - sx1) / (sx2 - sx1))

        # 거리 계산
        dist = abs(cy - line_y)

        # 임계값 내에서 가장 가까운 현 선택
        if dist < threshold and dist < min_dist:
            min_dist = dist
            detected_string = si

    return detected_string


def detect_fret(finger_ratio, learned_fret_ratios, tolerance=0.02):
    """
    손가락 비율로 프렛 번호 찾기

    Args:
        finger_ratio: 넛으로부터의 거리 비율
        learned_fret_ratios: 학습된 프렛 비율 리스트
        tolerance: 허용 오차 비율 (기본값: 2%)

    Returns:
        int: 프렛 번호 또는 -1 (감지 실패)
    """
    if not learned_fret_ratios:
        return -1

    detected_fret = -1

    # 프렛 비율과 비교
    for fi, fr in enumerate(learned_fret_ratios):
        if finger_ratio <= (fr + tolerance):
            detected_fret = fi + 1
            break

    # 마지막 프렛을 넘어선 경우
    if finger_ratio > (learned_fret_ratios[-1] + tolerance):
        detected_fret = len(learned_fret_ratios)

    return detected_fret


def calculate_fret_positions_math(num_frets=21):
    """
    수학적 공식으로 프렛 위치 계산 (fallback)

    12평균율 기반: 각 프렛은 (2^(1/12))배 간격

    Args:
        num_frets: 프렛 개수 (기본값: 21)

    Returns:
        list: 프렛 비율 리스트
    """
    k = 2 ** (-1 / 12)  # 반음 간격 비율
    scale_len = 1.0 / (1 - k ** num_frets)

    fret_ratios = []
    for i in range(1, num_frets + 1):
        ratio = scale_len * (1 - k ** i)
        fret_ratios.append(ratio)

    return fret_ratios


def merge_fret_ratios(ratios, merge_threshold=0.013):
    """
    유사한 프렛 비율들을 그룹화하여 평균

    Args:
        ratios: 프렛 비율 리스트 (정렬되어 있어야 함)
        merge_threshold: 병합 임계값

    Returns:
        list: 병합된 프렛 비율 리스트
    """
    if not ratios:
        return []

    ratios_sorted = sorted(ratios)
    merged_ratios = []
    curr_group = [ratios_sorted[0]]

    for r in ratios_sorted[1:]:
        # 현재 그룹의 평균과 비교
        if r - np.mean(curr_group) < merge_threshold:
            curr_group.append(r)
        else:
            # 그룹 완료 - 평균 저장
            merged_ratios.append(np.mean(curr_group))
            curr_group = [r]

    # 마지막 그룹 추가
    merged_ratios.append(np.mean(curr_group))

    return merged_ratios


def get_finger_names():
    """
    손가락 이름 반환

    Returns:
        list: ["I", "M", "R", "P"]
    """
    return ["I", "M", "R", "P"]


def get_finger_tip_indices():
    """
    MediaPipe 손가락 끝 랜드마크 인덱스

    Returns:
        dict: {finger_name: landmark_index}
    """
    return {
        "I": 8,  # Index finger (검지)
        "M": 12,  # Middle finger (중지)
        "R": 16,  # Ring finger (약지)
        "P": 20  # Pinky finger (새끼)
    }


def get_string_names():
    """
    현 이름 반환

    Returns:
        list: ["E", "A", "D", "G"]
    """
    return STRING_NAMES