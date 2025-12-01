"""
Vision Analysis Configuration

비전 분석에 필요한 모든 설정값을 정의합니다.
"""

# =========================================================
# [기본 설정]
# =========================================================

# 타겟 손 (MediaPipe가 인식할 손)
TARGET_HAND = "Right"  # "Right" or "Left"

# 영상 리사이징 최대 너비
MAX_WIDTH = 1280

# 손 추적 실패 시 메모리 유지 프레임 수 (약 1초)
MAX_MISSING_TOLERANCE = 30


# =========================================================
# [MediaPipe 설정]
# =========================================================

MEDIAPIPE_CONFIG = {
    "static_image_mode": False,      # 비디오 모드 (연속 프레임 처리)
    "max_num_hands": 2,              # 최대 감지 손 개수
    "model_complexity": 1,           # 모델 복잡도 (0=Lite, 1=Full)
    "min_detection_confidence": 0.3, # 최소 감지 신뢰도
    "min_tracking_confidence": 0.3   # 최소 추적 신뢰도
}


# =========================================================
# [YOLO 설정]
# =========================================================

YOLO_CONFIG = {
    "verbose": False,                # 출력 메시지 표시 여부
    "conf_threshold": 0.5,           # 신뢰도 임계값
    "iou_threshold": 0.45            # NMS IoU 임계값
}


# =========================================================
# [Optical Flow 추적 설정]
# =========================================================

OPTICAL_FLOW_PARAMS = {
    "winSize": (21, 21),             # 탐색 윈도우 크기
    "maxLevel": 3,                   # 피라미드 레벨
    "criteria": (
        2 | 1,  # TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT
        30,     # maxCount
        0.01    # epsilon
    )
}


# =========================================================
# [기타 설정]
# =========================================================

# 프렛 감지 허용 오차 (비율)
FRET_TOLERANCE = 0.02  # 2%

# 현 구분선 마진 (상하 여백 비율)
STRING_MARGIN = 0.12

# 감마 보정값 (손 인식 실패 시)
GAMMA_CORRECTION = 1.5

# 손가락 끝 인덱스 (MediaPipe landmark 번호)
FINGER_TIP_INDICES = {
    "I": 8,   # Index finger (검지)
    "M": 12,  # Middle finger (중지)
    "R": 16,  # Ring finger (약지)
    "P": 20   # Pinky finger (새끼)
}

# 현 이름 (베이스 기타 4현)
STRING_NAMES = ["E", "A", "D", "G"]

# MIDI 음높이 기준값 (개방현)
MIDI_BASE_PITCHES = {
    'E': 28,  # E1
    'A': 33,  # A1
    'D': 38,  # D2
    'G': 43   # G2
}

# 현 감지 허용 오차 (픽셀)
STRING_DETECTION_TOLERANCE = 25


# =========================================================
# [Pre-Scan 설정 딕셔너리]
# =========================================================

PRESCAN_CONFIG = {
    "scan_duration_sec": 1.0,                    # 각 구간 스캔 시간 (초)
    "scan_positions": [0, 0.25, 0.50, 0.75],    # 스캔할 영상 위치 (비율)
    "fret_tolerance": 0.013,                     # 프렛 병합 임계값
    "ratio_tolerance": 0.02                      # 프렛 매칭 허용 오차 (2%)
}


# =========================================================
# [One Euro Filter 설정]
# =========================================================

FILTER_CONFIG = {
    "min_cutoff": 1.0,
    "beta": 0.05,
    "d_cutoff": 1.0
}


# =========================================================
# [시각화 설정 딕셔너리]
# =========================================================

VISUALIZATION = {
    "string_color": (0, 255, 0),           # 현 색상 (초록)
    "divider_color": (0, 0, 255),          # 구분선 색상 (빨강)
    "divider_opacity": 0.3,                # 구분선 투명도
    "fret_color": (255, 255, 0),           # 프렛 마커 색상 (노랑)
    "finger_color": (0, 255, 255),         # 손가락 색상 (cyan)
    "string_line_thickness": 2,
    "divider_line_thickness": 2,
    "fret_marker_radius": 3,
    "finger_marker_radius": 3,
    "string_detection_threshold": 25       # 손가락-현 거리 임계값 (픽셀)
}