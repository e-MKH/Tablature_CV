"""
Bass Tablature System

베이스 기타 타블러처 분석을 위한 메인 시스템
모든 모듈을 통합하여 비디오에서 타블러처를 추출합니다.
"""

import cv2
import numpy as np
import time

from .calibration import SimpleCalibrationUI
from .filters import MultiPointFilter
from .tracking import CornerTracker
from .detection import HandDetector, FretDetector, PreScanner
from .bass_geometry import (
    get_geometry,
    get_strings_lines,
    detect_string,
    detect_fret,
    get_midi_note,
    get_finger_names,
    get_string_names
)
from ..config.vision_config import (
    MAX_MISSING_TOLERANCE,
    PRESCAN_CONFIG,
    VISUALIZATION,
    OPTICAL_FLOW_PARAMS
)
from ..utils.image_utils import resize_frame


class BassTabSystem:
    """
    베이스 타블러처 분석 시스템

    비디오에서 베이스 기타 연주를 분석하여 타블러처 데이터를 추출합니다.
    """

    def __init__(self, model_path):
        """
        Args:
            model_path: YOLO 모델 파일 경로
        """
        print("=" * 60)
        print("Initializing Bass Tablature System")
        print("=" * 60)

        # 컴포넌트 초기화
        self.hand_detector = HandDetector()
        self.fret_detector = FretDetector(model_path)
        self.corner_tracker = CornerTracker()
        self.pre_scanner = PreScanner(self.fret_detector)

        # 필터 초기화
        self.filters = MultiPointFilter()

        # 상태 변수
        self.learned_fret_ratios = []
        self.start_time = time.time()

        # 손 인식 실패 보완용
        self.last_valid_fingers = []
        self.missing_frames_count = 0
        self.max_missing_tolerance = MAX_MISSING_TOLERANCE

        # 현과 손가락 이름
        self.string_names = get_string_names()
        self.finger_names = get_finger_names()

        print("✅ BassTabSystem initialized successfully")
        print("=" * 60)

    def run_pre_scan(self, video_path, initial_corners):
        """
        Pre-Scan 실행하여 프렛 위치 학습

        Args:
            video_path: 비디오 파일 경로
            initial_corners: 초기 코너 좌표
        """
        self.learned_fret_ratios = self.pre_scanner.scan(
            video_path,
            initial_corners,
            self.corner_tracker.lk_params
        )

    def init_tracking(self, frame, corners):
        """
        추적 초기화

        Args:
            frame: 첫 프레임
            corners: 코너 좌표
        """
        self.corner_tracker.init_tracking(frame, corners)

    def apply_one_euro(self, finger_name, pos):
        """
        One Euro Filter 적용

        Args:
            finger_name: 손가락 이름 ("I", "M", "R", "P")
            pos: (x, y) 좌표

        Returns:
            tuple: 필터링된 (x, y) 좌표
        """
        current_time = time.time() - self.start_time
        filtered_x, filtered_y = self.filters.process(current_time, finger_name, pos)
        return int(filtered_x), int(filtered_y)

    def analyze(self, frame, corners, current_time_sec):
        """
        프레임 분석

        Args:
            frame: 현재 프레임
            corners: 현재 코너 좌표 (4x2)
            current_time_sec: 현재 시간 (초)

        Returns:
            tuple: (detected_data_list, visual_data)
                - detected_data_list: JSON 저장용 데이터
                - visual_data: 화면 표시용 데이터
        """
        corners_flat = corners.reshape(4, 2)
        nut_center, unit_vec, total_len = get_geometry(corners_flat)
        str_lines, div_lines = get_strings_lines(corners_flat)

        # 손 인식 시도 (1차: 일반 전처리)
        mp_results = self.hand_detector.detect(frame, use_enhancement=False)

        # 실패 시 2차 시도 (향상된 전처리)
        if not mp_results.multi_hand_landmarks:
            mp_results = self.hand_detector.detect(frame, use_enhancement=True)

        # 시각화: 구분선과 현 그리기
        self._draw_strings_and_dividers(frame, str_lines, div_lines)

        # 시각화: 프렛 마커 그리기
        self._draw_fret_markers(frame, nut_center, unit_vec, total_len)

        detected_data_list = []
        visual_data = []
        is_tracking_success = False

        # 손 감지 성공
        if mp_results.multi_hand_landmarks:
            is_tracking_success = True
            self.missing_frames_count = 0

            # 손가락 끝 좌표 추출
            fingertips = self.hand_detector.extract_fingertips(mp_results, frame.shape)

            for finger_name, raw_x, raw_y in fingertips:
                # One Euro Filter 적용
                cx, cy = self.apply_one_euro(finger_name, (raw_x, raw_y))

                # 손가락 시각화
                cv2.circle(frame, (cx, cy),
                           VISUALIZATION["finger_marker_radius"],
                           VISUALIZATION["finger_color"], -1)

                # 현 감지
                det_str = detect_string(
                    (cx, cy),
                    str_lines,
                    VISUALIZATION["string_detection_threshold"]
                )

                # 프렛 감지
                if det_str != -1 and self.learned_fret_ratios:
                    from .bass_geometry import project_ratio

                    f_ratio = project_ratio(
                        np.array([cx, cy]), nut_center, unit_vec, total_len
                    )

                    det_fret = detect_fret(
                        f_ratio,
                        self.learned_fret_ratios,
                        PRESCAN_CONFIG["ratio_tolerance"]
                    )

                    if det_fret != -1:
                        # 데이터 저장
                        note_data = {
                            "time_sec": round(current_time_sec, 3),
                            "mid": get_midi_note(self.string_names[det_str], det_fret),
                            "string": self.string_names[det_str],
                            "fret": det_fret,
                            "finger": finger_name,
                            "is_tracking": True  # 실제 추적됨
                        }
                        detected_data_list.append(note_data)
                        visual_data.append(note_data)

                        # 프렛 정보 표시
                        cv2.putText(
                            frame,
                            f"{self.string_names[det_str]}/{det_fret}",
                            (cx, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            VISUALIZATION["finger_color"],
                            2
                        )

            # 유효한 손가락 정보 저장
            if detected_data_list:
                self.last_valid_fingers = detected_data_list

        # 손 감지 실패 시 메모리 사용
        if not is_tracking_success:
            self.missing_frames_count += 1

            # 허용 범위 내에서 마지막 값 사용
            if (self.missing_frames_count <= self.max_missing_tolerance
                    and self.last_valid_fingers):

                # 메모리 모드 표시
                cv2.putText(
                    frame,
                    "MEMORY (Lost)",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (100, 100, 100),
                    2
                )

                # 마지막 유효 데이터 사용
                for item in self.last_valid_fingers:
                    new_item = item.copy()
                    new_item['time_sec'] = round(current_time_sec, 3)
                    new_item['is_tracking'] = False  # 추정값
                    detected_data_list.append(new_item)

        return detected_data_list, visual_data

    def _draw_strings_and_dividers(self, frame, str_lines, div_lines):
        """
        현과 구분선 그리기

        Args:
            frame: 프레임
            str_lines: 현 라인 리스트
            div_lines: 구분선 리스트
        """
        # 구분선 (반투명)
        overlay = frame.copy()
        for p1, p2 in div_lines:
            cv2.line(
                overlay, p1, p2,
                VISUALIZATION["divider_color"],
                VISUALIZATION["divider_line_thickness"]
            )
        cv2.addWeighted(
            overlay,
            VISUALIZATION["divider_opacity"],
            frame,
            1.0 - VISUALIZATION["divider_opacity"],
            0,
            frame
        )

        # 현 (불투명)
        for p1, p2 in str_lines:
            cv2.line(
                frame, p1, p2,
                VISUALIZATION["string_color"],
                VISUALIZATION["string_line_thickness"]
            )

    def _draw_fret_markers(self, frame, nut_center, unit_vec, total_len):
        """
        프렛 마커 그리기

        Args:
            frame: 프레임
            nut_center: 넛 중심
            unit_vec: 넥 방향 단위 벡터
            total_len: 넥 길이
        """
        for i, ratio in enumerate(self.learned_fret_ratios):
            pos = nut_center + unit_vec * (total_len * ratio)
            cv2.circle(
                frame,
                (int(pos[0]), int(pos[1])),
                VISUALIZATION["fret_marker_radius"],
                VISUALIZATION["fret_color"],
                -1
            )

    def print_tab(self, fingers):
        """
        터미널에 타블러처 출력

        Args:
            fingers: 손가락 데이터 리스트
        """
        if not fingers:
            return

        output = " | ".join([
            f"[{f['finger']}] {f['string']} {f['fret']}"
            for f in fingers
        ])
        print(f"\r▶ {output}                ", end="")

    def close(self):
        """리소스 해제"""
        self.hand_detector.close()
        cv2.destroyAllWindows()