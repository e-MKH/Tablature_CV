"""
Hand and Fret Detection Module

MediaPipe를 사용한 손 인식과 YOLO를 사용한 프렛 탐지
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

from ..config.vision_config import (
    MEDIAPIPE_CONFIG,
    YOLO_CONFIG,
    TARGET_HAND,
    PRESCAN_CONFIG
)
from .bass_geometry import (
    get_geometry,
    project_ratio,
    calculate_fret_positions_math,
    merge_fret_ratios,
    get_finger_tip_indices
)
from ..utils.image_utils import preprocess_image, enhance_for_detection


class HandDetector:
    """
    MediaPipe Hands를 사용한 손 인식기

    손의 랜드마크를 감지하고 손가락 끝 좌표를 추출합니다.
    """

    def __init__(self, target_hand=TARGET_HAND):
        """
        Args:
            target_hand: 감지할 손 ("Right" or "Left")
        """
        self.target_hand = target_hand
        self.mp_hands = mp.solutions.hands

        # MediaPipe Hands 초기화
        self.hands = self.mp_hands.Hands(
            static_image_mode=MEDIAPIPE_CONFIG["static_image_mode"],
            max_num_hands=MEDIAPIPE_CONFIG["max_num_hands"],
            model_complexity=MEDIAPIPE_CONFIG["model_complexity"],
            min_detection_confidence=MEDIAPIPE_CONFIG["min_detection_confidence"],
            min_tracking_confidence=MEDIAPIPE_CONFIG["min_tracking_confidence"]
        )

        # 손가락 끝 인덱스
        self.finger_tips = get_finger_tip_indices()

        print(f"✅ HandDetector initialized (Target: {self.target_hand})")

    def detect(self, frame, use_enhancement=False):
        """
        손 감지 수행

        Args:
            frame: 입력 프레임 (BGR)
            use_enhancement: 향상된 전처리 사용 여부

        Returns:
            mediapipe.python.solution_base.SolutionOutputs or None
        """
        # 전처리
        if use_enhancement:
            processed = enhance_for_detection(frame)
        else:
            processed = preprocess_image(frame)

        # RGB 변환 후 감지
        rgb_frame = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        return results

    def extract_fingertips(self, results, frame_shape):
        """
        손가락 끝 좌표 추출

        Args:
            results: MediaPipe 감지 결과
            frame_shape: 프레임 크기 (height, width, channels)

        Returns:
            list: [(finger_name, x, y), ...]
        """
        if not results.multi_hand_landmarks:
            return []

        h, w = frame_shape[:2]
        fingertips = []

        for hand_lms, handedness in zip(results.multi_hand_landmarks,
                                        results.multi_handedness):
            # 타겟 손만 처리
            if handedness.classification[0].label != self.target_hand:
                continue

            # 각 손가락 끝 좌표 추출
            for finger_name, idx in self.finger_tips.items():
                landmark = hand_lms.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                fingertips.append((finger_name, x, y))

        return fingertips

    def close(self):
        """리소스 해제"""
        self.hands.close()


class FretDetector:
    """
    YOLO를 사용한 프렛 탐지기

    기타 넥의 프렛을 감지합니다.
    """

    def __init__(self, model_path):
        """
        Args:
            model_path: YOLO 모델 파일 경로 (.pt)
        """
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.verbose = YOLO_CONFIG["verbose"]
        print(f"✅ FretDetector initialized")

    def detect(self, frame):
        """
        프렛 감지 수행

        Args:
            frame: 입력 프레임 (BGR)

        Returns:
            ultralytics.engine.results.Results
        """
        results = self.model(frame, verbose=self.verbose)
        return results

    def extract_fret_boxes(self, results):
        """
        프렛 바운딩 박스 추출

        Args:
            results: YOLO 감지 결과

        Returns:
            list: [(x1, y1, x2, y2, confidence), ...]
        """
        fret_boxes = []

        for r in results:
            for box in r.boxes:
                class_name = self.model.names[int(box.cls[0])].lower()

                # 'fret'이 포함된 클래스만 필터링
                if 'fret' in class_name:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    fret_boxes.append((x1, y1, x2, y2, conf))

        return fret_boxes

    def get_fret_centers(self, fret_boxes):
        """
        프렛 바운딩 박스의 중심점 계산

        Args:
            fret_boxes: [(x1, y1, x2, y2, conf), ...]

        Returns:
            list: [(cx, cy), ...]
        """
        centers = []
        for x1, y1, x2, y2, conf in fret_boxes:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centers.append((cx, cy))

        return centers


class PreScanner:
    """
    프렛 위치 사전 스캔 (Pre-Scan)

    영상의 여러 구간에서 YOLO로 프렛을 탐지하여
    프렛 위치 맵을 학습합니다.
    """

    def __init__(self, fret_detector):
        """
        Args:
            fret_detector: FretDetector 인스턴스
        """
        self.fret_detector = fret_detector

        # Pre-Scan 설정
        self.scan_duration_sec = PRESCAN_CONFIG["scan_duration_sec"]
        self.scan_positions = PRESCAN_CONFIG["scan_positions"]
        self.fret_tolerance = PRESCAN_CONFIG["fret_tolerance"]

    def scan(self, video_path, initial_corners, lk_params):
        """
        Pre-Scan 실행

        Args:
            video_path: 비디오 파일 경로
            initial_corners: 초기 코너 좌표 (4x2)
            lk_params: Optical Flow 파라미터

        Returns:
            list: 학습된 프렛 비율 리스트
        """
        print(">>> Starting Pre-Scan Sequence...")

        # 비디오 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("⚠ Failed to open video for pre-scan")
            return self._fallback_fret_positions()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        scan_duration_frames = int(fps * self.scan_duration_sec)

        # 스캔할 프레임 위치 계산
        target_frames = [int(total_frames * pos) for pos in self.scan_positions]

        # 스캔 수행
        collected_ratios = self._perform_scan_pass(
            cap, target_frames, scan_duration_frames,
            initial_corners, lk_params
        )

        cap.release()

        # 결과 처리
        if not collected_ratios:
            print("⚠ No frets detected, using math fallback")
            return self._fallback_fret_positions()

        # 프렛 비율 병합
        collected_ratios.sort()
        merged_ratios = merge_fret_ratios(collected_ratios, self.fret_tolerance)

        print(f"✅ Pre-Scan Complete! {len(merged_ratios)} frets mapped.")
        return merged_ratios

    def _perform_scan_pass(self, cap, target_frames, duration_frames,
                           initial_corners, lk_params):
        """
        스캔 패스 수행
        """
        from ..utils.image_utils import resize_frame  # 추가!

        collected_ratios = []
        prev_gray = None
        curr_corners = initial_corners.reshape(-1, 1, 2)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame_idx > target_frames[-1] + duration_frames:
                break

            frame = resize_frame(frame)  # ← 이 줄 추가!
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Optical Flow로 코너 추적
            if prev_gray is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    prev_gray, frame_gray, curr_corners, None, **lk_params
                )
                curr_corners = p1

            prev_gray = frame_gray.copy()

            # 현재 프레임이 스캔 구간인지 확인
            is_scanning = False
            for target in target_frames:
                if target <= frame_idx < target + duration_frames:
                    is_scanning = True
                    break

            # 스캔 수행
            if is_scanning:
                corners_flat = curr_corners.reshape(4, 2)
                nut_center, unit_vec, total_len = get_geometry(corners_flat)

                # YOLO로 프렛 탐지
                results = self.fret_detector.detect(frame)
                fret_boxes = self.fret_detector.extract_fret_boxes(results)
                fret_centers = self.fret_detector.get_fret_centers(fret_boxes)

                # 프렛 비율 계산
                for cx, cy in fret_centers:
                    ratio = project_ratio(
                        np.array([cx, cy]), nut_center, unit_vec, total_len
                    )
                    # 유효 범위 체크
                    if 0 < ratio < 1.1:
                        collected_ratios.append(ratio)

                # 진행 상황 표시 (5프레임마다)
                if frame_idx % 5 == 0:
                    display = frame.copy()
                    cv2.putText(display, "Scanning...", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Bass Tab System', display)
                    cv2.waitKey(1)

            frame_idx += 1

        return collected_ratios

    def _fallback_fret_positions(self):
        """
        수학적 계산으로 프렛 위치 생성 (fallback)

        Returns:
            list: 프렛 비율 리스트
        """
        print("⚠ Using mathematical fallback for fret positions")
        return calculate_fret_positions_math(num_frets=21)