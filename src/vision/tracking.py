"""
Corner Tracking Module

Optical Flow를 사용한 기타 넥 코너 추적
"""

import cv2
import numpy as np
from ..config.vision_config import OPTICAL_FLOW_PARAMS


class CornerTracker:
    """
    Lucas-Kanade Optical Flow를 사용한 코너 추적기

    기타 넥의 4개 코너를 실시간으로 추적하여
    프레임 간 기타의 위치/각도 변화를 감지합니다.
    """

    def __init__(self):
        """초기화"""
        self.prev_gray = None
        self.corners = None

        # Optical Flow 파라미터 설정
        self.lk_params = dict(
            winSize=OPTICAL_FLOW_PARAMS["winSize"],
            maxLevel=OPTICAL_FLOW_PARAMS["maxLevel"],
            criteria=OPTICAL_FLOW_PARAMS["criteria"]
        )

    def init_tracking(self, frame, corners):
        """
        추적 초기화

        Args:
            frame: 첫 프레임 (BGR)
            corners: 초기 코너 좌표 (4x2 numpy array)
        """
        # 그레이스케일 변환
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 코너를 OpenCV Optical Flow 형식으로 변환 (N, 1, 2)
        self.corners = corners.reshape(-1, 1, 2).astype(np.float32)

        print(f"✅ Corner tracking initialized")

    def update_tracking(self, frame):
        """
        코너 위치 업데이트

        Args:
            frame: 현재 프레임 (BGR)

        Returns:
            np.ndarray: 업데이트된 코너 좌표 (4x2) 또는 None (추적 실패 시)
        """
        if self.corners is None:
            print("⚠ Tracking not initialized")
            return None

        # 그레이스케일 변환
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical Flow로 코너 추적
        p1, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            frame_gray,
            self.corners,
            None,
            **self.lk_params
        )

        # 추적 실패 확인
        if p1 is None or not np.all(status):
            print("⚠ Corner tracking lost")
            return None

        # 코너 업데이트
        self.corners = p1
        self.prev_gray = frame_gray.copy()

        # (4, 2) 형태로 반환
        return p1.reshape(4, 2)

    def reset(self, frame, corners):
        """
        추적 재초기화 (Space 키 입력 시)

        Args:
            frame: 현재 프레임
            corners: 새로운 코너 좌표
        """
        print("\n>>> Resetting corner tracker...")
        self.init_tracking(frame, corners)
        print(">>> Corner tracker reset complete")

    def is_initialized(self):
        """
        추적기 초기화 여부 확인

        Returns:
            bool: 초기화 여부
        """
        return self.corners is not None

    def get_current_corners(self):
        """
        현재 코너 좌표 반환

        Returns:
            np.ndarray: 현재 코너 좌표 (4x2) 또는 None
        """
        if self.corners is None:
            return None
        return self.corners.reshape(4, 2)

    def visualize_corners(self, frame, color=(0, 255, 255), radius=5):
        """
        코너를 프레임에 시각화

        Args:
            frame: 프레임 (BGR)
            color: 코너 색상 (기본값: cyan)
            radius: 코너 반지름 (기본값: 5)

        Returns:
            np.ndarray: 시각화된 프레임
        """
        if self.corners is None:
            return frame

        display = frame.copy()
        corners_flat = self.corners.reshape(4, 2)

        # 각 코너 그리기
        for i, (x, y) in enumerate(corners_flat):
            cv2.circle(display, (int(x), int(y)), radius, color, -1)

            # 코너 번호 표시
            label = ["Nut-T", "Nut-B", "Body-T", "Body-B"][i]
            cv2.putText(display, label, (int(x) + 10, int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 사각형 그리기
        pts = corners_flat.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(display, [pts], True, color, 2)

        return display


class CornerValidator:
    """
    코너 추적 유효성 검증기

    추적된 코너가 비정상적인 위치에 있는지 검증합니다.
    """

    def __init__(self, max_displacement_ratio=0.1):
        """
        Args:
            max_displacement_ratio: 프레임 간 최대 이동 비율 (기본값: 0.1)
        """
        self.max_displacement_ratio = max_displacement_ratio
        self.prev_corners = None

    def validate(self, corners, frame_shape):
        """
        코너 유효성 검증

        Args:
            corners: 추적된 코너 (4x2)
            frame_shape: 프레임 크기 (height, width)

        Returns:
            bool: 유효하면 True, 그렇지 않으면 False
        """
        h, w = frame_shape[:2]

        # 1. 경계 체크 (프레임 안에 있는지)
        if not self._check_bounds(corners, w, h):
            return False

        # 2. 사각형 형태 체크 (너무 찌그러지지 않았는지)
        if not self._check_quadrilateral(corners):
            return False

        # 3. 이동 거리 체크 (급격한 변화가 없는지)
        if self.prev_corners is not None:
            if not self._check_displacement(corners, self.prev_corners, w, h):
                return False

        # 검증 통과 - 현재 코너 저장
        self.prev_corners = corners.copy()
        return True

    def _check_bounds(self, corners, width, height):
        """코너가 프레임 안에 있는지 확인"""
        margin = 10
        for x, y in corners:
            if x < margin or x > width - margin or y < margin or y > height - margin:
                return False
        return True

    def _check_quadrilateral(self, corners):
        """사각형 형태가 유지되는지 확인"""
        # 간단한 체크: 대각선 길이 비율
        lt, lb, rt, rb = corners

        diag1 = np.linalg.norm(rb - lt)
        diag2 = np.linalg.norm(rt - lb)

        if diag1 == 0 or diag2 == 0:
            return False

        ratio = max(diag1, diag2) / min(diag1, diag2)

        # 대각선 비율이 너무 차이나면 비정상
        if ratio > 3.0:
            return False

        return True

    def _check_displacement(self, curr_corners, prev_corners, width, height):
        """프레임 간 이동 거리가 과도하지 않은지 확인"""
        max_dist = max(width, height) * self.max_displacement_ratio

        for curr, prev in zip(curr_corners, prev_corners):
            dist = np.linalg.norm(curr - prev)
            if dist > max_dist:
                return False

        return True

    def reset(self):
        """검증기 리셋"""
        self.prev_corners = None