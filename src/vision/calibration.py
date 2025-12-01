"""
Guitar Neck Calibration UI Module

사용자가 기타 넥의 4개 코너를 지정하는 보정 UI를 제공합니다.
"""

import cv2
import numpy as np


class SimpleCalibrationUI:
    """
    기타 넥 보정을 위한 간단한 UI 클래스

    사용자가 4개의 코너 포인트를 클릭하여 지정:
    1. Nut Top (넛 상단)
    2. Nut Bottom (넛 하단)
    3. Body Top (바디 상단)
    4. Body Bottom (바디 하단)
    """

    def __init__(self):
        """초기화"""
        self.corners = []
        self.instructions = [
            "1. Click NUT TOP",
            "2. Click NUT BOTTOM",
            "3. Click BODY TOP",
            "4. Click BODY BOTTOM"
        ]

    def mouse_callback(self, event, x, y, flags, param):
        """
        마우스 클릭 이벤트 처리

        Args:
            event: OpenCV 마우스 이벤트
            x, y: 클릭 좌표
            flags: 이벤트 플래그
            param: 추가 파라미터
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.corners) < 4:
                self.corners.append((float(x), float(y)))
                print(f"✅ Point {len(self.corners)} Set: ({x}, {y})")

    def run(self, frame):
        """
        보정 UI 실행

        Args:
            frame: 첫 프레임 이미지

        Returns:
            np.ndarray: 4개의 코너 좌표 (4x2 배열)
        """
        self.corners = []
        temp_frame = frame.copy()

        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', self.mouse_callback)

        while True:
            display = temp_frame.copy()

            # 현재 상태에 따른 메시지 표시
            idx = len(self.corners)
            if idx < 4:
                msg = self.instructions[idx]
                color = (0, 0, 255)  # 빨강
            else:
                msg = "Setup Done! Press SPACE to Start."
                color = (0, 255, 0)  # 초록

            # 메시지 출력
            cv2.putText(display, msg, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 선택된 포인트 표시
            for i, pt in enumerate(self.corners):
                cv2.circle(display, (int(pt[0]), int(pt[1])),
                           5, (0, 255, 255), -1)
                cv2.putText(display, str(i + 1),
                            (int(pt[0]) + 10, int(pt[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # 4개 포인트가 모두 선택되면 사각형 그리기
            if len(self.corners) == 4:
                pts = np.array(self.corners, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display, [pts], True, (0, 255, 0), 2)

            cv2.imshow('Calibration', display)

            key = cv2.waitKey(1) & 0xFF
            if key == 32 and len(self.corners) == 4:  # SPACE
                break
            elif key == 27:  # ESC
                print("❌ Calibration cancelled")
                exit()

        cv2.destroyWindow('Calibration')

        print(f"✅ Calibration Complete!")
        print(f"   Corners: {self.corners}")

        return np.array(self.corners, dtype=np.float32)

    def recalibrate(self, frame):
        """
        재보정 (실행 중 Space 키 입력 시)

        Args:
            frame: 현재 프레임

        Returns:
            np.ndarray: 새로운 4개의 코너 좌표
        """
        print("\n>>> PAUSED: Recalibrating...")
        new_corners = self.run(frame)
        print(">>> RESUMED with new coordinates")
        return new_corners