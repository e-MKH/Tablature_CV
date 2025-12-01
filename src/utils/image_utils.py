"""
Image Preprocessing Utilities

영상 전처리를 위한 유틸리티 함수들
"""

import cv2
import numpy as np
from ..config.vision_config import MAX_WIDTH


def resize_frame(frame, max_width=MAX_WIDTH):
    """
    프레임을 최대 너비에 맞춰 리사이징
    """
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_h = int(h * scale)
        # INTER_LINEAR로 변경 (프렛 같은 작은 객체 보존)
        return cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_LINEAR)
    return frame


def adjust_gamma(image, gamma=1.0):
    """
    감마 보정을 적용하여 밝기 조정

    Args:
        image: 입력 이미지
        gamma: 감마 값 (1.0보다 크면 밝게, 작으면 어둡게)

    Returns:
        np.ndarray: 감마 보정된 이미지

    Note:
        gamma < 1.0: 어두운 영역을 밝게
        gamma > 1.0: 밝은 영역을 어둡게
        gamma = 1.0: 변화 없음
    """
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ]).astype("uint8")

    return cv2.LUT(image, table)


def preprocess_image(frame, apply_clahe=True, apply_bilateral=False):
    """
    이미지 전처리 파이프라인

    손 인식 정확도를 높이기 위한 전처리 수행:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Bilateral Filter (선택적)

    Args:
        frame: 입력 프레임 (BGR)
        apply_clahe: CLAHE 적용 여부 (기본값: True)
        apply_bilateral: Bilateral Filter 적용 여부 (기본값: False)

    Returns:
        np.ndarray: 전처리된 프레임 (BGR)
    """
    processed = frame.copy()

    # CLAHE 적용 (대비 향상)
    if apply_clahe:
        # BGR을 LAB 색공간으로 변환
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # L 채널에만 CLAHE 적용
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # LAB 다시 합치고 BGR로 변환
        lab = cv2.merge([l, a, b])
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Bilateral Filter 적용 (노이즈 제거 + 엣지 보존)
    if apply_bilateral:
        processed = cv2.bilateralFilter(processed, d=9, sigmaColor=75, sigmaSpace=75)

    return processed


def enhance_for_detection(frame, gamma=1.5):
    """
    손 인식 실패 시 향상된 전처리 적용

    Args:
        frame: 입력 프레임
        gamma: 감마 보정값 (기본값: 1.5)

    Returns:
        np.ndarray: 향상된 프레임
    """
    # 감마 보정
    enhanced = adjust_gamma(frame, gamma)

    # CLAHE + Bilateral Filter 모두 적용
    enhanced = preprocess_image(enhanced, apply_clahe=True, apply_bilateral=True)

    return enhanced


def create_visualization_overlay(frame, alpha=0.3):
    """
    시각화를 위한 오버레이 생성

    Args:
        frame: 원본 프레임
        alpha: 투명도 (0.0 ~ 1.0)

    Returns:
        np.ndarray: 오버레이용 프레임
    """
    return frame.copy()


def blend_overlay(frame, overlay, alpha=0.3):
    """
    오버레이를 원본 프레임에 블렌딩

    Args:
        frame: 원본 프레임
        overlay: 오버레이 프레임
        alpha: 오버레이 투명도

    Returns:
        np.ndarray: 블렌딩된 프레임
    """
    return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)