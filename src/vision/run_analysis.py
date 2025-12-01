"""
Video Analysis Pipeline

비디오 분석 파이프라인 실행 함수
"""

import cv2
import json
import os

from .bass_system import BassTabSystem
from .calibration import SimpleCalibrationUI
from ..utils.image_utils import resize_frame
from ..config.paths import DEFAULT_MODEL_PATH


def vision_analysis(video_path, output_json, model_path=None):
    """
    비디오에서 베이스 타블러처 추출

    Args:
        video_path: 입력 비디오 경로
        output_json: 출력 JSON 경로
        model_path: YOLO 모델 경로 (None이면 자동 탐색)

    Returns:
        bool: 성공 여부
    """
    print("\n" + "=" * 60)
    print("Starting Video Analysis Pipeline")
    print("=" * 60)

    # YOLO 모델 경로 설정
    if model_path is None:
        model_path = str(DEFAULT_MODEL_PATH)  # data/models/weights.pt 사용

    if not os.path.exists(model_path):
        print(f"❌ YOLO model not found: {model_path}")
        return False

    # 1. 비디오 열기
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False

    print(f"✅ Video opened: {video_path}")

    # 2. 첫 프레임 읽기
    ret, first_frame = cap.read()
    if not ret:
        print("❌ Could not read first frame")
        cap.release()
        return False

    frame = resize_frame(first_frame)

    # 3. 사용자 보정 (UI)
    print("\n>>> Starting Calibration UI...")
    ui = SimpleCalibrationUI()
    corners = ui.run(frame)

    # 4. 시스템 초기화
    system = BassTabSystem(model_path)

    # 5. Pre-Scan 실행
    print("\n>>> Running Pre-Scan...")
    system.run_pre_scan(str(video_path), corners)

    # 6. 비디오 재오픈 (Pre-Scan에서 사용했으므로)
    print("\n>>> Re-opening video for main analysis...")
    cap.release()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("❌ Could not re-open video")
        return False

    # 7. 추적 초기화
    ret, frame = cap.read()
    if not ret:
        print("❌ Could not read frame for tracking initialization")
        cap.release()
        return False

    frame = resize_frame(frame)
    system.init_tracking(frame, corners)

    # 8. 메인 분석 루프
    print("\n>>> Starting main analysis...")
    print("Press ESC to stop, SPACE to recalibrate\n")

    all_video_data = []
    frame_count = 0

    while cap.isOpened():
        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        ret, frame = cap.read()

        if not ret:
            break

        frame = resize_frame(frame)
        frame_count += 1

        # 코너 추적
        curr_corners = system.corner_tracker.update_tracking(frame)
        if curr_corners is None:
            print(f"\n⚠ Corner tracking lost at frame {frame_count}")
            break

        # 분석 수행
        json_data, visual_data = system.analyze(frame, curr_corners, current_time_sec)

        # JSON 데이터 저장
        if json_data:
            all_video_data.extend(json_data)

        # 터미널 출력
        system.print_tab(visual_data)

        # 화면 표시
        cv2.imshow('Bass Tab System', frame)

        # 키 입력 처리
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            print("\n\n>>> Analysis stopped by user")
            break
        elif key == 32:  # SPACE
            print("\n\n>>> PAUSED: Recalibrating...")
            new_corners = ui.recalibrate(frame)
            system.init_tracking(frame, new_corners)
            print(">>> RESUMED with new coordinates\n")

    # 9. JSON 저장
    print(f"\n\n>>> Saving results to {output_json}...")
    output_dir = os.path.dirname(output_json)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_video_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Analysis Complete!")
    print(f"   Total frames analyzed: {frame_count}")
    print(f"   Total notes detected: {len(all_video_data)}")
    print(f"   Output saved to: {output_json}")
    print("=" * 60)

    # 10. 정리
    cap.release()
    system.close()

    return True