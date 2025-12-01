import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
import os
import math
import time
import json 

# =========================================================
# [설정] 사용자 환경에 맞게 수정하세요
# =========================================================
TARGET_HAND = "Right"  # 미디어파이프가 인식하는 손
MAX_WIDTH = 1280       # 리사이징 최대 폭 (속도 최적화)
VIDEO_PATH = r"C:\Users\asdxo\Desktop\capstone\ex_main7.mp4" # 분석할 영상 경로
# =========================================================

# --- [유틸] 리사이징 함수 ---
def resize_frame_fn(frame, max_w=MAX_WIDTH):
    h, w = frame.shape[:2]
    if w > max_w:
        scale = max_w / w
        new_h = int(h * scale)
        return cv2.resize(frame, (max_w, new_h))
    return frame

# --- [유틸] 감마 보정 함수 (재탐지용) ---
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# --- [1] OneEuroFilter (떨림 보정) ---
class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def filter(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev 

        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class PointFilter:
    def __init__(self):
        self.first_time = True
        self.x_filter = None
        self.y_filter = None
        self.config = (1.0, 0.05, 1.0) 

    def process(self, t, pos):
        x, y = pos
        if self.first_time:
            self.x_filter = OneEuroFilter(t, x, *self.config)
            self.y_filter = OneEuroFilter(t, y, *self.config)
            self.first_time = False
            return x, y
        
        x_hat = self.x_filter.filter(t, x)
        y_hat = self.y_filter.filter(t, y)
        return x_hat, y_hat

# --- [2] UI (보정 시스템) ---
class SimpleCalibrationUI:
    def __init__(self):
        self.corners = [] 
        self.instructions = [
            "1. Click NUT TOP",
            "2. Click NUT BOTTOM",
            "3. Click BODY TOP",
            "4. Click BODY BOTTOM"
        ]

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.corners) < 4:
                self.corners.append((float(x), float(y)))
                print(f"✅ Point {len(self.corners)} Set")

    def run(self, frame):
        self.corners = [] # 재실행 시 초기화
        temp_frame = frame.copy()
        cv2.namedWindow('Calibration')
        cv2.setMouseCallback('Calibration', self.mouse_callback)

        while True:
            display = temp_frame.copy()
            idx = len(self.corners)
            if idx < 4:
                msg = self.instructions[idx]
                color = (0, 0, 255)
            else:
                msg = "Setup Done! Press SPACE to Start/Resume."
                color = (0, 255, 0)

            cv2.putText(display, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            for i, pt in enumerate(self.corners):
                cv2.circle(display, (int(pt[0]), int(pt[1])), 5, (0, 255, 255), -1)
            
            if len(self.corners) == 4:
                pts = np.array(self.corners, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display, [pts], True, (0, 255, 0), 2)

            cv2.imshow('Calibration', display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32 and len(self.corners) == 4: break
            elif key == 27: exit()

        cv2.destroyWindow('Calibration')
        return np.array(self.corners, dtype=np.float32)

# --- [3] 메인 시스템 (분석 로직) ---
class BassTabSystem:
    def __init__(self, model_path):
        print(f"Loading YOLO from {model_path}...")
        self.yolo_model = YOLO(model_path)
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,           
            min_detection_confidence=0.3, 
            min_tracking_confidence=0.3
        )
        self.string_names = ["E", "A", "D", "G"]
        
        self.prev_gray = None
        self.corners = None 
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        
        self.learned_fret_ratios = [] 
        self.filters = {} 
        self.start_time = time.time()
        
        self.last_valid_fingers = [] 
        self.missing_frames_count = 0
        
        # [설정] 메모리 유지 프레임 (30프레임 = 약 1초)
        self.MAX_MISSING_TOLERANCE = 30 

    def init_tracking(self, frame, corners):
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.corners = corners.reshape(-1, 1, 2)

    def update_tracking(self, frame):
        if self.corners is None: return None
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p0 = self.corners
        p1, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, frame_gray, p0, None, **self.lk_params
        )
        self.corners = p1
        self.prev_gray = frame_gray.copy()
        return p1.reshape(4, 2)

    def get_geometry(self, corners):
        lt, lb, rt, rb = corners
        nut_center = (lt + lb) / 2
        body_center = (rt + rb) / 2
        neck_vector = body_center - nut_center
        neck_len = np.linalg.norm(neck_vector)
        if neck_len == 0: neck_len = 1
        unit_vec = neck_vector / neck_len
        return nut_center, unit_vec, neck_len

    def project_ratio(self, point, origin, unit_vec, total_len):
        vec = point - origin
        dist = np.dot(vec, unit_vec)
        return dist / total_len

    def get_midi_note(self, string_name, fret_number):
        base_pitches = {'E': 28, 'A': 33, 'D': 38, 'G': 43}
        if string_name in base_pitches:
            return base_pitches[string_name] + fret_number
        return -1

    def _perform_scan_pass(self, cap, targets, duration_frames, initial_corners):
        collected_ratios = []
        prev_gray = None
        curr_corners = initial_corners.reshape(-1, 1, 2)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = resize_frame_fn(frame)

            if frame_idx > targets[-1] + duration_frames: break 
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, curr_corners, None, **self.lk_params)
                curr_corners = p1
            prev_gray = frame_gray.copy()
            
            is_scanning = False
            for t in targets:
                if t <= frame_idx < t + duration_frames:
                    is_scanning = True
                    break
            
            if is_scanning:
                corners_flat = curr_corners.reshape(4, 2)
                nut_center, unit_vec, total_len = self.get_geometry(corners_flat)
                results = self.yolo_model(frame, verbose=False)
                
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        if 'fret' in self.yolo_model.names[cls_id].lower():
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cx, cy = (x1+x2)/2, (y1+y2)/2
                            ratio = self.project_ratio(np.array([cx, cy]), nut_center, unit_vec, total_len)
                            if 0 < ratio < 1.1: collected_ratios.append(ratio)
                
                # [유지] 스캐닝 퍼센티지 표시
                if frame_idx % 5 == 0:
                    display = frame.copy()
                    cv2.putText(display, f"Scanning... {int(frame_idx/total_frames*100)}%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Bass Tab System', display)
                    cv2.waitKey(1)
            frame_idx += 1
        return collected_ratios

    def run_pre_scan(self, cap, initial_corners):
        print(">>> Starting Pre-Scan Sequence...")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        scan_duration_frames = int(fps * 1.0) 

        targets_phase1 = [0, int(total_frames*0.25), int(total_frames*0.50), int(total_frames*0.75)]
        print(f"▶ Phase 1 Scan: {targets_phase1}")
        ratios = self._perform_scan_pass(cap, targets_phase1, scan_duration_frames, initial_corners)

        if not ratios:
            print("⚠ Phase 1 Failed. Attempting Phase 2...")
            targets_phase2 = [int(total_frames*0.20), int(total_frames*0.40), int(total_frames*0.60), int(total_frames*0.80)]
            print(f"▶ Phase 2 Scan: {targets_phase2}")
            ratios = self._perform_scan_pass(cap, targets_phase2, scan_duration_frames, initial_corners)

        if not ratios:
            print("❌ All Scans Failed. Using Math Fallback.")
            k = 2 ** (-1/12)
            scale_len = 1.0 / (1 - k**22)
            fallback = [scale_len * (1 - k**i) for i in range(1, 22)]
            self.learned_fret_ratios = fallback
        else:
            ratios.sort()
            merged_ratios = []
            curr_group = [ratios[0]]
            for r in ratios[1:]:
                # [설정값] 1.3% 이내 거리면 같은 프렛으로 인식 (지도 작성용)
                if r - np.mean(curr_group) < 0.013: curr_group.append(r)
                else:
                    merged_ratios.append(np.mean(curr_group))
                    curr_group = [r]
            merged_ratios.append(np.mean(curr_group))
            self.learned_fret_ratios = merged_ratios
            print(f"✅ Map Created! {len(merged_ratios)} frets identified.")

    def get_strings_lines(self, corners):
        lt, lb, rt, rb = corners
        left_vec = lb - lt
        right_vec = rb - rt
        MARGIN = 0.12
        string_lines = []
        for i in range(4):
            r = MARGIN + (i * (1 - 2*MARGIN) / 3.0)
            start = lt + left_vec * r
            end = rt + right_vec * r
            string_lines.append(((int(start[0]), int(start[1])), (int(end[0]), int(end[1]))))
        divider_lines = []
        for i in range(3):
            r1 = MARGIN + (i * (1 - 2*MARGIN) / 3.0)
            r2 = MARGIN + ((i+1) * (1 - 2*MARGIN) / 3.0)
            mid = (r1 + r2) / 2
            start = lt + left_vec * mid
            end = rt + right_vec * mid
            divider_lines.append(((int(start[0]), int(start[1])), (int(end[0]), int(end[1]))))
        return string_lines, divider_lines

    def preprocess_image(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def apply_one_euro(self, finger_name, curr_pos):
        curr_time = time.time() - self.start_time
        if finger_name not in self.filters:
            self.filters[finger_name] = PointFilter()
        filtered_x, filtered_y = self.filters[finger_name].process(curr_time, curr_pos)
        return (int(filtered_x), int(filtered_y))

    def analyze(self, frame, corners, current_time_sec):
        corners_flat = corners.reshape(4, 2)
        nut_center, unit_vec, total_len = self.get_geometry(corners_flat)
        str_lines, div_lines = self.get_strings_lines(corners_flat)

        # 재탐지 로직
        enhanced_frame = self.preprocess_image(frame)
        mp_results = self.hands.process(cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB))
        
        detected_valid_hand = False
        
        if not mp_results.multi_hand_landmarks:
             bright_frame = adjust_gamma(enhanced_frame, gamma=1.5)
             mp_results = self.hands.process(cv2.cvtColor(bright_frame, cv2.COLOR_BGR2RGB))

        # 시각화
        overlay = frame.copy()
        for p1, p2 in div_lines: cv2.line(overlay, p1, p2, (0, 0, 255), 2)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        for p1, p2 in str_lines: cv2.line(frame, p1, p2, (0, 255, 0), 2)
        
        for i, ratio in enumerate(self.learned_fret_ratios):
            pos = nut_center + unit_vec * (total_len * ratio)
            px, py = int(pos[0]), int(pos[1])
            cv2.circle(frame, (px, py), 3, (255, 255, 0), -1)
            cv2.putText(frame, str(i+1), (px, py-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        detected_data_list = []
        detected_fingers_visual = []

        if mp_results.multi_hand_landmarks:
            detected_valid_hand = True
            self.missing_frames_count = 0 

            for hand_lms, handedness in zip(mp_results.multi_hand_landmarks, mp_results.multi_handedness):
                if handedness.classification[0].label != TARGET_HAND: continue 
                
                h, w, c = frame.shape
                finger_names = ["I", "M", "R", "P"]
                for i, idx in enumerate([8, 12, 16, 20]):
                    raw_cx, raw_cy = int(hand_lms.landmark[idx].x * w), int(hand_lms.landmark[idx].y * h)
                    
                    cx, cy = self.apply_one_euro(finger_names[i], (raw_cx, raw_cy))
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
                    
                    det_str = -1
                    min_d = 999
                    for si, ((sx1, sy1), (sx2, sy2)) in enumerate(str_lines):
                        if sx2-sx1 == 0: continue
                        r = (cx-sx1)/(sx2-sx1)
                        ly = sy1 + (sy2-sy1)*r
                        if abs(cy-ly) < 25 and abs(cy-ly) < min_d:
                            min_d = abs(cy-ly)
                            det_str = si
                    
                    if det_str != -1 and self.learned_fret_ratios:
                        f_ratio = self.project_ratio(np.array([cx, cy]), nut_center, unit_vec, total_len)
                        det_fret = -1
                        
                        # [핵심 수정] 동적 오차 범위 (Dynamic Tolerance)
                        # 현재 프렛과 다음 프렛 사이 거리의 20%를 오차범위로 설정
                        for fi, fr in enumerate(self.learned_fret_ratios):
                            # 다음 프렛 위치 가져오기 (마지막 프렛이면 대략적 3%로 계산)
                            if fi < len(self.learned_fret_ratios) - 1:
                                next_fr = self.learned_fret_ratios[fi+1]
                                fret_width = next_fr - fr
                            else:
                                fret_width = 0.03 # 대략적인 값

                            dynamic_tol = fret_width * 0.20 # 20%만 더 봐줌 (겹침 방지)

                            if f_ratio <= (fr + dynamic_tol):
                                det_fret = fi + 1
                                break
                        
                        # 마지막 프렛을 넘었을 경우
                        if f_ratio > (self.learned_fret_ratios[-1] + 0.01):
                             det_fret = len(self.learned_fret_ratios)

                        # det_fret이 결정되었을 때만 데이터 저장
                        if det_fret != -1:
                            midi_val = self.get_midi_note(self.string_names[det_str], det_fret)
                            
                            note_data = {
                                "time_sec": round(current_time_sec, 3),
                                "duration_sec": None, 
                                "mid": midi_val,
                                "string": self.string_names[det_str],
                                "fret": det_fret,
                                "finger": finger_names[i],
                                "is_tracking": True # [추가] 실제 추적 중
                            }
                            detected_data_list.append(note_data)

                            detected_fingers_visual.append({'f': finger_names[i], 's': self.string_names[det_str], 'n': det_fret})
                            cv2.putText(frame, f"{self.string_names[det_str]}/{det_fret}", (cx, cy-15), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            if detected_data_list:
                self.last_valid_fingers = detected_data_list
                
        # 메모리 기능
        if not detected_valid_hand:
            self.missing_frames_count += 1
            if self.missing_frames_count <= self.MAX_MISSING_TOLERANCE and self.last_valid_fingers:
                cv2.putText(frame, "Memory", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                recalled_data = []
                for item in self.last_valid_fingers:
                    new_item = item.copy()
                    new_item['time_sec'] = round(current_time_sec, 3)
                    new_item['is_tracking'] = False # [추가] 추정치(Memory)임
                    recalled_data.append(new_item)
                detected_data_list = recalled_data

        return detected_data_list, detected_fingers_visual

    def print_tab(self, fingers):
        if not fingers: return
        output = " | ".join([f"[{f['f']}] {f['s']} {f['n']}" for f in fingers])
        print(f"\r▶ {output}                ", end="")

# --- [4] 메인 실행 함수 ---
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'weights1.pt')
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): 
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    ret, first_frame = cap.read()
    if not ret: return
    
    first_frame = resize_frame_fn(first_frame)

    ui = SimpleCalibrationUI()
    corners = ui.run(first_frame) 

    system = BassTabSystem(model_path)
    system.run_pre_scan(cap, corners)
    
    print(">>> Re-opening video for main preprocessing...")
    cap.release()
    cap = cv2.VideoCapture(VIDEO_PATH) 
    
    if not cap.isOpened(): return

    ret, frame = cap.read()
    if not ret: return
    frame = resize_frame_fn(frame)

    system.init_tracking(frame, corners) 

    all_video_data = []

    while cap.isOpened():
        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        ret, frame = cap.read()
        if not ret: break
        
        frame = resize_frame_fn(frame)

        curr_corners = system.update_tracking(frame)
        if curr_corners is None: break 
        
        json_data, visual_data = system.analyze(frame, curr_corners, current_time_sec)
        
        if json_data:
            all_video_data.extend(json_data)
        
        system.print_tab(visual_data)

        cv2.imshow('Bass Tab System', frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == 27: # ESC
            break
        elif key == 32: # SPACE
            print("\n>>> PAUSED: Recalibrating...")
            new_corners = ui.run(frame) 
            system.init_tracking(frame, new_corners) 
            print(">>> RESUMED with new coordinates")

    output_path = os.path.join(current_dir, 'video_output3.json')
    with open(output_path, 'w') as f:
        json.dump(all_video_data, f, indent=4)
    
    print(f"\n✅ Analysis Complete! Data saved to: {output_path}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()