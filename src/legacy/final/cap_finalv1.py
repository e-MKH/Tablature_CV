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
MAX_WIDTH = 1280       # 리사이징 최대 폭
VIDEO_PATH = r"C:\Users\asdxo\Desktop\capstone\ex_main7.mp4" # 영상 경로
# =========================================================

# --- [유틸] 리사이징 함수 ---
def resize_frame_fn(frame, max_w=MAX_WIDTH):
    h, w = frame.shape[:2]
    if w > max_w:
        scale = max_w / w
        new_h = int(h * scale)
        return cv2.resize(frame, (max_w, new_h))
    return frame

# --- [유틸] 감마 보정 함수 ---
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# --- [1] 필터링 클래스 ---
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

# --- [2] UI 클래스 ---
class SimpleCalibrationUI:
    def __init__(self):
        self.corners = [] 
        self.instructions = [
            "1. Click NUT TOP", "2. Click NUT BOTTOM",
            "3. Click BODY TOP", "4. Click BODY BOTTOM"
        ]

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.corners) < 4:
                self.corners.append((float(x), float(y)))
                print(f"✅ Point {len(self.corners)} Set")

    def run(self, frame):
        self.corners = [] 
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
                msg = "Setup Done! Press SPACE to Start."
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

# --- [3] 메인 시스템 클래스 ---
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
        
        # [핵심] 추적 실패 보완용 변수
        self.last_valid_fingers = [] 
        self.missing_frames_count = 0
        self.MAX_MISSING_TOLERANCE = 30 # 약 1초 정도 유지

    def init_tracking(self, frame, corners):
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.corners = corners.reshape(-1, 1, 2)

    def update_tracking(self, frame):
        if self.corners is None: return None
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, status, error = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, frame_gray, self.corners, None, **self.lk_params
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
        return base_pitches.get(string_name, 0) + fret_number

    def _perform_scan_pass(self, cap, targets, duration_frames, initial_corners):
        collected_ratios = []
        prev_gray = None
        curr_corners = initial_corners.reshape(-1, 1, 2)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx > targets[-1] + duration_frames: break
            frame = resize_frame_fn(frame)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, frame_gray, curr_corners, None, **self.lk_params)
                curr_corners = p1
            prev_gray = frame_gray.copy()
            
            is_scanning = False
            for t in targets:
                if t <= frame_idx < t + duration_frames:
                    is_scanning = True; break
            
            if is_scanning:
                corners_flat = curr_corners.reshape(4, 2)
                nut_center, unit_vec, total_len = self.get_geometry(corners_flat)
                results = self.yolo_model(frame, verbose=False)
                for r in results:
                    for box in r.boxes:
                        if 'fret' in self.yolo_model.names[int(box.cls[0])].lower():
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cx, cy = (x1+x2)/2, (y1+y2)/2
                            ratio = self.project_ratio(np.array([cx, cy]), nut_center, unit_vec, total_len)
                            if 0 < ratio < 1.1: collected_ratios.append(ratio)
                
                if frame_idx % 5 == 0:
                    display = frame.copy()
                    cv2.putText(display, "Scanning...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Bass Tab System', display)
                    cv2.waitKey(1)
            frame_idx += 1
        return collected_ratios

    def run_pre_scan(self, cap, initial_corners):
        print(">>> Starting Pre-Scan Sequence...")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        scan_duration = int(fps * 1.0) 

        targets = [0, int(total_frames*0.25), int(total_frames*0.50), int(total_frames*0.75)]
        ratios = self._perform_scan_pass(cap, targets, scan_duration, initial_corners)

        if not ratios:
            print("⚠ Use Math Fallback")
            k = 2 ** (-1/12); scale_len = 1.0 / (1 - k**22)
            self.learned_fret_ratios = [scale_len * (1 - k**i) for i in range(1, 22)]
        else:
            ratios.sort()
            merged_ratios = []; curr_group = [ratios[0]]
            for r in ratios[1:]:
                if r - np.mean(curr_group) < 0.013: curr_group.append(r)
                else:
                    merged_ratios.append(np.mean(curr_group))
                    curr_group = [r]
            merged_ratios.append(np.mean(curr_group))
            self.learned_fret_ratios = merged_ratios
        print(f"✅ Map Created! {len(self.learned_fret_ratios)} frets.")

    def get_strings_lines(self, corners):
        lt, lb, rt, rb = corners
        left_vec = lb - lt; right_vec = rb - rt
        MARGIN = 0.12
        string_lines = []
        for i in range(4):
            r = MARGIN + (i * (1 - 2*MARGIN) / 3.0)
            start = lt + left_vec * r; end = rt + right_vec * r
            string_lines.append(((int(start[0]), int(start[1])), (int(end[0]), int(end[1]))))
        divider_lines = []
        for i in range(3):
            r1 = MARGIN + (i * (1 - 2*MARGIN) / 3.0)
            r2 = MARGIN + ((i+1) * (1 - 2*MARGIN) / 3.0)
            mid = (r1 + r2) / 2
            start = lt + left_vec * mid; end = rt + right_vec * mid
            divider_lines.append(((int(start[0]), int(start[1])), (int(end[0]), int(end[1]))))
        return string_lines, divider_lines

    def analyze(self, frame, corners, current_time_sec):
        corners_flat = corners.reshape(4, 2)
        nut_center, unit_vec, total_len = self.get_geometry(corners_flat)
        str_lines, div_lines = self.get_strings_lines(corners_flat)

        # 전처리
        enhanced_frame = self.preprocess_image(frame)
        mp_results = self.hands.process(cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB))
        if not mp_results.multi_hand_landmarks:
             mp_results = self.hands.process(cv2.cvtColor(adjust_gamma(enhanced_frame, 1.5), cv2.COLOR_BGR2RGB))

        # 시각화
        overlay = frame.copy()
        for p1, p2 in div_lines: cv2.line(overlay, p1, p2, (0, 0, 255), 2)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        for p1, p2 in str_lines: cv2.line(frame, p1, p2, (0, 255, 0), 2)
        for i, ratio in enumerate(self.learned_fret_ratios):
            pos = nut_center + unit_vec * (total_len * ratio)
            cv2.circle(frame, (int(pos[0]), int(pos[1])), 3, (255, 255, 0), -1)

        detected_data_list = []
        visual_data = []
        is_tracking_success = False

        if mp_results.multi_hand_landmarks:
            is_tracking_success = True
            self.missing_frames_count = 0 

            for hand_lms, handedness in zip(mp_results.multi_hand_landmarks, mp_results.multi_handedness):
                if handedness.classification[0].label != TARGET_HAND: continue 
                
                h, w, c = frame.shape
                finger_names = ["I", "M", "R", "P"]
                for i, idx in enumerate([8, 12, 16, 20]):
                    raw_cx, raw_cy = int(hand_lms.landmark[idx].x * w), int(hand_lms.landmark[idx].y * h)
                    cx, cy = self.apply_one_euro(finger_names[i], (raw_cx, raw_cy))
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)
                    
                    det_str = -1; min_d = 999
                    for si, ((sx1, sy1), (sx2, sy2)) in enumerate(str_lines):
                        if sx2==sx1: continue
                        ly = sy1 + (sy2-sy1)*((cx-sx1)/(sx2-sx1))
                        if abs(cy-ly) < 25 and abs(cy-ly) < min_d:
                            min_d = abs(cy-ly); det_str = si
                    
                    if det_str != -1 and self.learned_fret_ratios:
                        f_ratio = self.project_ratio(np.array([cx, cy]), nut_center, unit_vec, total_len)
                        det_fret = -1
                        TOLERANCE = 0.02 # 오차 범위 (2%)
                        
                        for fi, fr in enumerate(self.learned_fret_ratios):
                            if f_ratio <= (fr + TOLERANCE):
                                det_fret = fi + 1; break
                        if f_ratio > (self.learned_fret_ratios[-1] + TOLERANCE):
                             det_fret = len(self.learned_fret_ratios)

                        note_data = {
                            "time_sec": round(current_time_sec, 3),
                            "mid": self.get_midi_note(self.string_names[det_str], det_fret),
                            "string": self.string_names[det_str],
                            "fret": det_fret,
                            "finger": finger_names[i],
                            "is_tracking": True # [중요] 실제 추적됨
                        }
                        detected_data_list.append(note_data)
                        visual_data.append(note_data)
                        cv2.putText(frame, f"{self.string_names[det_str]}/{det_fret}", (cx, cy-15), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            if detected_data_list:
                self.last_valid_fingers = detected_data_list
                
        if not is_tracking_success:
            self.missing_frames_count += 1
            if self.missing_frames_count <= self.MAX_MISSING_TOLERANCE and self.last_valid_fingers:
                cv2.putText(frame, "MEMORY (Lost)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
                for item in self.last_valid_fingers:
                    new_item = item.copy()
                    new_item['time_sec'] = round(current_time_sec, 3)
                    new_item['is_tracking'] = False # [중요] 추정값
                    detected_data_list.append(new_item)

        return detected_data_list, visual_data

    def print_tab(self, fingers):
        if not fingers: return
        output = " | ".join([f"[{f['finger']}] {f['string']} {f['fret']}" for f in fingers])
        print(f"\r▶ {output}                ", end="")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'weights1.pt')
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): 
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    ret, first_frame = cap.read()
    if not ret: return
    frame = resize_frame_fn(first_frame)

    ui = SimpleCalibrationUI()
    corners = ui.run(frame) 

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
        
        if json_data: all_video_data.extend(json_data)
        
        system.print_tab(visual_data)
        cv2.imshow('Bass Tab System', frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == 27: break
        elif key == 32: # SPACE: 재보정
            print("\n>>> PAUSED: Recalibrating...")
            new_corners = ui.run(frame) 
            system.init_tracking(frame, new_corners) 
            print(">>> RESUMED with new coordinates")

    output_path = os.path.join(current_dir, 'video_output2.json')
    with open(output_path, 'w') as f:
        json.dump(all_video_data, f, indent=4)
    
    print(f"\n✅ Analysis Complete! Data saved to: {output_path}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()