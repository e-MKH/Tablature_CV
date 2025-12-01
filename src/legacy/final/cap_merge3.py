import json
import math
import os

# =========================================================
# [설정]
VIDEO_JSON_FILE = "video_output2.json"
AUDIO_JSON_FILE = "audio_output.json"
FPS = 30.0
SEARCH_WINDOW = 3
PITCH_TOLERANCE = 1  # [수정] 2 -> 1 (더 엄격하게 검사)
STRING_OFFSETS = {'E': 28, 'A': 33, 'D': 38, 'G': 43}
# =========================================================

def onset_to_frame(onset_sec, fps):
    return int(round(onset_sec * fps))

def load_json(path):
    if not os.path.exists(path): return None
    with open(path, 'r', encoding='utf-8') as f: return json.load(f)

def get_video_midi(string_name, fret_num):
    if string_name not in STRING_OFFSETS: return -1
    return STRING_OFFSETS[string_name] + fret_num

def build_video_lookup(video_data, fps):
    lookup = {}
    for item in video_data:
        f_idx = onset_to_frame(item['time_sec'], fps)
        if f_idx not in lookup: lookup[f_idx] = []
        lookup[f_idx].append(item)
    return lookup

def get_possible_positions(midi_note):
    possibilities = []
    for s_name, s_midi in STRING_OFFSETS.items():
        fret = midi_note - s_midi
        if 0 <= fret <= 22:
            possibilities.append({'string': s_name, 'fret': fret})
    return possibilities

def predict_best_position(target_midi, last_pos):
    candidates = get_possible_positions(target_midi)
    if not candidates: return {'string': '?', 'fret': '?'} 
    
    if not last_pos:
        candidates.sort(key=lambda x: x['fret'])
        return candidates[0]

    best_candidate = None
    min_dist = 999
    
    str_map = {'E': 0, 'A': 1, 'D': 2, 'G': 3}
    last_s_idx = str_map.get(last_pos['string'], 0)
    
    for cand in candidates:
        curr_s_idx = str_map.get(cand['string'], 0)
        dist = abs(cand['fret'] - last_pos['fret']) + abs(curr_s_idx - last_s_idx) * 5
        if dist < min_dist:
            min_dist = dist
            best_candidate = cand
            
    return best_candidate

def merge_data(audio_data, video_lookup, fps):
    merged_tab = []
    last_known_pos = {'string': 'E', 'fret': 0} 

    print(f">>> 병합 시작 (Tolerance: ±{PITCH_TOLERANCE})")

    for note in audio_data['notes']:
        onset = note['onset_sec']
        audio_midi = int(round(note['midi']))
        center_frame = onset_to_frame(onset, fps)
        
        # 1. 영상 데이터 후보군 수집
        candidates = []
        for f_idx in range(center_frame - SEARCH_WINDOW, center_frame + SEARCH_WINDOW + 1):
            if f_idx in video_lookup:
                candidates.extend(video_lookup[f_idx])

        # 2. [핵심 수정] 최적의 매칭 찾기 (First Match -> Best Match)
        best_match = None
        min_pitch_diff = 999 # 음정 차이 초기값 (무한대)

        # 후보군 전체를 돌면서 "가장 오차가 적은 놈"을 찾음
        for v in candidates:
            # 추적 실패(False) 데이터는 우선순위를 낮추기 위해 페널티 부여 가능
            # 여기서는 단순히 pitch 차이만 비교
            
            v_midi = get_video_midi(v['string'], v['fret'])
            diff = abs(audio_midi - v_midi)

            # 허용 오차 내에 들어오는지 확인
            if diff <= PITCH_TOLERANCE:
                # 기존에 찾은 놈보다 더 오차가 작으면 교체
                # (만약 오차가 같다면, '추적 성공(True)'인 놈을 우선)
                if diff < min_pitch_diff:
                    min_pitch_diff = diff
                    best_match = v
                elif diff == min_pitch_diff:
                    # 오차가 같을 때: 현재 best가 False고, 새로운 v가 True면 교체
                    if best_match and (not best_match['is_tracking']) and v['is_tracking']:
                        best_match = v

        # 3. 결과 결정
        final_string = "?"
        final_fret = "?"
        status = "UNKNOWN"

        if best_match:
            final_string = best_match['string']
            final_fret = best_match['fret']
            status = "TRACKED" if best_match['is_tracking'] else "ESTIMATED"
            last_known_pos = {'string': final_string, 'fret': final_fret}
        else:
            predicted = predict_best_position(audio_midi, last_known_pos)
            final_string = predicted['string']
            final_fret = predicted['fret']
            status = "PREDICTED"
            last_known_pos = predicted

        merged_tab.append({
            "time": onset,
            "string": final_string,
            "fret": final_fret,
            "midi": audio_midi,
            "status": status
        })
            
    return merged_tab

def draw_terminal_tab(tab_data):
    if not tab_data: return
    RES = 10; CPL = 80
    last_t = tab_data[-1]['time']
    slots = int(math.ceil(last_t * RES)) + 10
    strs = {k: ["-"] * slots for k in ["G", "D", "A", "E"]}
    
    for n in tab_data:
        if n['string'] not in strs: continue
        idx = int(n['time'] * RES)
        fs = str(n['fret'])
        for i, c in enumerate(fs):
            if idx+i < slots: strs[n['string']][idx+i] = c

    print("\n" + "="*CPL)
    print("🎸 FINAL TAB (Logic Fixed)")
    print("="*CPL)
    curr = 0
    while curr < slots:
        end = min(curr + CPL, slots)
        print(f"\n[ {curr/RES:.1f}s ~ {end/RES:.1f}s ]")
        for s in ["G", "D", "A", "E"]: print(f"{s} |{''.join(strs[s][curr:end])}|")
        curr += CPL
    print("\n✅ Done.")

if __name__ == "__main__":
    v_data = load_json(VIDEO_JSON_FILE); a_data = load_json(AUDIO_JSON_FILE)
    if v_data and a_data:
        v_lookup = build_video_lookup(v_data, FPS)
        res = merge_data(a_data, v_lookup, FPS)
        draw_terminal_tab(res)
        with open("final_tab_sheet.json", "w") as f: json.dump(res, f, indent=4)
        print("📁 Saved to final_tab_sheet.json")