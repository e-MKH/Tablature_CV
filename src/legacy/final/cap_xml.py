import json
import math
import os
import argparse
from music21 import stream, note, tempo, meter, instrument

# =========================================================
# [설정]
DEFAULT_VIDEO = "video_output2.json"
DEFAULT_AUDIO = "audio_output.json"
DEFAULT_OUTPUT = "output.musicxml"

TIME_WINDOW_SEC = 0.15 
PITCH_TOLERANCE = 1
STRING_OFFSETS = {'E': 28, 'A': 33, 'D': 38, 'G': 43}

# [핵심 설정] 양자화 그리드 (0.25 = 16분음표, 0.125 = 32분음표)
# 이 값 단위로 박자를 딱딱 맞춥니다.
QUANTIZE_GRID = 0.125 
# =========================================================

def load_json(path):
    if not os.path.exists(path): return None
    with open(path, 'r', encoding='utf-8') as f: return json.load(f)

def get_video_midi(string_name, fret_num):
    if string_name not in STRING_OFFSETS: return -1
    return STRING_OFFSETS[string_name] + fret_num

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

# [추가됨] 박자 양자화 함수 (오류 방지용)
def quantize(value, grid=QUANTIZE_GRID):
    """값을 grid 단위로 반올림 (예: 0.23 -> 0.25)"""
    return round(value / grid) * grid

def generate_musicxml(video_data, audio_data, output_path):
    video_data.sort(key=lambda x: x['time_sec'])
    
    sc = stream.Score()
    part = stream.Part()
    part.id = "BassPart"
    
    part.insert(0, instrument.ElectricBass())
    part.insert(0, meter.TimeSignature("4/4"))
    
    bpm = audio_data.get("bpm", 120)
    part.insert(0, tempo.MetronomeMark(number=bpm))

    last_known_pos = {'string': 'E', 'fret': 0}
    
    # 이전 노트가 끝난 지점 (Beat 단위)
    prev_end_beat = 0.0

    print(f">>> 변환 시작 (BPM: {bpm}, Grid: {QUANTIZE_GRID})...")

    # 오디오 노트 순회
    for i, note_info in enumerate(audio_data['notes']):
        onset_sec = note_info['onset_sec']
        
        # 1. 박자 계산 및 양자화 (Quantization)
        # ---------------------------------------------------------
        if "onset_beat" in note_info:
            raw_onset = float(note_info["onset_beat"])
            raw_dur = float(note_info.get("duration_beats", 1.0))
        else:
            raw_onset = onset_sec * (bpm / 60.0)
            raw_dur = note_info.get("duration_sec", 0.5) * (bpm / 60.0)

        # [핵심 수정] 모든 박자를 그리드에 맞춰 반올림
        onset_beat = quantize(raw_onset)
        duration_beat = quantize(raw_dur)
        
        # 최소 길이 보장 (0이 되면 안됨)
        if duration_beat < QUANTIZE_GRID:
            duration_beat = QUANTIZE_GRID

        audio_midi = int(round(note_info['midi']))

        # ---------------------------------------------------------
        # [A] 비디오 데이터 매칭
        # ---------------------------------------------------------
        candidates = [v for v in video_data if abs(v['time_sec'] - onset_sec) <= TIME_WINDOW_SEC]
        best_match = None
        min_pitch_diff = 999

        for v in candidates:
            v_midi = get_video_midi(v['string'], v['fret'])
            diff = abs(audio_midi - v_midi)
            if diff <= PITCH_TOLERANCE:
                if diff < min_pitch_diff:
                    min_pitch_diff = diff
                    best_match = v
                elif diff == min_pitch_diff:
                    if best_match and (not best_match.get('is_tracking', False)) and v.get('is_tracking', False):
                        best_match = v

        # ---------------------------------------------------------
        # [B] 데이터 결정
        # ---------------------------------------------------------
        if best_match:
            final_string = best_match['string']
            final_fret = best_match['fret']
            finger = best_match.get('finger', '')
            status = "TRACKED"
            last_known_pos = {'string': final_string, 'fret': final_fret}
        else:
            predicted = predict_best_position(audio_midi, last_known_pos)
            final_string = predicted['string']
            final_fret = predicted['fret']
            finger = ""
            status = "PREDICTED"
            last_known_pos = predicted

        # ---------------------------------------------------------
        # [C] 쉼표(Rest) 및 음표 추가 (오류 방지 로직 적용)
        # ---------------------------------------------------------
        
        # 1. 쉼표 추가: 이전 노트 끝과 현재 노트 시작 사이의 갭
        gap = onset_beat - prev_end_beat
        
        # 갭이 그리드보다 작으면 무시 (아주 미세한 차이는 붙임)
        if gap >= QUANTIZE_GRID:
            # 갭도 양자화 확인
            gap = quantize(gap)
            if gap > 0:
                r = note.Rest()
                r.quarterLength = gap
                part.append(r)
                prev_end_beat += gap # 쉼표만큼 시간 전진

        # 2. 시간 역전 방지 (정렬 문제나 양자화 오차로 겹칠 경우)
        if onset_beat < prev_end_beat:
            # 겹치면 현재 위치를 강제로 밈 (또는 무시 가능)
            onset_beat = prev_end_beat

        # 3. 음표 생성
        nt = note.Note(audio_midi)
        nt.quarterLength = duration_beat
        
        # 가사 입력
        tab_text = f"{final_string}:{final_fret}"
        nt.addLyric(tab_text)
        if finger: nt.addLyric(f"({finger})")
        if status == "PREDICTED":
            nt.style.color = 'red'
            nt.addLyric("?")

        part.append(nt)
        
        # 다음 노트의 시작 기준점 업데이트
        prev_end_beat = onset_beat + duration_beat

    sc.append(part)
    try:
        sc.write("musicxml", fp=output_path)
        print(f"✅ 생성 완료: {output_path}")
    except Exception as e:
        print(f"❌ MusicXML 저장 실패: {e}")
        print("💡 힌트: 박자가 너무 복잡해서 그럴 수 있습니다. QUANTIZE_GRID를 0.25로 늘려보세요.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="비디오 JSON 경로")
    parser.add_argument("--audio", default=DEFAULT_AUDIO, help="오디오 JSON 경로")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="저장할 MusicXML 경로")
    args = parser.parse_args()

    v_data = load_json(args.video)
    a_data = load_json(args.audio)

    if v_data and a_data:
        try:
            generate_musicxml(v_data, a_data, args.output)
        except Exception as e:
            print(f"❌ 심각한 오류: {e}")
    else:
        print("❌ JSON 파일을 찾을 수 없습니다.")