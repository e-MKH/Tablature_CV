import argparse
import subprocess
import shutil
from pathlib import Path
import json
import numpy as np
import soundfile as sf
import librosa

# ============================================================
# 🔧 고정 세팅 (여기만 수정해서 다른 곡에도 재사용하면 됨)
# ============================================================
DEFAULT_INPUT = Path("data/input/drowning.mp4")  # 기본 입력 영상/오디오 경로
DEFAULT_BPM = 111                                # 기본 BPM

DEFAULT_OUTPUT_JSON = Path("data/output/drowning_audio_notes_librosa.json")
DEFAULT_WORK_DIR = Path("work_drowning")

DEFAULT_SAMPLE_RATE = 44100
DEFAULT_STEP_SIZE_MS = 10
DEFAULT_FMIN_HZ = 40.0
DEFAULT_FMAX_HZ = 300.0

# ============================================================
# 1. 파일 타입 판별용 확장자
# ============================================================
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def is_audio(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTS


# ============================================================
# 2. ffmpeg로 영상 -> full_audio.wav
# ============================================================
def extract_audio_from_video(
    input_video: Path,
    output_audio: Path,
    sr: int = 44100,
    mono: bool = True,
):
    """
    영상 파일에서 오디오만 추출해서 wav로 저장
    """
    output_audio.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
    ]

    if mono:
        cmd += ["-ac", "1"]
    else:
        cmd += ["-ac", "2"]

    cmd.append(str(output_audio))

    print("[ffmpeg] 명령어:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[ffmpeg] 추출 완료 → {output_audio}")


# ============================================================
# 3. Demucs로 베이스 stem 분리
# ============================================================
def separate_bass_with_demucs(full_audio: Path, work_dir: Path) -> Path:
    """
    full_audio에서 Demucs로 베이스 stem 추출
    - 리턴: work_dir/bass.wav (복사본)
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python",
        "-m",
        "demucs",
        "-n",
        "htdemucs",
        "--two-stems=bass",
        str(full_audio),
        "-o",
        str(work_dir),
    ]
    print("[demucs] 명령어:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    demucs_root = work_dir / "htdemucs"
    candidates = list(demucs_root.rglob("bass.wav"))
    if not candidates:
        raise FileNotFoundError(f"[demucs] bass stem을 찾을 수 없음: {demucs_root} 아래에 bass.wav 없음")

    bass_src = candidates[0]
    bass_dst = work_dir / "bass.wav"
    shutil.copy2(bass_src, bass_dst)

    print(f"[demucs] 베이스 stem 추출 완료 → {bass_src}")
    print(f"[demucs] 사용 편하게 work/bass.wav 로 복사 → {bass_dst}")
    return bass_dst


# ============================================================
# 4. librosa pyin 으로 bass.wav 피치 추출
# ============================================================
def run_pyin_on_bass(
    bass_wav: Path,
    step_size_ms: int = 10,
    fmin_hz: float = 40.0,
    fmax_hz: float = 300.0,
):
    """
    bass.wav 에 대해 librosa.pyin 실행
    return: sample_rate, time(s), freq(Hz), midi, confidence
    """
    audio, sr = sf.read(str(bass_wav))

    # 스테레오면 모노로
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    hop_length = int(sr * step_size_ms / 1000)  # 예: 10ms → 441
    frame_length = 2048

    print(
        f"[pyin] 분석 시작: {bass_wav}, sr={sr}, step={step_size_ms}ms, "
        f"hop_length={hop_length}, frame_length={frame_length}"
    )

    # librosa.pyin: f0, voiced_flag, voiced_prob 반환
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=fmin_hz,
        fmax=fmax_hz,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    # 시간축 (초)
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    # freq, conf 배열 정리
    freq = np.array(f0, dtype=float)          # Hz, unvoiced → NaN
    conf = np.array(voiced_prob, dtype=float) # 0~1, unvoiced → NaN

    # NaN 처리: freq NaN이면 conf=0으로
    invalid = np.isnan(freq)
    conf[invalid] = 0.0

    # Hz → MIDI
    midi = np.full_like(freq, np.nan, dtype=float)
    valid = ~invalid & (freq > 0)
    midi[valid] = 69 + 12 * np.log2(freq[valid] / 440.0)

    print(f"[pyin] frame 개수: {len(times)}")
    print(f"[pyin] 유효 freq 프레임 수: {np.sum(valid)}")
    return sr, times, freq, midi, conf


# ============================================================
# 5. frame → note로 묶기
# ============================================================
def frames_to_notes(
    time,
    freq,
    midi,
    conf,
    max_midi_diff: float = 0.5,
    min_duration_sec: float = 0.1,
):
    """
    frame들을 연속 구간으로 묶어서 note 리스트로 변환
    ⚠ librosa-pyin 버전에서는 conf(신뢰도)는 참고만 하고,
      실제로는 'freq가 NaN이 아닌 프레임'만 유효 프레임으로 사용.
    """
    notes = []
    n = len(time)
    current_start = None

    for i in range(n):
        # 유효한 프레임인지: freq가 NaN이면 pitched 아님
        if np.isnan(freq[i]):
            if current_start is not None:
                notes.append((current_start, i - 1))
                current_start = None
            continue

        if current_start is None:
            current_start = i
        else:
            if np.isnan(midi[i]) or np.isnan(midi[i - 1]):
                # midi가 NaN이면 그냥 끊어버리기
                notes.append((current_start, i - 1))
                current_start = None
                continue

            if abs(midi[i] - midi[i - 1]) > max_midi_diff:
                notes.append((current_start, i - 1))
                current_start = i

    if current_start is not None:
        notes.append((current_start, n - 1))

    note_dicts = []
    nid = 0
    for s, e in notes:
        onset = float(time[s])
        offset = float(time[e])
        dur = max(offset - onset, 0.0)

        if dur < min_duration_sec:
            continue

        freq_seg = freq[s:e + 1]
        midi_seg = midi[s:e + 1]
        conf_seg = conf[s:e + 1]

        freq_mean = float(np.nanmean(freq_seg))
        midi_mean = float(np.nanmean(midi_seg))
        conf_mean = float(np.nanmean(conf_seg)) if not np.all(np.isnan(conf_seg)) else 0.0

        note_dicts.append({
            "id": nid,
            "time_sec": onset,          # ← 여기 time_sec 사용
            "duration_sec": dur,
            "freq_hz": freq_mean,
            "midi": midi_mean,
            "confidence_mean": conf_mean,
        })
        nid += 1

    print(f"[notes] 생성된 노트 개수: {len(note_dicts)}")
    return note_dicts


# ============================================================
# 6. BPM 기준 beat 정보 + 양자화
# ============================================================
def add_beat_info_to_notes(notes, bpm: int):
    """
    각 note dict에 onset_beat, duration_beats 필드 추가
    beat = 4분음표 기준
    """
    for n in notes:
        time_sec = n["time_sec"]
        duration_sec = n["duration_sec"]

        onset_beat = time_sec * bpm / 60.0
        duration_beats = duration_sec * bpm / 60.0

        n["onset_beat"] = onset_beat
        n["duration_beats"] = duration_beats

    return notes


def quantize_duration_beats(duration_beats, allowed=None):
    """
    duration_beats 값을 주어진 후보 값들 중 가장 가까운 값으로 양자화
    예: allowed = [0.25, 0.5, 1.0, 1.5, 2.0]
    """
    if allowed is None:
        # 16분, 8분, 4분, 점4분, 2분
        allowed = [0.25, 0.5, 1.0, 1.5, 2.0]
    return min(allowed, key=lambda x: abs(x - duration_beats))


def quantize_notes_beats(notes):
    """
    각 노트의 onset_beat / duration_beats를 격자에 맞춰 양자화한 필드를 추가
    - onset_beat_q: 16분음표(0.25 beat) 단위로 스냅
    - duration_beats_q: [0.25, 0.5, 1.0, 1.5, 2.0] 중 가장 가까운 값
    """
    beat_grid = 0.25  # 16분음표 단위 격자

    for n in notes:
        ob = n["onset_beat"]
        db = n["duration_beats"]

        n["onset_beat_q"] = round(ob / beat_grid) * beat_grid
        n["duration_beats_q"] = quantize_duration_beats(db)

    return notes


# ============================================================
# 7. JSON 저장
# ============================================================
def save_audio_notes_json(
    output_json: Path,
    bpm: int,
    source_type: str,
    source_path: Path,
    full_audio: Path,
    bass_audio: Path,
    sample_rate: int,
    step_size_ms: int,
    fmin_hz: float,
    fmax_hz: float,
    notes,
):
    data = {
        "bpm": bpm,
        "source": {
            "type": source_type,
            "path": str(source_path),
        },
        "audio": {
            "full_path": str(full_audio),
            "bass_path": str(bass_audio),
            "sample_rate": sample_rate,
        },
        "pitch": {
            "backend": "librosa-pyin",
            "step_size_ms": step_size_ms,
            "fmin_hz": fmin_hz,
            "fmax_hz": fmax_hz,
        },
        "notes": notes,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[json] 노트 정보 저장 완료 → {output_json}")


# ============================================================
# 8. main
# ============================================================
def main():
    """
    기본적으로는 아무 인자 없이 실행해도 됨:

        python src/audio/audio_pipeline_librosa.py

    필요하면 옵션으로 override 가능:
        python src/audio/audio_pipeline_librosa.py --bpm 120
    """
    parser = argparse.ArgumentParser(
        description="베이스 탭 서비스를 위한 오디오 파이프라인 (ffmpeg + Demucs + librosa.pyin, 고정 세팅 지원)"
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help=f"입력 파일 경로 (기본: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--bpm",
        type=int,
        default=DEFAULT_BPM,
        help=f"곡의 BPM (기본: {DEFAULT_BPM})",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT_JSON),
        help=f"노트 정보를 저장할 JSON 경로 (기본: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--work-dir",
        default=str(DEFAULT_WORK_DIR),
        help=f"중간 파일 저장 디렉터리 (기본: {DEFAULT_WORK_DIR})",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"추출할 오디오 샘플레이트 (기본: {DEFAULT_SAMPLE_RATE})",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=DEFAULT_STEP_SIZE_MS,
        help=f"pyin step size (ms, 기본: {DEFAULT_STEP_SIZE_MS})",
    )
    parser.add_argument(
        "--fmin",
        type=float,
        default=DEFAULT_FMIN_HZ,
        help=f"pyin 최소 주파수 (Hz, 기본: {DEFAULT_FMIN_HZ})",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=DEFAULT_FMAX_HZ,
        help=f"pyin 최대 주파수 (Hz, 기본: {DEFAULT_FMAX_HZ})",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    work_dir = Path(args.work_dir)
    output_json = Path(args.output_json)

    work_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없음: {input_path}")

    full_audio_path = work_dir / "full_audio.wav"

    # 1) 영상/오디오 구분 + full_audio.wav 준비
    if is_video(input_path):
        print(f"[입력] 영상 파일로 인식됨: {input_path.name}")
        source_type = "video"
        extract_audio_from_video(
            input_video=input_path,
            output_audio=full_audio_path,
            sr=args.sample_rate,
            mono=True,
        )
    elif is_audio(input_path):
        print(f"[입력] 오디오 파일로 인식됨: {input_path.name}")
        source_type = "audio"
        full_audio_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, full_audio_path)
        print(f"[복사] {input_path} → {full_audio_path}")
    else:
        raise ValueError(
            f"지원하지 않는 확장자: {input_path.suffix} "
            f"(영상: {VIDEO_EXTS}, 오디오: {AUDIO_EXTS})"
        )

    # 2) Demucs로 베이스 stem 분리
    bass_audio_path = separate_bass_with_demucs(full_audio_path, work_dir)

    # 3) librosa pyin 으로 bass.wav 피치 분석
    sr, time, freq, midi, conf = run_pyin_on_bass(
        bass_wav=bass_audio_path,
        step_size_ms=args.step_size,
        fmin_hz=args.fmin,
        fmax_hz=args.fmax,
    )

    # 4) frame → note 리스트
    notes = frames_to_notes(time, freq, midi, conf)

    # 5) BPM 기반 beat 정보 추가
    notes = add_beat_info_to_notes(notes, bpm=args.bpm)

    # 6) beat 양자화
    notes = quantize_notes_beats(notes)

    # 7) JSON 저장
    save_audio_notes_json(
        output_json=output_json,
        bpm=args.bpm,
        source_type=source_type,
        source_path=input_path,
        full_audio=full_audio_path,
        bass_audio=bass_audio_path,
        sample_rate=sr,
        step_size_ms=args.step_size,
        fmin_hz=args.fmin,
        fmax_hz=args.fmax,
        notes=notes,
    )

    print()
    print("=== 오디오 파이프라인 (librosa 버전, 고정 세팅) 완료 ===")
    print(f"- 입력           : {input_path}")
    print(f"- BPM            : {args.bpm}")
    print(f"- full_audio     : {full_audio_path}")
    print(f"- bass_audio     : {bass_audio_path}")
    print(f"- note 개수      : {len(notes)}")
    print(f"- output_json    : {output_json}")


if __name__ == "__main__":
    main()