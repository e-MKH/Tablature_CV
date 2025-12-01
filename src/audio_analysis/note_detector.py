from pathlib import Path
from typing import List, Dict, Any
import json
import numpy as np


def detect_notes(
        times: np.ndarray,
        frequencies: np.ndarray,
        midi_notes: np.ndarray,
        confidences: np.ndarray,
        max_midi_diff: float = 0.5,
        min_duration_sec: float = 0.1,
) -> List[Dict[str, Any]]:
    """
    프레임 단위 피치 분석 결과를 음표 단위로 변환합니다.

    연속된 유효 프레임을 하나의 음표로 묶되, MIDI 값이 급격히 변하면
    다른 음표로 분리합니다. 너무 짧은 음표는 필터링됩니다.

    Parameters
    ----------
    times : np.ndarray
        각 프레임의 시간 (초 단위)
    frequencies : np.ndarray
        각 프레임의 주파수 (Hz). NaN은 유효하지 않은 프레임
    midi_notes : np.ndarray
        각 프레임의 MIDI 노트 번호. NaN은 유효하지 않은 프레임
    confidences : np.ndarray
        각 프레임의 신뢰도 (0~1)
    max_midi_diff : float, default=0.5
        연속 프레임 간 MIDI 차이 임계값. 이보다 크면 다른 음표로 분리
    min_duration_sec : float, default=0.1
        최소 음표 지속 시간 (초). 이보다 짧은 음표는 필터링됨

    Returns
    -------
    notes : List[Dict[str, Any]]
        검출된 음표 리스트. 각 음표는 다음 키를 포함:
        - id (int): 음표 ID (0부터 시작)
        - time_sec (float): 음표 시작 시간 (초)
        - duration_sec (float): 음표 지속 시간 (초)
        - freq_hz (float): 평균 주파수 (Hz)
        - midi (float): 평균 MIDI 노트 번호
        - confidence_mean (float): 평균 신뢰도 (0~1)

    Notes
    -----
    - 주파수가 NaN인 프레임은 무시됩니다 (무음 또는 감지 실패)
    - 연속된 프레임에서 MIDI 값 차이가 max_midi_diff를 초과하면 새 음표 시작
    - 각 음표의 주파수, MIDI, 신뢰도는 해당 구간 프레임들의 평균값

    Examples
    --------
    >>> notes = detect_notes(
    ...     times, freqs, midi, conf,
    ...     max_midi_diff=0.5,
    ...     min_duration_sec=0.1
    ... )
    >>> print(f"검출된 음표: {len(notes)}개")
    >>> print(f"첫 번째 음표: {notes[0]['time_sec']:.2f}초, MIDI {notes[0]['midi']:.1f}")
    """

    n_frames = len(times)
    note_segments = []  # (start_idx, end_idx) 튜플 리스트
    current_start = None

    print(f"[음표 검출] 시작")
    print(f"  - 총 프레임: {n_frames}개")
    print(f"  - MIDI 차이 임계값: {max_midi_diff}")
    print(f"  - 최소 지속 시간: {min_duration_sec}초")

    # 프레임을 순회하며 연속 구간 찾기
    for i in range(n_frames):
        # 유효하지 않은 프레임인지 확인 (주파수 NaN)
        if np.isnan(frequencies[i]):
            # 진행 중이던 음표가 있으면 종료
            if current_start is not None:
                note_segments.append((current_start, i - 1))
                current_start = None
            continue

        # 새로운 음표 시작
        if current_start is None:
            current_start = i
        else:
            # MIDI 값 체크 (둘 다 유효한지 확인)
            if np.isnan(midi_notes[i]) or np.isnan(midi_notes[i - 1]):
                # MIDI가 NaN이면 음표 끊기
                note_segments.append((current_start, i - 1))
                current_start = None
                continue

            # MIDI 값이 급격히 변하면 새 음표로 분리
            if abs(midi_notes[i] - midi_notes[i - 1]) > max_midi_diff:
                note_segments.append((current_start, i - 1))
                current_start = i

    # 마지막 진행 중이던 음표 처리
    if current_start is not None:
        note_segments.append((current_start, n_frames - 1))

    print(f"  - 초기 음표 구간: {len(note_segments)}개")

    # 각 구간을 음표 딕셔너리로 변환
    notes = []
    note_id = 0

    for start_idx, end_idx in note_segments:
        # 시간 정보
        onset = float(times[start_idx])
        offset = float(times[end_idx])
        duration = max(offset - onset, 0.0)

        # 너무 짧은 음표 필터링
        if duration < min_duration_sec:
            continue

        # 해당 구간의 프레임 데이터 추출
        freq_segment = frequencies[start_idx:end_idx + 1]
        midi_segment = midi_notes[start_idx:end_idx + 1]
        conf_segment = confidences[start_idx:end_idx + 1]

        # 평균값 계산
        freq_mean = float(np.nanmean(freq_segment))
        midi_mean = float(np.nanmean(midi_segment))

        # 신뢰도 평균 (모두 NaN이면 0.0)
        if not np.all(np.isnan(conf_segment)):
            conf_mean = float(np.nanmean(conf_segment))
        else:
            conf_mean = 0.0

        # 음표 딕셔너리 생성
        note = {
            "id": note_id,
            "time_sec": onset,
            "duration_sec": duration,
            "freq_hz": freq_mean,
            "midi": midi_mean,
            "confidence_mean": conf_mean,
        }

        notes.append(note)
        note_id += 1

    print(f"[음표 검출] 완료")
    print(f"  - 최종 음표: {len(notes)}개")

    return notes


def save_analysis_result(
        output_path: Path,
        notes: List[Dict[str, Any]],
        metadata: Dict[str, Any],
) -> None:
    """
    오디오 분석 결과를 JSON 파일로 저장합니다.

    Parameters
    ----------
    output_path : Path
        저장할 JSON 파일 경로
    notes : List[Dict[str, Any]]
        detect_notes()로 생성된 음표 리스트
    metadata : Dict[str, Any]
        분석 메타데이터. 다음 키를 포함해야 함:
        - bpm (int): 곡의 BPM
        - source_type (str): 'video' 또는 'audio'
        - source_path (Path): 원본 파일 경로
        - full_audio_path (Path): 전체 오디오 경로
        - session_audio_path (Path): 세션 오디오 경로 (bass.wav 등)
        - sample_rate (int): 샘플레이트
        - pitch_backend (str): 피치 분석 백엔드 (예: 'librosa-pyin')
        - step_size_ms (int): 피치 분석 스텝 크기 (ms)
        - fmin_hz (float): 최소 주파수 (Hz)
        - fmax_hz (float): 최대 주파수 (Hz)

    Notes
    -----
    저장되는 JSON 구조:
    {
        "bpm": 111,
        "source": {"type": "video", "path": "..."},
        "audio": {
            "full_path": "...",
            "session_path": "...",
            "sample_rate": 44100
        },
        "pitch": {
            "backend": "librosa-pyin",
            "step_size_ms": 10,
            "fmin_hz": 40.0,
            "fmax_hz": 300.0
        },
        "notes": [...]
    }

    Examples
    --------
    >>> metadata = {
    ...     'bpm': 111,
    ...     'source_type': 'video',
    ...     'source_path': Path('input.mp4'),
    ...     'full_audio_path': Path('full_audio.wav'),
    ...     'session_audio_path': Path('bass.wav'),
    ...     'sample_rate': 44100,
    ...     'pitch_backend': 'librosa-pyin',
    ...     'step_size_ms': 10,
    ...     'fmin_hz': 40.0,
    ...     'fmax_hz': 300.0,
    ... }
    >>> save_analysis_result(
    ...     Path('output/analysis.json'),
    ...     notes,
    ...     metadata
    ... )
    """

    # JSON 데이터 구조 생성
    data = {
        "bpm": metadata["bpm"],
        "source": {
            "type": metadata["source_type"],
            "path": str(metadata["source_path"]),
        },
        "audio": {
            "full_path": str(metadata["full_audio_path"]),
            "session_path": str(metadata["session_audio_path"]),
            "sample_rate": metadata["sample_rate"],
        },
        "pitch": {
            "backend": metadata["pitch_backend"],
            "step_size_ms": metadata["step_size_ms"],
            "fmin_hz": metadata["fmin_hz"],
            "fmax_hz": metadata["fmax_hz"],
        },
        "notes": notes,
    }

    # 출력 디렉터리 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON 파일로 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[결과 저장] 완료")
    print(f"  - 출력 파일: {output_path}")
    print(f"  - 음표 개수: {len(notes)}개")
    print(f"  - 파일 크기: {output_path.stat().st_size / 1024:.1f} KB")


def load_analysis_result(input_path: Path) -> Dict[str, Any]:
    """
    저장된 분석 결과 JSON 파일을 로드합니다.

    Parameters
    ----------
    input_path : Path
        로드할 JSON 파일 경로

    Returns
    -------
    data : Dict[str, Any]
        분석 결과 딕셔너리 (bpm, source, audio, pitch, notes 키 포함)

    Examples
    --------
    >>> data = load_analysis_result(Path('output/analysis.json'))
    >>> print(f"BPM: {data['bpm']}")
    >>> print(f"음표 개수: {len(data['notes'])}개")
    """

    if not input_path.exists():
        raise FileNotFoundError(f"분석 결과 파일을 찾을 수 없습니다: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[결과 로드] 완료")
    print(f"  - 입력 파일: {input_path}")
    print(f"  - 음표 개수: {len(data['notes'])}개")

    return data