"""
피치 분석 모듈

오디오 파일(주로 베이스 트랙)에서 librosa.pyin을 사용하여
시간에 따른 피치(음정) 정보를 추출합니다.
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import soundfile as sf
import librosa


def run_pyin_analysis(
        audio_path: Path,
        step_size_ms: int = 10,
        fmin_hz: float = 40.0,
        fmax_hz: float = 300.0,
        frame_length: int = 2048,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    오디오 파일에서 librosa.pyin을 사용하여 피치를 분석합니다.

    Parameters
    ----------
    audio_path : Path
        분석할 오디오 파일 경로 (주로 bass.wav)
    step_size_ms : int, default=10
        프레임 간격 (밀리초). 작을수록 시간 해상도가 높아짐
    fmin_hz : float, default=40.0
        탐지할 최소 주파수 (Hz). 베이스의 낮은 음역대 고려
    fmax_hz : float, default=300.0
        탐지할 최대 주파수 (Hz). 베이스의 높은 음역대 고려
    frame_length : int, default=2048
        FFT 프레임 길이

    Returns
    -------
    sample_rate : int
        오디오 샘플레이트
    times : np.ndarray
        각 프레임의 시간 (초 단위)
    frequencies : np.ndarray
        각 프레임의 주파수 (Hz). 유효하지 않은 프레임은 NaN
    midi_notes : np.ndarray
        각 프레임의 MIDI 노트 번호. 유효하지 않은 프레임은 NaN
    confidences : np.ndarray
        각 프레임의 신뢰도 (0~1). 유효하지 않은 프레임은 0.0

    Notes
    -----
    - 스테레오 파일은 자동으로 모노로 변환됩니다.
    - 주파수가 감지되지 않은 프레임은 NaN으로 표시됩니다.
    - MIDI 노트 번호는 A4=440Hz=69를 기준으로 계산됩니다.

    Examples
    --------
    >>> from pathlib import Path
    >>> sr, times, freqs, midi, conf = run_pyin_analysis(
    ...     Path("bass.wav"),
    ...     step_size_ms=10,
    ...     fmin_hz=40.0,
    ...     fmax_hz=300.0
    ... )
    >>> print(f"총 {len(times)}개 프레임 분석 완료")
    >>> valid_frames = ~np.isnan(freqs)
    >>> print(f"유효 프레임: {np.sum(valid_frames)}개")
    """

    # 오디오 파일 읽기
    audio, sr = sf.read(str(audio_path))

    # 스테레오를 모노로 변환
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # hop_length 계산: step_size_ms를 샘플 수로 변환
    # 예: sr=44100, step_size_ms=10 → hop_length=441
    hop_length = int(sr * step_size_ms / 1000)

    print(f"[피치 분석] 시작")
    print(f"  - 파일: {audio_path.name}")
    print(f"  - 샘플레이트: {sr} Hz")
    print(f"  - Step size: {step_size_ms} ms (hop_length={hop_length})")
    print(f"  - Frame length: {frame_length}")
    print(f"  - 주파수 범위: {fmin_hz:.1f} ~ {fmax_hz:.1f} Hz")

    # librosa.pyin 실행
    # f0: 기본 주파수 (Hz), voiced_flag: 유성음 여부, voiced_prob: 유성음 확률
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=fmin_hz,
        fmax=fmax_hz,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    # 시간축 생성 (초 단위)
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    # 결과 배열 준비
    frequencies = np.array(f0, dtype=float)  # Hz, unvoiced → NaN
    confidences = np.array(voiced_prob, dtype=float)  # 0~1, unvoiced → NaN

    # NaN 프레임 처리: 주파수가 NaN이면 신뢰도를 0으로 설정
    invalid_mask = np.isnan(frequencies)
    confidences[invalid_mask] = 0.0

    # 주파수를 MIDI 노트 번호로 변환
    # MIDI = 69 + 12 * log2(freq / 440)
    midi_notes = np.full_like(frequencies, np.nan, dtype=float)
    valid_mask = ~invalid_mask & (frequencies > 0)
    midi_notes[valid_mask] = 69 + 12 * np.log2(frequencies[valid_mask] / 440.0)

    # 분석 결과 요약
    total_frames = len(times)
    valid_frames = np.sum(valid_mask)
    print(f"[피치 분석] 완료")
    print(f"  - 총 프레임: {total_frames}개")
    print(f"  - 유효 프레임: {valid_frames}개 ({valid_frames / total_frames * 100:.1f}%)")

    return sr, times, frequencies, midi_notes, confidences


def analyze_pitch_statistics(
        frequencies: np.ndarray,
        midi_notes: np.ndarray,
        confidences: np.ndarray
) -> dict:
    """
    피치 분석 결과의 통계 정보를 계산합니다.

    Parameters
    ----------
    frequencies : np.ndarray
        주파수 배열 (Hz)
    midi_notes : np.ndarray
        MIDI 노트 번호 배열
    confidences : np.ndarray
        신뢰도 배열 (0~1)

    Returns
    -------
    stats : dict
        통계 정보 딕셔너리
        - valid_count: 유효한 프레임 개수
        - total_count: 전체 프레임 개수
        - valid_ratio: 유효 프레임 비율
        - freq_min, freq_max, freq_mean: 주파수 통계
        - midi_min, midi_max, midi_mean: MIDI 통계
        - confidence_mean: 평균 신뢰도
    """
    valid_mask = ~np.isnan(frequencies)

    stats = {
        "valid_count": int(np.sum(valid_mask)),
        "total_count": len(frequencies),
        "valid_ratio": float(np.mean(valid_mask)),
    }

    if stats["valid_count"] > 0:
        valid_freqs = frequencies[valid_mask]
        valid_midi = midi_notes[valid_mask]
        valid_conf = confidences[valid_mask]

        stats.update({
            "freq_min": float(np.min(valid_freqs)),
            "freq_max": float(np.max(valid_freqs)),
            "freq_mean": float(np.mean(valid_freqs)),
            "midi_min": float(np.min(valid_midi)),
            "midi_max": float(np.max(valid_midi)),
            "midi_mean": float(np.mean(valid_midi)),
            "confidence_mean": float(np.mean(valid_conf)),
        })
    else:
        stats.update({
            "freq_min": None,
            "freq_max": None,
            "freq_mean": None,
            "midi_min": None,
            "midi_max": None,
            "midi_mean": None,
            "confidence_mean": 0.0,
        })

    return stats