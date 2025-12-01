from pathlib import Path
from .bass_pitch_analyzer import run_pyin_analysis
from .note_detector import detect_notes, save_analysis_result
from ..notation.quantizer import (
    calculate_beat_info,
    quantize_notes,
    print_quantization_summary
)


def analyze_audio(
        session_audio_path: Path,
        output_json_path: Path,
        bpm: int,
        source_path: Path,
        full_audio_path: Path,
        step_size_ms: int = 10,
        fmin_hz: float = 40.0,
        fmax_hz: float = 300.0,
        max_midi_diff: float = 0.5,
        min_duration_sec: float = 0.1,
        onset_grid: float = 0.25,
) -> None:
    """
    오디오 분석 전체 파이프라인을 실행합니다.

    피치 분석 → 음표 검출 → Beat 계산 → 양자화 → JSON 저장

    Parameters
    ----------
    session_audio_path : Path
        분석할 세션 오디오 파일 경로 (bass.wav, guitar.wav 등)
    output_json_path : Path
        결과를 저장할 JSON 파일 경로
    bpm : int
        곡의 BPM (Beats Per Minute)
    source_path : Path
        원본 입력 파일 경로 (동영상 또는 오디오)
    full_audio_path : Path
        전체 오디오 파일 경로 (세션 분리 전)
    step_size_ms : int, default=10
        피치 분석 스텝 크기 (밀리초)
    fmin_hz : float, default=40.0
        최소 주파수 (Hz). 베이스: 40, 기타: 80
    fmax_hz : float, default=300.0
        최대 주파수 (Hz). 베이스: 300, 기타: 1000
    max_midi_diff : float, default=0.5
        연속 프레임 간 MIDI 차이 임계값
    min_duration_sec : float, default=0.1
        최소 음표 지속 시간 (초)
    onset_grid : float, default=0.25
        Onset 양자화 그리드 (beat 단위, 0.25 = 16분음표)

    Examples
    --------
    >>> from pathlib import Path
    >>> from src.audio_analysis.audio_pipeline import analyze_audio
    >>>
    >>> analyze_audio(
    ...     session_audio_path=Path('processed/bass.wav'),
    ...     output_json_path=Path('analysis/bass_notes.json'),
    ...     bpm=120,
    ...     source_path=Path('input/song.mp4'),
    ...     full_audio_path=Path('processed/audio.wav')
    ... )
    """

    print("\n" + "=" * 60)
    print("오디오 분석 파이프라인 시작")
    print("=" * 60)
    print(f"입력: {session_audio_path.name}")
    print(f"출력: {output_json_path}")
    print(f"BPM: {bpm}")

    # 1. 피치 분석
    print("\n[1/5] 피치 분석 (librosa.pyin)")
    sr, times, freqs, midi, conf = run_pyin_analysis(
        audio_path=session_audio_path,
        step_size_ms=step_size_ms,
        fmin_hz=fmin_hz,
        fmax_hz=fmax_hz,
    )

    # 2. 음표 검출
    print("\n[2/5] 음표 검출")
    notes = detect_notes(
        times=times,
        frequencies=freqs,
        midi_notes=midi,
        confidences=conf,
        max_midi_diff=max_midi_diff,
        min_duration_sec=min_duration_sec,
    )

    # 3. Beat 계산
    print(f"\n[3/5] Beat 계산")
    notes = calculate_beat_info(notes, bpm=bpm)

    # 4. 양자화
    print("\n[4/5] 양자화")
    notes = quantize_notes(
        notes=notes,
        onset_grid=onset_grid,
        allowed_durations=[0.25, 0.5, 1.0, 1.5, 2.0]
    )

    # 양자화 결과 요약
    print_quantization_summary(notes)

    # 5. 메타데이터 준비 및 저장
    print("\n[5/5] 결과 저장")

    # 소스 타입 판별
    video_exts = {'.mp4', '.mov', '.mkv', '.avi'}
    source_type = 'video' if source_path.suffix.lower() in video_exts else 'audio'

    metadata = {
        'bpm': bpm,
        'source_type': source_type,
        'source_path': source_path,
        'full_audio_path': full_audio_path,
        'session_audio_path': session_audio_path,
        'sample_rate': sr,
        'pitch_backend': 'librosa-pyin',
        'step_size_ms': step_size_ms,
        'fmin_hz': fmin_hz,
        'fmax_hz': fmax_hz,
    }

    save_analysis_result(
        output_path=output_json_path,
        notes=notes,
        metadata=metadata
    )

    print("\n" + "=" * 60)
    print("오디오 분석 파이프라인 완료")
    print("=" * 60)