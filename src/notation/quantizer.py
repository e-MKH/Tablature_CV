"""
음표 양자화 모듈

음표의 시간 정보를 BPM 기반의 beat(박자) 단위로 변환하고,
음악 이론에 맞게 양자화(스냅)합니다.
"""

from typing import List, Dict, Any


def quantize(value: float, grid: float = 0.125) -> float:
    """
    값을 지정된 그리드 단위로 반올림합니다.

    Examples
    --------
    >>> quantize(0.23, 0.25)
    0.25
    >>> quantize(1.13, 0.25)
    1.0
    >>> quantize(1.37, 0.25)
    1.25
    """
    return round(value / grid) * grid


def calculate_beat_info(
        notes: List[Dict[str, Any]],
        bpm: int
) -> List[Dict[str, Any]]:
    """
    각 음표에 BPM 기반 beat 정보를 추가합니다.

    음표의 시작 시간(time_sec)과 지속 시간(duration_sec)을
    4분음표 기준의 beat 단위로 변환하여 추가합니다.

    Parameters
    ----------
    notes : List[Dict[str, Any]]
        음표 리스트. 각 음표는 'time_sec'과 'duration_sec' 키를 포함해야 함
    bpm : int
        곡의 BPM (Beats Per Minute, 4분음표 기준)

    Returns
    -------
    notes : List[Dict[str, Any]]
        beat 정보가 추가된 음표 리스트. 각 음표에 다음 키가 추가됨:
        - onset_beat (float): 음표 시작 beat (4분음표 기준)
        - duration_beats (float): 음표 지속 시간 (beat 단위)

    Notes
    -----
    beat 계산 공식:
    - onset_beat = time_sec × (bpm / 60)
    - duration_beats = duration_sec × (bpm / 60)

    예: BPM 120일 때
    - 1초 = 2 beats (0.5초 = 1 beat = 4분음표)
    - 0.25초 = 0.5 beats (8분음표)

    Examples
    --------
    >>> notes = [
    ...     {'time_sec': 0.0, 'duration_sec': 0.5},
    ...     {'time_sec': 0.5, 'duration_sec': 1.0},
    ... ]
    >>> notes = calculate_beat_info(notes, bpm=120)
    >>> print(notes[0]['onset_beat'])  # 0.0
    >>> print(notes[0]['duration_beats'])  # 1.0 (4분음표)
    >>> print(notes[1]['onset_beat'])  # 1.0
    >>> print(notes[1]['duration_beats'])  # 2.0 (2분음표)
    """

    print(f"[Beat 계산] 시작 (BPM={bpm})")

    for note in notes:
        time_sec = note["time_sec"]
        duration_sec = note["duration_sec"]

        # 초 → beat 변환
        # beat = 4분음표 기준
        # 예: BPM 60 → 1초 = 1 beat, BPM 120 → 1초 = 2 beats
        onset_beat = time_sec * bpm / 60.0
        duration_beats = duration_sec * bpm / 60.0

        note["onset_beat"] = onset_beat
        note["duration_beats"] = duration_beats

    print(f"[Beat 계산] 완료 ({len(notes)}개 음표)")

    return notes


def quantize_duration(
        duration_beats: float,
        allowed_durations: List[float] = None
) -> float:
    """
    음표 지속 시간을 허용된 음표 길이 중 가장 가까운 값으로 양자화합니다.

    Parameters
    ----------
    duration_beats : float
        원본 지속 시간 (beat 단위)
    allowed_durations : List[float], optional
        허용된 음표 길이 리스트 (beat 단위).
        기본값: [0.25, 0.5, 1.0, 1.5, 2.0]
        (16분음표, 8분음표, 4분음표, 점4분음표, 2분음표)

    Returns
    -------
    quantized : float
        양자화된 지속 시간 (beat 단위)

    Examples
    --------
    >>> quantize_duration(0.3)  # 8분음표에 가까움
    0.5
    >>> quantize_duration(0.9)  # 4분음표에 가까움
    1.0
    >>> quantize_duration(1.3)  # 점4분음표에 가까움
    1.5
    """

    if allowed_durations is None:
        # 기본 허용 음표 길이
        # 16분음표(0.25), 8분음표(0.5), 4분음표(1.0),
        # 점4분음표(1.5), 2분음표(2.0)
        allowed_durations = [0.25, 0.5, 1.0, 1.5, 2.0]

    # 가장 가까운 값 찾기
    return min(allowed_durations, key=lambda x: abs(x - duration_beats))


def quantize_notes(
        notes: List[Dict[str, Any]],
        onset_grid: float = 0.25,
        allowed_durations: List[float] = None
) -> List[Dict[str, Any]]:
    """
    음표의 onset과 duration을 양자화합니다.

    onset_beat은 지정된 그리드(기본 16분음표)에 스냅되고,
    duration_beats는 허용된 음표 길이 중 가장 가까운 값으로 스냅됩니다.

    Parameters
    ----------
    notes : List[Dict[str, Any]]
        음표 리스트. 각 음표는 'onset_beat'과 'duration_beats'를 포함해야 함
        (calculate_beat_info로 추가된 필드)
    onset_grid : float, default=0.25
        onset 양자화 그리드 (beat 단위).
        기본값 0.25 = 16분음표 단위로 스냅
    allowed_durations : List[float], optional
        허용된 음표 길이 리스트 (beat 단위).
        기본값: [0.25, 0.5, 1.0, 1.5, 2.0]

    Returns
    -------
    notes : List[Dict[str, Any]]
        양자화된 정보가 추가된 음표 리스트. 각 음표에 다음 키가 추가됨:
        - onset_beat_q (float): 양자화된 onset (beat)
        - duration_beats_q (float): 양자화된 duration (beat)

    Notes
    -----
    - onset_beat_q: 16분음표(0.25 beat) 그리드에 반올림
      예: 1.37 → 1.25, 1.62 → 1.75
    - duration_beats_q: 미리 정의된 음표 길이 중 선택
      예: 0.3 → 0.25(16분음표), 0.9 → 1.0(4분음표)

    Examples
    --------
    >>> notes = [
    ...     {'onset_beat': 0.13, 'duration_beats': 0.87},
    ...     {'onset_beat': 1.05, 'duration_beats': 1.48},
    ... ]
    >>> notes = quantize_notes(notes)
    >>> print(notes[0]['onset_beat_q'])  # 0.0 (16분음표 그리드)
    >>> print(notes[0]['duration_beats_q'])  # 1.0 (4분음표)
    >>> print(notes[1]['onset_beat_q'])  # 1.0
    >>> print(notes[1]['duration_beats_q'])  # 1.5 (점4분음표)
    """

    print(f"[양자화] 시작")
    print(f"  - Onset 그리드: {onset_grid} beat (16분음표 단위)")

    if allowed_durations is None:
        allowed_durations = [0.25, 0.5, 1.0, 1.5, 2.0]

    print(f"  - 허용 음표 길이: {allowed_durations}")

    for note in notes:
        onset_beat = note["onset_beat"]
        duration_beats = note["duration_beats"]

        # onset을 그리드에 스냅
        # 예: onset_grid=0.25일 때
        # 1.13 → round(1.13/0.25)*0.25 = 1.0
        # 1.37 → round(1.37/0.25)*0.25 = 1.25
        note["onset_beat_q"] = round(onset_beat / onset_grid) * onset_grid

        # duration을 허용된 값으로 스냅
        note["duration_beats_q"] = quantize_duration(
            duration_beats,
            allowed_durations
        )

    print(f"[양자화] 완료 ({len(notes)}개 음표)")

    return notes


def get_note_name(duration_beats: float) -> str:
    """
    beat 길이를 음표 이름으로 변환합니다.

    Parameters
    ----------
    duration_beats : float
        음표 길이 (beat 단위)

    Returns
    -------
    name : str
        음표 이름 (한글)

    Examples
    --------
    >>> get_note_name(0.25)
    '16분음표'
    >>> get_note_name(1.0)
    '4분음표'
    >>> get_note_name(1.5)
    '점4분음표'
    """

    note_names = {
        0.125: "32분음표",
        0.25: "16분음표",
        0.5: "8분음표",
        0.75: "점8분음표",
        1.0: "4분음표",
        1.5: "점4분음표",
        2.0: "2분음표",
        3.0: "점2분음표",
        4.0: "온음표",
    }

    return note_names.get(duration_beats, f"{duration_beats} beats")


def print_quantization_summary(notes: List[Dict[str, Any]]) -> None:
    """
    양자화 결과를 요약하여 출력합니다.

    Parameters
    ----------
    notes : List[Dict[str, Any]]
        양자화된 음표 리스트

    Examples
    --------
    >>> print_quantization_summary(notes)
    [양자화 요약]
      - 16분음표: 15개
      - 8분음표: 42개
      - 4분음표: 38개
      - 점4분음표: 5개
    """

    from collections import Counter

    durations = [note.get("duration_beats_q", 0) for note in notes]
    duration_counts = Counter(durations)

    print(f"\n[양자화 요약]")
    for duration in sorted(duration_counts.keys()):
        count = duration_counts[duration]
        name = get_note_name(duration)
        print(f"  - {name}: {count}개")