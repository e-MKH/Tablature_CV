"""
비디오-오디오 매칭 모듈

오디오 분석에서 감지된 음표(onset)와 비디오 분석에서 감지된
손가락 포지션을 시간 기반으로 매칭합니다.

매칭 과정:
1. 오디오 onset 시점 ± TIME_WINDOW 내의 비디오 프레임 탐색
2. 비디오 프레임의 MIDI 음정과 오디오 MIDI 비교
3. 음정 차이가 PITCH_TOLERANCE 이내인 후보들 중 최적 선택
4. tracking 상태인 프레임을 우선적으로 선택
"""

from typing import List, Dict, Optional
from src.config.constants import TIME_WINDOW_SEC, PITCH_TOLERANCE
from src.notation.position_predictor import get_video_midi


def find_candidates_in_time_window(
        video_data: List[Dict],
        onset_sec: float,
        time_window: float = TIME_WINDOW_SEC
) -> List[Dict]:
    """
    시간 윈도우 내의 비디오 프레임 후보들을 찾습니다.

    Parameters
    ----------
    video_data : List[Dict]
        비디오 분석 결과 리스트. 각 항목은 다음을 포함:
        - time_sec: 프레임 시간 (초)
        - string: 현 이름
        - fret: 프렛 번호
        - finger: 손가락 정보 (옵션)
        - is_tracking: 추적 상태 (옵션)
    onset_sec : float
        오디오 onset 시간 (초)
    time_window : float, default=TIME_WINDOW_SEC
        허용 시간 윈도우 (초). ± 이 시간 내의 프레임을 후보로 간주

    Returns
    -------
    candidates : List[Dict]
        시간 윈도우 내의 비디오 프레임들

    Examples
    --------
    >>> video_data = [
    ...     {'time_sec': 1.0, 'string': 'E', 'fret': 5},
    ...     {'time_sec': 1.1, 'string': 'A', 'fret': 3},
    ...     {'time_sec': 1.5, 'string': 'D', 'fret': 0}
    ... ]
    >>> candidates = find_candidates_in_time_window(video_data, 1.05, 0.15)
    >>> len(candidates)
    2  # 1.0초와 1.1초 프레임만 포함 (1.5초는 윈도우 밖)
    """
    candidates = []

    for frame in video_data:
        frame_time = frame['time_sec']
        time_diff = abs(frame_time - onset_sec)

        if time_diff <= time_window:
            candidates.append(frame)

    return candidates


def calculate_pitch_difference(
        video_frame: Dict,
        audio_midi: int
) -> int:
    """
    비디오 프레임과 오디오 MIDI 간의 음정 차이를 계산합니다.

    Parameters
    ----------
    video_frame : Dict
        비디오 프레임 정보 (string, fret 포함)
    audio_midi : int
        오디오에서 분석된 MIDI 번호

    Returns
    -------
    pitch_diff : int
        음정 차이 (반음 단위). 비디오 프레임이 유효하지 않으면 999 반환

    Examples
    --------
    >>> frame = {'string': 'E', 'fret': 5}
    >>> calculate_pitch_difference(frame, 33)
    0  # E-5프렛 = MIDI 33, 정확히 일치

    >>> frame = {'string': 'A', 'fret': 3}
    >>> calculate_pitch_difference(frame, 35)
    1  # A-3프렛 = MIDI 36, 1반음 차이
    """
    video_midi = get_video_midi(video_frame['string'], video_frame['fret'])

    # 유효하지 않은 포지션
    if video_midi == -1:
        return 999

    return abs(audio_midi - video_midi)


def select_best_match(
        candidates: List[Dict],
        audio_midi: int,
        pitch_tolerance: int = PITCH_TOLERANCE
) -> Optional[Dict]:
    """
    후보 프레임들 중 최적의 매칭을 선택합니다.

    선택 기준 (우선순위):
    1. 음정 차이가 pitch_tolerance 이내인가?
    2. 음정 차이가 더 작은가?
    3. 같은 차이면 is_tracking=True인가? (추적 중인 프레임 우선)

    Parameters
    ----------
    candidates : List[Dict]
        시간 윈도우 내의 후보 프레임들
    audio_midi : int
        오디오에서 분석된 MIDI 번호
    pitch_tolerance : int, default=PITCH_TOLERANCE
        허용 음정 오차 (반음 단위)

    Returns
    -------
    best_match : Dict or None
        최적 매칭 프레임. 적합한 후보가 없으면 None

    Examples
    --------
    >>> candidates = [
    ...     {'string': 'E', 'fret': 5, 'is_tracking': False},  # MIDI 33
    ...     {'string': 'A', 'fret': 0, 'is_tracking': True},   # MIDI 33
    ... ]
    >>> match = select_best_match(candidates, 33, pitch_tolerance=1)
    >>> match['is_tracking']
    True  # 같은 음정이면 tracking 우선

    >>> candidates = [
    ...     {'string': 'E', 'fret': 5},  # MIDI 33
    ...     {'string': 'A', 'fret': 3},  # MIDI 36
    ... ]
    >>> match = select_best_match(candidates, 33, pitch_tolerance=1)
    >>> match['fret']
    5  # 음정이 정확히 맞는 것 선택
    """
    if not candidates:
        return None

    best_match = None
    min_pitch_diff = 999

    for candidate in candidates:
        # 음정 차이 계산
        pitch_diff = calculate_pitch_difference(candidate, audio_midi)

        # pitch_tolerance를 초과하면 무시
        if pitch_diff > pitch_tolerance:
            continue

        # 더 작은 차이를 가진 후보 발견
        if pitch_diff < min_pitch_diff:
            min_pitch_diff = pitch_diff
            best_match = candidate

        # 같은 차이인 경우 → tracking 상태 우선
        elif pitch_diff == min_pitch_diff:
            # 기존 best_match가 tracking이 아닌데, 새 candidate가 tracking이면 교체
            if best_match and not best_match.get('is_tracking', False) \
                    and candidate.get('is_tracking', False):
                best_match = candidate

    return best_match


def match_video_to_audio(
        video_data: List[Dict],
        audio_note: Dict,
        time_window: float = TIME_WINDOW_SEC,
        pitch_tolerance: int = PITCH_TOLERANCE
) -> Optional[Dict]:
    """
    단일 오디오 음표에 대해 최적의 비디오 프레임을 매칭합니다.

    전체 매칭 프로세스:
    1. 시간 윈도우 내의 비디오 프레임 후보 탐색
    2. 각 후보의 음정과 오디오 음정 비교
    3. 최적 매칭 선택 (음정 차이 최소 + tracking 우선)

    Parameters
    ----------
    video_data : List[Dict]
        전체 비디오 분석 결과
    audio_note : Dict
        매칭할 오디오 음표. 다음을 포함:
        - onset_sec: 음표 시작 시간
        - midi: MIDI 번호
    time_window : float, default=TIME_WINDOW_SEC
        시간 윈도우 (초)
    pitch_tolerance : int, default=PITCH_TOLERANCE
        음정 허용 오차 (반음)

    Returns
    -------
    matched_frame : Dict or None
        매칭된 비디오 프레임. 매칭 실패 시 None

    Examples
    --------
    >>> video_data = [
    ...     {'time_sec': 1.0, 'string': 'E', 'fret': 5, 'finger': '1'},
    ...     {'time_sec': 2.0, 'string': 'A', 'fret': 0, 'is_tracking': True}
    ... ]
    >>> audio_note = {'onset_sec': 1.05, 'midi': 33}
    >>> match = match_video_to_audio(video_data, audio_note)
    >>> match['fret']
    5

    Notes
    -----
    매칭 실패 케이스:
    - 시간 윈도우 내에 비디오 프레임이 없음
    - 모든 후보의 음정 차이가 pitch_tolerance 초과
    """
    onset_sec = audio_note['time_sec']
    audio_midi = int(round(audio_note['midi']))

    # 1단계: 시간 윈도우 내 후보 탐색
    candidates = find_candidates_in_time_window(
        video_data,
        onset_sec,
        time_window
    )

    if not candidates:
        return None

    # 2단계: 최적 매칭 선택
    best_match = select_best_match(
        candidates,
        audio_midi,
        pitch_tolerance
    )

    return best_match


def batch_match_video_audio(
        video_data: List[Dict],
        audio_notes: List[Dict],
        time_window: float = TIME_WINDOW_SEC,
        pitch_tolerance: int = PITCH_TOLERANCE
) -> List[Optional[Dict]]:
    """
    여러 오디오 음표를 비디오 데이터와 일괄 매칭합니다.

    Parameters
    ----------
    video_data : List[Dict]
        전체 비디오 분석 결과
    audio_notes : List[Dict]
        매칭할 오디오 음표 리스트
    time_window : float, default=TIME_WINDOW_SEC
        시간 윈도우 (초)
    pitch_tolerance : int, default=PITCH_TOLERANCE
        음정 허용 오차 (반음)

    Returns
    -------
    matches : List[Optional[Dict]]
        각 오디오 음표에 대한 매칭 결과 리스트
        매칭 실패 시 해당 인덱스는 None

    Examples
    --------
    >>> audio_notes = [
    ...     {'onset_sec': 1.0, 'midi': 33},
    ...     {'onset_sec': 2.0, 'midi': 38}
    ... ]
    >>> matches = batch_match_video_audio(video_data, audio_notes)
    >>> len(matches)
    2
    """
    matches = []

    for audio_note in audio_notes:
        match = match_video_to_audio(
            video_data,
            audio_note,
            time_window,
            pitch_tolerance
        )
        matches.append(match)

    return matches


def get_matching_statistics(matches: List[Optional[Dict]]) -> Dict[str, any]:
    """
    매칭 결과 통계를 계산합니다.

    Parameters
    ----------
    matches : List[Optional[Dict]]
        batch_match_video_audio의 결과

    Returns
    -------
    stats : Dict[str, any]
        매칭 통계
        - total: 전체 음표 수
        - matched: 매칭 성공 수
        - unmatched: 매칭 실패 수
        - match_rate: 매칭 성공률 (%)
        - tracking_count: tracking 상태인 매칭 수

    Examples
    --------
    >>> matches = [
    ...     {'string': 'E', 'fret': 5, 'is_tracking': True},
    ...     None,
    ...     {'string': 'A', 'fret': 0, 'is_tracking': False}
    ... ]
    >>> stats = get_matching_statistics(matches)
    >>> stats['match_rate']
    66.67
    >>> stats['tracking_count']
    1
    """
    total = len(matches)
    matched = sum(1 for m in matches if m is not None)
    unmatched = total - matched

    match_rate = (matched / total * 100) if total > 0 else 0.0

    tracking_count = sum(
        1 for m in matches
        if m is not None and m.get('is_tracking', False)
    )

    return {
        'total': total,
        'matched': matched,
        'unmatched': unmatched,
        'match_rate': round(match_rate, 2),
        'tracking_count': tracking_count
    }