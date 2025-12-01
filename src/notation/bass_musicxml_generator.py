"""
베이스 기타 MusicXML 생성 모듈

비디오 분석과 오디오 분석 결과를 통합하여 MusicXML 파일을 생성합니다.
비디오-오디오 매칭, 포지션 예측, 박자 양자화 등을 수행하고
music21 라이브러리를 사용하여 최종 악보를 생성합니다.
"""

from typing import List, Dict, Optional
from pathlib import Path

from music21 import stream, note, tempo, meter, instrument

from src.config.constants import (
    QUANTIZE_GRID,
    DEFAULT_BPM,
    DEFAULT_TIME_SIGNATURE
)
from src.matching.video_audio_matcher import match_video_to_audio
from src.notation.position_predictor import predict_best_position
from src.notation.quantizer import quantize


def generate_bass_musicxml(
        vision_json: Path,
        audio_json: Path,
        output_path: Path,
        quantize_grid: float = QUANTIZE_GRID
) -> bool:
    """
    비디오/오디오 분석 결과를 MusicXML 악보로 변환

    Parameters
    ----------
    vision_json : Path
        비디오 분석 결과 JSON 경로
    audio_json : Path
        오디오 분석 결과 JSON 경로
    output_path : Path
        저장할 MusicXML 파일 경로
    quantize_grid : float, default=QUANTIZE_GRID
        박자 양자화 그리드 (beat 단위)
        0.25 = 16분음표, 0.125 = 32분음표

    Returns
    -------
    success : bool
        생성 성공 여부

    Examples
    --------
    >>> success = generate_bass_musicxml(
    ...     vision_json=Path("data/analysis/vision/test_video.json"),
    ...     audio_json=Path("data/analysis/audio/test_audio.json"),
    ...     output_path=Path("output.musicxml")
    ... )
    >>> if success:
    ...     print("MusicXML 생성 완료")
    """
    print(f"\n{'='*60}")
    print(f"베이스 기타 MusicXML 생성 시작")
    print(f"{'='*60}")

    # JSON 데이터 로드
    from src.utils.json_utils import load_json

    print(f"\n📂 데이터 로드 중...")
    print(f"  - 비디오: {vision_json}")
    print(f"  - 오디오: {audio_json}")

    video_data = load_json(vision_json)
    audio_data = load_json(audio_json)

    if video_data is None:
        print(f"❌ 비디오 분석 결과를 불러올 수 없습니다: {vision_json}")
        return False

    if audio_data is None:
        print(f"❌ 오디오 분석 결과를 불러올 수 없습니다: {audio_json}")
        return False

    # 데이터 유효성 검증
    if not validate_video_data(video_data):
        return False

    if not validate_audio_data(audio_data):
        return False

    print(f"✅ 데이터 로드 완료")

    # 1. 데이터 정렬 (시간순)
    video_data = sorted(video_data, key=lambda x: x['time_sec'])

    # 2. Score 및 Part 생성
    score = stream.Score()
    part = stream.Part()
    part.id = "BassPart"

    # 3. 악기 및 박자 설정
    part.insert(0, instrument.ElectricBass())
    part.insert(0, meter.TimeSignature(DEFAULT_TIME_SIGNATURE))

    bpm = audio_data.get("bpm", DEFAULT_BPM)
    part.insert(0, tempo.MetronomeMark(number=bpm))

    print(f"설정: BPM={bpm}, 박자표={DEFAULT_TIME_SIGNATURE}, Grid={quantize_grid}")

    # 4. 음표 처리
    audio_notes = audio_data.get('notes', [])

    # 이전 포지션 추적 (예측에 사용)
    last_known_pos = {'string': 'E', 'fret': 0}

    # 이전 노트가 끝난 박자 위치
    prev_end_beat = 0.0

    # 통계
    stats = {
        'total': len(audio_notes),
        'tracked': 0,
        'predicted': 0
    }

    print(f"\n총 {stats['total']}개 음표 처리 시작...")

    # 5. 각 오디오 음표 처리
    for i, audio_note in enumerate(audio_notes):
        onset_sec = audio_note['time_sec']

        # 5-1. 박자 계산 및 양자화
        if "onset_beat" in audio_note:
            raw_onset = float(audio_note["onset_beat"])
            raw_dur = float(audio_note.get("duration_beats", 1.0))
        else:
            raw_onset = onset_sec * (bpm / 60.0)
            raw_dur = audio_note.get("duration_sec", 0.5) * (bpm / 60.0)

        onset_beat = quantize(raw_onset, quantize_grid)
        duration_beat = quantize(raw_dur, quantize_grid)

        # 최소 길이 보장
        if duration_beat < quantize_grid:
            duration_beat = quantize_grid

        audio_midi = int(round(audio_note['midi']))

        # 5-2. 비디오 데이터 매칭
        best_match = match_video_to_audio(video_data, audio_note)

        # 5-3. 포지션 결정 (매칭 or 예측)
        if best_match:
            # 매칭 성공
            final_string = best_match['string']
            final_fret = best_match['fret']
            finger = best_match.get('finger', '')
            status = "TRACKED"
            stats['tracked'] += 1
            last_known_pos = {'string': final_string, 'fret': final_fret}
        else:
            # 매칭 실패 → 예측
            predicted = predict_best_position(audio_midi, last_known_pos)
            final_string = predicted['string']
            final_fret = predicted['fret']
            finger = ""
            status = "PREDICTED"
            stats['predicted'] += 1
            last_known_pos = predicted

        # 5-4. 쉼표 추가 (이전 노트와 간격이 있으면)
        gap = onset_beat - prev_end_beat

        if gap >= quantize_grid:
            gap = quantize(gap, quantize_grid)
            if gap > 0:
                rest = note.Rest()
                rest.quarterLength = gap
                part.append(rest)
                prev_end_beat += gap

        # 5-5. 시간 역전 방지
        if onset_beat < prev_end_beat:
            onset_beat = prev_end_beat

        # 5-6. 음표 생성
        nt = note.Note(audio_midi)
        nt.quarterLength = duration_beat

        # 5-7. 가사(탭 정보) 추가
        tab_text = f"{final_string}:{final_fret}"
        nt.addLyric(tab_text)

        if finger:
            nt.addLyric(f"({finger})")

        if status == "PREDICTED":
            nt.style.color = 'red'  # 예측된 포지션은 빨간색
            nt.addLyric("?")

        part.append(nt)

        # 다음 노트의 시작 기준점 업데이트
        prev_end_beat = onset_beat + duration_beat

        # 진행 상황 출력 (10개마다)
        if (i + 1) % 10 == 0:
            print(f"  진행: {i + 1}/{stats['total']} 음표 처리 완료")

    # 6. Score에 Part 추가
    score.append(part)

    # 7. MusicXML 파일 저장
    try:
        score.write("musicxml", fp=output_path)

        print(f"\n{'='*60}")
        print(f"✅ MusicXML 생성 완료!")
        print(f"{'='*60}")
        print(f"파일: {output_path}")
        print(f"\n📊 통계:")
        print(f"  - 전체 음표: {stats['total']}개")
        print(f"  - 비디오 매칭: {stats['tracked']}개 ({stats['tracked']/stats['total']*100:.1f}%)")
        print(f"  - 예측: {stats['predicted']}개 ({stats['predicted']/stats['total']*100:.1f}%)")
        print(f"\n💡 예측된 포지션은 악보에서 빨간색으로 표시됩니다.")
        print(f"{'='*60}\n")

        return True

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ MusicXML 저장 실패!")
        print(f"{'='*60}")
        print(f"오류: {e}")
        print(f"\n💡 힌트:")
        print(f"  - 박자가 너무 복잡하면 QUANTIZE_GRID를 0.25로 늘려보세요.")
        print(f"  - music21 라이브러리가 제대로 설치되었는지 확인하세요.")
        print(f"{'='*60}\n")

        return False


def create_note_with_tablature(
        midi_number: int,
        duration_beats: float,
        string_name: str,
        fret_number: int,
        finger: str = "",
        is_predicted: bool = False
) -> note.Note:
    """
    탭 정보가 포함된 음표 객체 생성

    Parameters
    ----------
    midi_number : int
        MIDI 번호
    duration_beats : float
        음표 길이 (beat 단위)
    string_name : str
        현 이름 (E, A, D, G)
    fret_number : int
        프렛 번호
    finger : str, optional
        손가락 번호 (1, 2, 3, 4 등)
    is_predicted : bool, default=False
        예측된 포지션인지 여부 (True면 빨간색 표시)

    Returns
    -------
    nt : note.Note
        생성된 음표 객체

    Examples
    --------
    >>> nt = create_note_with_tablature(
    ...     midi_number=33,
    ...     duration_beats=1.0,
    ...     string_name='A',
    ...     fret_number=0,
    ...     finger='1'
    ... )
    >>> nt.pitch.midi
    33
    >>> nt.lyrics[0].text
    'A:0'
    """
    nt = note.Note(midi_number)
    nt.quarterLength = duration_beats

    # 탭 정보 추가
    tab_text = f"{string_name}:{fret_number}"
    nt.addLyric(tab_text)

    if finger:
        nt.addLyric(f"({finger})")

    if is_predicted:
        nt.style.color = 'red'
        nt.addLyric("?")

    return nt


def add_metadata_to_score(
        score: stream.Score,
        title: str = "Bass Tablature",
        composer: str = "Tablature_CV"
) -> stream.Score:
    """
    악보에 메타데이터 추가

    Parameters
    ----------
    score : stream.Score
        music21 Score 객체
    title : str, default="Bass Tablature"
        곡 제목
    composer : str, default="Tablature_CV"
        작곡가/생성 도구

    Returns
    -------
    score : stream.Score
        메타데이터가 추가된 Score 객체

    Examples
    --------
    >>> score = stream.Score()
    >>> score = add_metadata_to_score(score, title="My Song")
    >>> score.metadata.title
    'My Song'
    """
    score.metadata = stream.metadata.Metadata()
    score.metadata.title = title
    score.metadata.composer = composer

    return score


def validate_audio_data(audio_data: Dict) -> bool:
    """
    오디오 데이터 유효성 검증

    Parameters
    ----------
    audio_data : Dict
        검증할 오디오 분석 데이터

    Returns
    -------
    is_valid : bool
        유효하면 True

    Notes
    -----
    검증 항목:
    - 'notes' 키가 있는지
    - notes가 리스트인지
    - 각 note에 필수 필드(onset_sec, midi)가 있는지
    """
    if not audio_data or not isinstance(audio_data, dict):
        print("❌ 오디오 데이터가 유효하지 않습니다.")
        return False

    if 'notes' not in audio_data:
        print("❌ 오디오 데이터에 'notes' 키가 없습니다.")
        return False

    notes = audio_data['notes']

    if not isinstance(notes, list):
        print("❌ 'notes'가 리스트가 아닙니다.")
        return False

    if len(notes) == 0:
        print("⚠️ 오디오 데이터에 음표가 없습니다.")
        return False

    # 첫 번째 음표의 필수 필드 확인
    first_note = notes[0]
    required_fields = ['time_sec', 'midi']

    for field in required_fields:
        if field not in first_note:
            print(f"❌ 음표에 필수 필드 '{field}'가 없습니다.")
            return False

    return True


def validate_video_data(video_data: List[Dict]) -> bool:
    """
    비디오 데이터 유효성 검증

    Parameters
    ----------
    video_data : List[Dict]
        검증할 비디오 분석 데이터

    Returns
    -------
    is_valid : bool
        유효하면 True

    Notes
    -----
    검증 항목:
    - 리스트인지
    - 각 프레임에 필수 필드(time_sec, string, fret)가 있는지
    """
    if not video_data or not isinstance(video_data, list):
        print("❌ 비디오 데이터가 유효하지 않습니다.")
        return False

    if len(video_data) == 0:
        print("⚠️ 비디오 데이터에 프레임이 없습니다.")
        return False

    # 첫 번째 프레임의 필수 필드 확인
    first_frame = video_data[0]
    required_fields = ['time_sec', 'string', 'fret']

    for field in required_fields:
        if field not in first_frame:
            print(f"❌ 비디오 프레임에 필수 필드 '{field}'가 없습니다.")
            return False

    return True