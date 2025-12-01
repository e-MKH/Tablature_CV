"""
비디오-오디오 매칭 파이프라인

JSON 파일 경로를 입력받아 비디오-오디오 매칭을 수행하고
결과를 저장합니다.
"""

from pathlib import Path
from typing import Optional

from src.utils.json_utils import load_json, save_json
from src.matching.video_audio_matcher import batch_match_video_audio, get_matching_statistics


def match_audio_video(
        vision_json: Path,
        audio_json: Path,
        output_json: Path
) -> bool:
    """
    비디오와 오디오 분석 결과를 매칭하고 통계를 저장합니다.

    Parameters
    ----------
    vision_json : Path
        비디오 분석 결과 JSON 경로
    audio_json : Path
        오디오 분석 결과 JSON 경로
    output_json : Path
        매칭 결과를 저장할 JSON 경로

    Returns
    -------
    success : bool
        매칭 성공 여부

    Examples
    --------
    >>> success = match_audio_video(
    ...     vision_json=Path("data/analysis/vision/test_video.json"),
    ...     audio_json=Path("data/analysis/audio/test_audio.json"),
    ...     output_json=Path("data/analysis/matched/test.json")
    ... )
    """
    print(f"\n{'=' * 60}")
    print(f"비디오-오디오 매칭 시작")
    print(f"{'=' * 60}")

    # 1. JSON 데이터 로드
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

    print(f"✅ 데이터 로드 완료")

    # 2. 매칭 수행
    print(f"\n🔗 매칭 수행 중...")

    audio_notes = audio_data.get('notes', [])

    if not audio_notes:
        print(f"⚠️ 오디오 분석 결과에 음표가 없습니다.")
        return False

    matches = batch_match_video_audio(video_data, audio_notes)

    # 3. 통계 계산
    stats = get_matching_statistics(matches)

    print(f"\n📊 매칭 통계:")
    print(f"  - 전체 음표: {stats['total']}개")
    print(f"  - 매칭 성공: {stats['matched']}개 ({stats['match_rate']}%)")
    print(f"  - 매칭 실패: {stats['unmatched']}개")
    print(f"  - Tracking 상태: {stats['tracking_count']}개")

    # 4. 결과 저장
    matched_data = {
        'statistics': stats,
        'matches': matches
    }

    try:
        save_json(matched_data, output_json)
        print(f"\n💾 매칭 결과 저장 완료: {output_json}")
        print(f"{'=' * 60}\n")
        return True

    except Exception as e:
        print(f"\n❌ 매칭 결과 저장 실패: {e}")
        print(f"{'=' * 60}\n")
        return False