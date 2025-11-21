from src.separator import AudioVideoSeparator
from src.audio.session_separator import SessionSeparator


def main():
    """
    프로그램 진행 과정:
    1. 동영상 업로드
    2. 오디오/비디오 분리 및 동기화 검증
    3. 세션 분리 (기타/베이스 트랙 분리)
    """

    print("\n" + "=" * 70)
    print("🎵 기타 연주 분석 시스템 - 오디오 전처리")
    print("=" * 70)

    # ========================================
    # 1단계: 오디오/비디오 분리
    # ========================================
    print("\n[1단계] 오디오/비디오 분리")
    print("-" * 70)

    av_separator = AudioVideoSeparator(output_dir="./data/output")

    # 대상 비디오 지정 (data/input/example.mp4 또는 파일명만)
    video_file = "example_001.mp4"
    av_result = av_separator.separate(video_file)

    print("\n✅ 1단계 완료!")
    print(f"   📹 비디오: {av_result.video_path.name}")
    print(f"   🎵 오디오: {av_result.audio_path.name}")
    print(f"   🔄 동기화: {'정상' if av_result.is_synced else '⚠️ 불일치'} (오프셋: {av_result.sync_offset:.3f}s)")

    # ========================================
    # 2단계: 세션 분리 (기타/베이스)
    # ========================================
    print("\n[2단계] 세션 분리 (기타/베이스)")
    print("-" * 70)

    session_separator = SessionSeparator(
        model_name="htdemucs",  # 고품질 모델
        output_dir="./data/output"
    )

    # 분리된 오디오 파일명 추출
    audio_filename = av_result.audio_path.name

    session_result = session_separator.separate(
        audio_filename=audio_filename,
        save_all_tracks=True,  # drums, vocals도 저장 (검증용)
        create_mixed=True  # 혼합 트랙 생성 (검증용)
    )

    print("\n✅ 2단계 완료!")

    # ========================================
    # 최종 결과 요약
    # ========================================
    print("\n" + "=" * 70)
    print("📊 전처리 완료 - 최종 결과")
    print("=" * 70)

    print("\n[원본 파일]")
    print(f"   🎬 입력 비디오: {video_file}")

    print("\n[오디오/비디오 분리]")
    print(f"   📹 비디오 (오디오 제거): {av_result.video_path.name}")
    print(f"   🎵 오디오 (원본): {av_result.audio_path.name}")

    print("\n[세션 분리 - 분석용 트랙]")
    print(f"   🎸 베이스 트랙: {session_result.bass_path.name}")
    print(f"   🎸 기타 트랙: {session_result.guitar_path.name}")

    print("\n[세션 분리 - 참고용 트랙]")
    if session_result.drums_path:
        print(f"   🥁 드럼 트랙: {session_result.drums_path.name}")
    if session_result.vocals_path:
        print(f"   🎤 보컬 트랙: {session_result.vocals_path.name}")
    if session_result.mixed_path:
        print(f"   🎵 혼합 트랙 (검증용): {session_result.mixed_path.name}")

    print("\n[다음 단계]")
    print("   → 비디오에서 손가락 위치 추적")
    print("   → 오디오에서 음정 분석")
    print("   → 비디오-오디오 매칭")

    # 분석용 트랙 경로 반환
    analysis_tracks = session_separator.get_analysis_tracks(session_result)

    print("\n[분석에 사용할 파일]")
    for instrument, path in analysis_tracks.items():
        print(f"   {instrument:10s}: {path}")

    print("\n" + "=" * 70)
    print("✅ 전처리 완료! 다음 단계를 진행하세요.")
    print("=" * 70 + "\n")

    return {
        'video_path': av_result.video_path,
        'audio_path': av_result.audio_path,
        'bass_track': session_result.bass_path,
        'guitar_track': session_result.guitar_path,
        'analysis_tracks': analysis_tracks
    }


if __name__ == "__main__":
    try:
        result = main()
    except FileNotFoundError as e:
        print(f"\n❌ 오류: {e}")
        print("   data/input/ 폴더에 example.mp4 파일이 있는지 확인하세요.")
    except Exception as e:
        print(f"\n❌예상치 못한 오류 발생: {e}")
        import traceback

        traceback.print_exc()