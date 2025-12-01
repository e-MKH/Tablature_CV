from src.utils.file_manager import FilePathManager
from src.preprocessing.separator import separate_av
from src.preprocessing.session_separator import separate_sessions
from src.audio_analysis.audio_pipeline import analyze_audio
from src.vision.run_analysis import vision_analysis
from src.matching.match_pipeline import match_audio_video
from src.notation.bass_musicxml_generator import generate_bass_musicxml


def main():
    # 1-1. 사용자 입력 받기
    filename = input("파일 탐색 (ex: movie.mp4): ").strip()

    if not filename:
        print("파일 이름이 입력되지 않았습니다.")
        return

    # 1-2.BPM 입력
    bpm_input = input("BPM 입력 (기본값: 120): ").strip()
    bpm = int(bpm_input) if bpm_input else 120

    # 2. FilePathManager 인스턴스 생성
    paths = FilePathManager(filename)

    # 3. 입력 파일 존재 확인
    if not paths.input_video.exists():
        print(f"탐색 실패: {paths.input_video}")
        return

    print(f"탐색 성공: {paths.input_video}\n")

    # 4. 각 단계에서 경로 사용
    # 1-2단계: 오디오/비디오 분리
    separate_av(
        input_video=paths.input_video,
        output_audio=paths.processed_audio,
        output_video=paths.processed_video,
        sample_rate=44100,
        mono=True
    )

    # 3단계: 세션 분리
    separate_sessions(
        audio_path=paths.processed_audio,
        bass_output=paths.session_audio('bass'),
        guitar_output=paths.session_audio('guitar')
    )

    # 4단계: CV 처리
    vision_analysis(
        video_path=paths.processed_video,
        output_json=paths.vision_json
    )

    # 5단계: 오디오 분석
    analyze_audio(
        session_audio_path=paths.session_audio('bass'),
        output_json_path=paths.audio_json,
        bpm=bpm,
        source_path=paths.input_video,
        full_audio_path=paths.processed_audio,
        fmin_hz=40.0,  # 베이스 주파수 범위
        fmax_hz=300.0
    )

    # 6단계: 매칭
    match_audio_video(
        paths.vision_json,
        paths.audio_json,
        paths.matched_json
    )

    # 7단계: MusicXML 생성
    generate_bass_musicxml(
        paths.vision_json,
        paths.audio_json,
        paths.output_musicxml
    )

    print(f"\n변환 완료: {paths.output_musicxml}")


if __name__ == "__main__":
    main()