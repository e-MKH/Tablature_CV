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

    # 4단계: CV 처리
    vision_analysis(
        video_path=paths.input_video,
        output_json=paths.vision_json
    )


if __name__ == "__main__":
    main()