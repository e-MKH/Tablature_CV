import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Optional


class SessionSeparator:
    """
    Demucs를 사용한 오디오 세션 분리 클래스

    지원하는 stem:
        - bass: 베이스 기타/더블 베이스 등
        - drums: 드럼
        - vocals: 보컬
        - other: 기타 악기 (guitar 포함)
    """

    AVAILABLE_STEMS = ["bass", "drums", "vocals", "other"]
    DEFAULT_MODEL = "htdemucs"

    def __init__(self, model: str = DEFAULT_MODEL):
        """
        Args:
            model: Demucs 모델 이름 (기본: htdemucs)
                - htdemucs: 기본 모델, 4개 stem 분리
                - htdemucs_ft: fine-tuned 버전
                - htdemucs_6s: 6개 stem 분리 (bass, drums, vocals, guitar, piano, other)
        """
        self.model = model

    def separate_stems(
            self,
            audio_path: Path,
            output_dir: Path,
            stems: Optional[List[str]] = None,
            keep_demucs_folder: bool = False
    ) -> Dict[str, Path]:
        """
        오디오 파일을 여러 stem으로 분리

        Args:
            audio_path: 입력 오디오 파일 경로
            output_dir: 출력 디렉터리
            stems: 분리할 stem 리스트 (None이면 모든 stem)
                예: ["bass", "drums"] - bass와 drums만 분리
                예: None - 모든 stem 분리
            keep_demucs_folder: Demucs 원본 폴더 유지 여부 (기본: False)

        Returns:
            {stem_name: stem_path} 딕셔너리
            예: {"bass": Path("output/bass.wav"), "drums": Path("output/drums.wav")}

        Raises:
            FileNotFoundError: 입력 파일이 없는 경우
            ValueError: 지원하지 않는 stem 이름이 있는 경우
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"입력 파일을 찾을 수 없음: {audio_path}")

        # stems 검증
        if stems is not None:
            invalid_stems = set(stems) - set(self.AVAILABLE_STEMS)
            if invalid_stems:
                raise ValueError(
                    f"지원하지 않는 stem: {invalid_stems}\n"
                    f"사용 가능한 stem: {self.AVAILABLE_STEMS}"
                )

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"Audio Session Separation (Demucs)")
        print(f"{'=' * 60}")
        print(f"입력 파일: {audio_path.name}")
        print(f"모델: {self.model}")
        print(f"분리할 stem: {stems if stems else 'all'}")

        # Demucs 명령어 구성
        cmd = ["python", "-m", "demucs", "-n", self.model]

        # 특정 stem만 분리하는 경우
        if stems is not None and len(stems) == 1:
            # two-stems 옵션: bass만, 또는 vocals만 분리 가능
            if stems[0] in ["bass", "vocals"]:
                cmd += [f"--two-stems={stems[0]}"]

        cmd += [str(audio_path), "-o", str(output_dir)]

        print(f"\n[Demucs] 실행 중...")
        print(f"명령어: {' '.join(cmd)}")

        # Demucs 실행
        subprocess.run(cmd, check=True)

        # Demucs 출력 디렉터리 찾기
        demucs_output = output_dir / self.model / audio_path.stem

        if not demucs_output.exists():
            raise FileNotFoundError(
                f"Demucs 출력 폴더를 찾을 수 없음: {demucs_output}"
            )

        # stem 파일들을 편한 위치로 복사
        result_stems = {}
        target_stems = stems if stems else self.AVAILABLE_STEMS

        for stem in target_stems:
            stem_src = demucs_output / f"{stem}.wav"

            if stem_src.exists():
                stem_dst = output_dir / f"{stem}.wav"
                shutil.copy2(stem_src, stem_dst)
                result_stems[stem] = stem_dst
                print(f"[완료] {stem} stem 복사: {stem_dst}")
            else:
                print(f"[경고] {stem} stem을 찾을 수 없음: {stem_src}")

        # Demucs 폴더 정리
        if not keep_demucs_folder:
            shutil.rmtree(output_dir / self.model, ignore_errors=True)
            print(f"\n[정리] Demucs 원본 폴더 삭제")

        print(f"\n{'=' * 60}")
        print(f"세션 분리 완료! (총 {len(result_stems)}개 stem)")
        print(f"{'=' * 60}\n")

        return result_stems

    def separate_bass(self, audio_path: Path, output_dir: Path) -> Path:
        """
        베이스 stem만 분리하는 간편 함수

        Args:
            audio_path: 입력 오디오 파일
            output_dir: 출력 디렉터리

        Returns:
            bass.wav 파일 경로
        """
        result = self.separate_stems(audio_path, output_dir, stems=["bass"])
        return result["bass"]

    def separate_guitar(self, audio_path: Path, output_dir: Path) -> Path:
        """
        기타 stem 분리 (other에 포함됨)

        Note:
            기본 htdemucs 모델은 guitar를 'other' stem에 포함시킴
            guitar를 별도로 분리하려면 'htdemucs_6s' 모델 사용 필요

        Args:
            audio_path: 입력 오디오 파일
            output_dir: 출력 디렉터리

        Returns:
            other.wav (또는 guitar.wav) 파일 경로
        """
        if self.model == "htdemucs_6s":
            # 6-stem 모델은 guitar를 별도 분리
            result = self.separate_stems(audio_path, output_dir, stems=["guitar"])
            return result["guitar"]
        else:
            # 기본 모델은 other에 포함
            print("[정보] 기본 모델은 guitar를 'other' stem에 포함시킵니다")
            print("       guitar 별도 분리를 원하시면 'htdemucs_6s' 모델을 사용하세요")
            result = self.separate_stems(audio_path, output_dir, stems=["other"])
            return result["other"]

    def separate_all(self, audio_path: Path, output_dir: Path) -> Dict[str, Path]:
        """
        모든 stem 분리

        Args:
            audio_path: 입력 오디오 파일
            output_dir: 출력 디렉터리

        Returns:
            {stem_name: stem_path} 딕셔너리
        """
        return self.separate_stems(audio_path, output_dir, stems=None)


# ============================================================================
# 편의 함수들
# ============================================================================

def separate_bass_stem(
        audio_path: Path,
        output_dir: Path,
        model: str = SessionSeparator.DEFAULT_MODEL
) -> Path:
    """
    베이스 stem만 빠르게 분리

    Args:
        audio_path: 입력 오디오 파일
        output_dir: 출력 디렉터리
        model: Demucs 모델 (기본: htdemucs)

    Returns:
        bass.wav 파일 경로
    """
    separator = SessionSeparator(model=model)
    return separator.separate_bass(audio_path, output_dir)


def separate_guitar_stem(
        audio_path: Path,
        output_dir: Path,
        model: str = SessionSeparator.DEFAULT_MODEL
) -> Path:
    """
    기타 stem 분리

    Args:
        audio_path: 입력 오디오 파일
        output_dir: 출력 디렉터리
        model: Demucs 모델 (기본: htdemucs)

    Returns:
        other.wav (또는 guitar.wav) 파일 경로
    """
    separator = SessionSeparator(model=model)
    return separator.separate_guitar(audio_path, output_dir)


def separate_all_stems(
        audio_path: Path,
        output_dir: Path,
        model: str = SessionSeparator.DEFAULT_MODEL
) -> Dict[str, Path]:
    """
    모든 stem 분리

    Args:
        audio_path: 입력 오디오 파일
        output_dir: 출력 디렉터리
        model: Demucs 모델 (기본: htdemucs)

    Returns:
        {stem_name: stem_path} 딕셔너리
    """
    separator = SessionSeparator(model=model)
    return separator.separate_all(audio_path, output_dir)


def separate_sessions(
        audio_path: Path,
        bass_output: Path,
        guitar_output: Path,
        model: str = SessionSeparator.DEFAULT_MODEL
):
    """
    오디오 파일을 베이스와 기타 세션으로 분리 (main.py용 함수)

    이 함수는 main.py에서 직접 호출하기 위한 간편 인터페이스입니다.
    임시 디렉토리에서 분리 후 지정된 경로로 파일을 이동합니다.

    Args:
        audio_path: 입력 오디오 파일 경로
        bass_output: 베이스 출력 경로
        guitar_output: 기타 출력 경로
        model: Demucs 모델 (기본: htdemucs)

    Example:
        from preprocessing.session_separator import separate_sessions

        separate_sessions(
            audio_path=Path("data/processed/audio.wav"),
            bass_output=Path("data/sessions/bass.wav"),
            guitar_output=Path("data/sessions/guitar.wav")
        )
    """
    # 임시 디렉토리 생성
    temp_dir = audio_path.parent / "temp_stems"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # SessionSeparator로 bass와 other(guitar) 분리
        separator = SessionSeparator(model=model)
        stems = separator.separate_stems(
            audio_path=audio_path,
            output_dir=temp_dir,
            stems=["bass", "other"],
            keep_demucs_folder=False
        )

        # 출력 디렉토리 생성
        bass_output.parent.mkdir(parents=True, exist_ok=True)
        guitar_output.parent.mkdir(parents=True, exist_ok=True)

        # 지정된 경로로 이동
        shutil.move(str(stems["bass"]), str(bass_output))
        shutil.move(str(stems["other"]), str(guitar_output))

        print(f"\n[이동 완료] 베이스: {bass_output}")
        print(f"[이동 완료] 기타: {guitar_output}")

    finally:
        # 임시 디렉토리 삭제
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"[정리] 임시 디렉토리 삭제: {temp_dir}")


# 사용 예시
if __name__ == "__main__":
    # 예시 1: 베이스만 분리
    # bass_path = separate_bass_stem(
    #     audio_path=Path("data/processed/full_audio.wav"),
    #     output_dir=Path("data/processed/stems")
    # )

    # 예시 2: 클래스 사용 - 여러 stem 분리
    # separator = SessionSeparator(model="htdemucs")
    # stems = separator.separate_stems(
    #     audio_path=Path("data/processed/full_audio.wav"),
    #     output_dir=Path("data/processed/stems"),
    #     stems=["bass", "drums", "other"]
    # )

    # 예시 3: main.py에서 사용하는 방식
    # separate_sessions(
    #     audio_path=Path("data/processed/audio.wav"),
    #     bass_output=Path("data/sessions/bass.wav"),
    #     guitar_output=Path("data/sessions/guitar.wav")
    # )

    pass