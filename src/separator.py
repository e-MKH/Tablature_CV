import ffmpeg
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple


@dataclass
class VideoMetadata:
    fps: float
    width: int
    height: int
    duration: float


@dataclass
class AudioMetadata:
    sample_rate: int
    channels: int
    duration: float


@dataclass
class SeparationResult:
    video_path: Path
    audio_path: Path
    video_metadata: VideoMetadata
    audio_metadata: AudioMetadata
    is_synced: bool
    sync_offset: float


class AudioVideoSeparator:
    """FFmpeg 기반 오디오/비디오 분리 및 동기화 검증"""

    def __init__(self, output_dir: str = "./data/temp"):  # 👈 변경 1
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def separate(self, video_path: str) -> SeparationResult:
        input_path = Path(video_path)

        # 👈 변경 2: 파일 이름만 입력된 경우 기본 경로 추가
        if not input_path.is_absolute() and input_path.parent == Path('.'):
            input_path = Path("data/input") / input_path

        if not input_path.exists():
            raise FileNotFoundError(f"비디오 파일을 찾을 수 없습니다: {input_path}")

        base_name = input_path.stem
        output_video = self.output_dir / f"{base_name}_video.mp4"
        output_audio = self.output_dir / f"{base_name}_audio.wav"

        print(f"\n{'=' * 60}")
        print(f"📹 비디오 처리: {input_path.name}")
        print(f"{'=' * 60}")

        # 메타데이터 추출
        video_meta = self._get_video_metadata(input_path)
        print("   📊 비디오 메타데이터 추출 완료")

        # 오디오 추출
        self._extract_audio(input_path, output_audio)
        audio_meta = self._get_audio_metadata(output_audio)
        print("   🎵 오디오 추출 완료")

        # 오디오 제거된 비디오 저장
        self._extract_video(input_path, output_video)
        print("   🎬 오디오 제거 비디오 저장 완료")

        # 동기화 검증
        is_synced, offset = self._verify_sync(video_meta, audio_meta)

        print(f"\n✅ 분리 완료!")
        print(f"   🎬 {output_video.name} ({video_meta.duration:.2f}s)")
        print(f"   🎧 {output_audio.name} ({audio_meta.duration:.2f}s)")
        print(f"   🔄 동기화 상태: {'정상' if is_synced else '⚠ 불일치'} (오프셋: {offset:.3f}s)")
        print(f"{'=' * 60}\n")

        return SeparationResult(
            video_path=output_video,
            audio_path=output_audio,
            video_metadata=video_meta,
            audio_metadata=audio_meta,
            is_synced=is_synced,
            sync_offset=offset,
        )

    def _get_video_metadata(self, video_path: Path) -> VideoMetadata:
        probe = ffmpeg.probe(str(video_path))
        video_stream = next(
            s for s in probe["streams"] if s["codec_type"] == "video"
        )
        fps = eval(video_stream["r_frame_rate"])
        width = int(video_stream["width"])
        height = int(video_stream["height"])
        duration = float(video_stream["duration"])
        return VideoMetadata(fps=fps, width=width, height=height, duration=duration)

    def _get_audio_metadata(self, audio_path: Path) -> AudioMetadata:
        info = sf.info(str(audio_path))
        return AudioMetadata(
            sample_rate=info.samplerate,
            channels=info.channels,
            duration=info.duration,
        )

    def _extract_audio(self, input_path: Path, output_audio: Path):
        (
            ffmpeg
            .input(str(input_path))
            .output(str(output_audio), acodec='pcm_s16le', vn=None, loglevel="error")
            .overwrite_output()
            .run()
        )

    def _extract_video(self, input_path: Path, output_video: Path):
        (
            ffmpeg
            .input(str(input_path))
            .output(str(output_video), c='copy', an=None, loglevel="error")
            .overwrite_output()
            .run()
        )

    def _verify_sync(
            self, video_meta: VideoMetadata, audio_meta: AudioMetadata
    ) -> Tuple[bool, float]:
        diff = abs(video_meta.duration - audio_meta.duration)
        tolerance = 2.0 / video_meta.fps if video_meta.fps > 0 else 0.1
        return diff <= tolerance, diff


if __name__ == "__main__":
    separator = AudioVideoSeparator()
    result = separator.separate("example.mp4")