from pathlib import Path
import json
from src.config.paths import *


class FilePathManager:
    """입력 파일명 기반으로 파일 경로 관리"""

    def __init__(self, input_file):
        # 파일명에서 확장자 제거
        self.file_name = Path(input_file).stem
        self.file_ext = Path(input_file).suffix

    @property
    def input_video(self):
        """data/input/drowning.mp4"""
        return INPUT_DIR / f"{self.file_name}{self.file_ext}"

    @property
    def processed_video(self):
        """data/processed/videos/drowning_video.mp4"""
        return PROCESSED_VIDEOS_DIR / f"{self.file_name}_video.mp4"

    @property
    def processed_audio(self):
        """data/processed/audios/drowning_audio.wav"""
        return PROCESSED_AUDIOS_DIR / f"{self.file_name}_audio.wav"

    def session_audio(self, session_name):
        """data/processed/sessions/drowning_bass.wav"""
        return PROCESSED_SESSIONS_DIR / f"{self.file_name}_{session_name}.wav"

    @property
    def vision_json(self):
        """data/analysis/vision/drowning_video.json"""
        return ANALYSIS_VISION_DIR / f"{self.file_name}_video.json"

    @property
    def audio_json(self):
        """data/analysis/audio/drowning_audio.json"""
        return ANALYSIS_AUDIO_DIR / f"{self.file_name}_audio.json"

    @property
    def matched_json(self):
        """data/analysis/matched/drowning.json"""
        return ANALYSIS_MATCHED_DIR / f"{self.file_name}.json"

    @property
    def output_musicxml(self):
        """data/output/drowning.musicxml"""
        return OUTPUT_DIR / f"{self.file_name}.musicxml"
