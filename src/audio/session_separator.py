import torch
import torchaudio
from pathlib import Path
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from demucs.pretrained import get_model
from demucs.apply import apply_model
import warnings

warnings.filterwarnings('ignore')


@dataclass
class SeparationResult:
    """세션 분리 결과"""
    bass_path: Path  # 베이스 트랙 (분석용)
    guitar_path: Path  # 기타 트랙 = other (분석용)
    drums_path: Optional[Path] = None  # 드럼 (참고용)
    vocals_path: Optional[Path] = None  # 보컬 (참고용)
    mixed_path: Optional[Path] = None  # bass+other 혼합 (검증용, 분석 금지!)


class SessionSeparator:
    """
    Demucs 기반 세션 분리 모듈

    기타와 베이스를 개별 트랙으로 분리 (정확한 구분 필수)
    - bass.wav: 베이스 트랙 (분석용)
    - guitar.wav: 기타 트랙 = other (분석용)
    - drums/vocals: 참고용
    - mixed: 검증용 (분석 사용 금지!)
    """

    def __init__(
            self,
            model_name: str = "htdemucs",
            device: Optional[str] = None,
            output_dir: str = "./data/temp"
    ):
        """
        Args:
            model_name: Demucs 모델 (htdemucs, htdemucs_ft, mdx_extra)
            device: 'cuda' 또는 'cpu'
            output_dir: 출력 디렉토리
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 디바이스 설정
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"\n🔧 Demucs 세션 분리기 초기화")
        print(f"{'=' * 60}")
        print(f"   모델: {model_name}")
        print(f"   디바이스: {self.device}")

        # 모델 로드
        self.model = get_model(model_name)
        self.model.to(self.device)
        self.model.eval()

        # 소스 목록
        self.sources = self.model.sources
        print(f"   분리 트랙: {', '.join(self.sources)}")
        print(f"✅ 초기화 완료!")
        print(f"{'=' * 60}\n")

    def separate(
            self,
            audio_filename: str,
            save_all_tracks: bool = True,
            create_mixed: bool = True
    ) -> SeparationResult:
        """
        오디오를 기타/베이스로 분리

        ⚠️ 중요: 분석에는 개별 트랙(bass, guitar)만 사용!

        Args:
            audio_filename: 오디오 파일명 (예: "example_audio.wav")
            save_all_tracks: drums, vocals도 저장 (참고용)
            create_mixed: bass+other 혼합 트랙 생성 (검증용, 분석 사용 금지)

        Returns:
            SeparationResult: 분리된 트랙 경로들
        """
        audio_path = self.output_dir / audio_filename

        if not audio_path.exists():
            raise FileNotFoundError(f"오디오 파일을 찾을 수 없습니다: {audio_path}")

        print(f"\n{'=' * 60}")
        print(f"🎸 세션 분리 시작: {audio_filename}")
        print(f"{'=' * 60}")

        # 오디오 로드
        print(f"   📂 오디오 로딩...")
        wav, sr = torchaudio.load(str(audio_path))

        # 스테레오 변환
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)

        wav = wav.to(self.device)

        # Demucs 분리
        print(f"   🔬 Demucs 분리 중... (시간 소요)")
        with torch.no_grad():
            sources = apply_model(
                self.model,
                wav.unsqueeze(0),
                device=self.device,
                split=True,
                overlap=0.25
            )[0]

        sources = sources.cpu()

        # 출력 파일명
        base_name = audio_path.stem.replace('_audio', '')

        # 인덱스 매핑
        idx_bass = self.sources.index('bass')
        idx_other = self.sources.index('other')
        idx_drums = self.sources.index('drums')
        idx_vocals = self.sources.index('vocals')

        print(f"\n   💾 트랙 저장 중...")

        # 1. 베이스 트랙 (필수)
        bass_path = self.output_dir / f"{base_name}_bass.wav"
        torchaudio.save(str(bass_path), sources[idx_bass], sr)
        print(f"      ✓ 베이스: {bass_path.name}")

        # 2. 기타 트랙 (other) (필수)
        guitar_path = self.output_dir / f"{base_name}_guitar.wav"
        torchaudio.save(str(guitar_path), sources[idx_other], sr)
        print(f"      ✓ 기타: {guitar_path.name}")

        # 3. 혼합 트랙 (bass + other) (검증용만!)
        mixed_path = None
        if create_mixed:
            mixed_path = self.output_dir / f"{base_name}_bass_guitar_mixed.wav"
            mixed_audio = sources[idx_bass] + sources[idx_other]

            # 클리핑 방지 (정규화)
            max_val = torch.abs(mixed_audio).max()
            if max_val > 1.0:
                mixed_audio = mixed_audio / max_val

            torchaudio.save(str(mixed_path), mixed_audio, sr)
            print(f"      ✓ 혼합 (검증용): {mixed_path.name}")

        # 4. 기타 트랙들 (참고용)
        drums_path = None
        vocals_path = None

        if save_all_tracks:
            drums_path = self.output_dir / f"{base_name}_drums.wav"
            vocals_path = self.output_dir / f"{base_name}_vocals.wav"

            torchaudio.save(str(drums_path), sources[idx_drums], sr)
            torchaudio.save(str(vocals_path), sources[idx_vocals], sr)

            print(f"      ✓ 드럼 (참고): {drums_path.name}")
            print(f"      ✓ 보컬 (참고): {vocals_path.name}")

        # 에너지 분석 (어느 트랙에 소리가 있는지)
        print(f"\n   📊 트랙 에너지 분석:")
        energies = self._analyze_energy(sources)

        for source_name, energy in zip(self.sources, energies):
            emoji = "🔊" if energy > 0.01 else "🔇"
            print(f"      {emoji} {source_name:8s}: {energy:.4f}")

        # 경고 메시지
        if energies[idx_drums] > 0.05 or energies[idx_vocals] > 0.05:
            print(f"\n   ⚠️  경고: drums 또는 vocals 트랙에 큰 신호 감지!")
            print(f"      기타/베이스 외 악기가 포함되어 있을 수 있습니다.")
            print(f"      → 분리 품질이 낮을 수 있으니 원본 오디오를 확인하세요.")

        print(f"\n✅ 세션 분리 완료!")
        print(f"{'=' * 60}\n")

        return SeparationResult(
            bass_path=bass_path,
            guitar_path=guitar_path,
            drums_path=drums_path,
            vocals_path=vocals_path,
            mixed_path=mixed_path
        )

    def _analyze_energy(self, sources: torch.Tensor) -> list[float]:
        """각 트랙의 에너지 계산 (RMS)"""
        energies = []
        for i in range(sources.shape[0]):
            rms = torch.sqrt(torch.mean(sources[i] ** 2)).item()
            energies.append(rms)
        return energies

    def get_analysis_tracks(self, result: SeparationResult) -> Dict[str, Path]:
        """
        분석에 사용할 트랙 반환 (개별 분리만)

        ⚠️ 중요: 기타와 베이스를 정확히 구분해야 하므로 개별 트랙만 사용!

        Args:
            result: 분리 결과

        Returns:
            {'bass': Path, 'guitar': Path}
        """
        return {
            'bass': result.bass_path,
            'guitar': result.guitar_path
        }

    def get_verification_track(self, result: SeparationResult) -> Path:
        """
        검증용 혼합 트랙 반환

        분석에 사용하지 말고, 디버깅/검증 목적으로만 사용!

        Args:
            result: 분리 결과

        Returns:
            혼합 트랙 경로
        """
        if result.mixed_path is None:
            raise ValueError("혼합 트랙이 생성되지 않았습니다. create_mixed=True로 설정하세요.")
        return result.mixed_path


if __name__ == "__main__":
    # 사용 예시
    separator = SessionSeparator(
        model_name="htdemucs",
        output_dir="./data/temp"
    )

    # 세션 분리
    result = separator.separate(
        audio_filename="example_audio.wav",
        save_all_tracks=True,  # drums, vocals도 저장 (검증용)
        create_mixed=True  # bass+other 혼합 트랙 생성 (검증용)
    )

    print("\n📁 분리된 파일:")
    print(f"   베이스: {result.bass_path}")
    print(f"   기타: {result.guitar_path}")
    if result.drums_path:
        print(f"   드럼 (참고): {result.drums_path}")
    if result.vocals_path:
        print(f"   보컬 (참고): {result.vocals_path}")
    if result.mixed_path:
        print(f"   혼합 (검증용): {result.mixed_path}")

    # 분석용 트랙 (필수: 개별 분리만 사용)
    print("\n🎯 분석용 트랙:")
    tracks = separator.get_analysis_tracks(result)
    for name, path in tracks.items():
        print(f"   {name}: {path.name}")

    print("\n⚠️  주의: 기타/베이스 구분이 핵심이므로 개별 트랙만 사용!")
    print("   혼합 트랙은 검증/디버깅 목적으로만 사용하세요.")