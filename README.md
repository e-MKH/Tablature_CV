# Tablature_CV

베이스 기타 연주 영상을 분석하여 자동으로 악보(타블러처)를 생성하는 컴퓨터 비전 + 오디오 분석 파이프라인입니다.

---

## 개요

Tablature_CV는 베이스 기타 연주 영상을 입력으로 받아, 영상에서 손가락 위치를 추적하고 오디오에서 음정을 분석하여 이를 통합한 MusicXML 악보 파일을 생성합니다.

- **입력**: 베이스 기타 연주 영상 (`.mp4` 등)
- **출력**: 타블러처 정보가 포함된 악보 (`.musicxml`)

---

## 주요 기술 스택

| 역할 | 라이브러리 |
|---|---|
| A/V 분리 | ffmpeg |
| 세션(악기) 분리 | Demucs (htdemucs) |
| 객체 탐지 (프렛) | YOLOv11 (Ultralytics) |
| 손 추적 | MediaPipe |
| 피치 분석 | librosa (pyin) |
| 악보 생성 | music21 |
| 영상 처리 | OpenCV |

---

## 프로젝트 구조

```
Tablature_CV/
├── main.py                          # 전체 파이프라인 실행
├── main2.py                         # Vision 단독 테스트
├── requirements.txt
└── src/
    ├── config/
    │   ├── paths.py                 # 데이터 디렉토리 경로 상수
    │   ├── constants.py             # BPM, 박자표 등 기본 상수
    │   └── vision_config.py        # Vision 파라미터 설정
    ├── preprocessing/
    │   ├── separator.py             # A/V 분리 (ffmpeg)
    │   └── session_separator.py    # 악기 세션 분리 (Demucs)
    ├── audio_analysis/
    │   ├── audio_pipeline.py        # 오디오 분석 파이프라인
    │   ├── bass_pitch_analyzer.py   # pyin 피치 분석
    │   └── note_detector.py         # 음표 검출
    ├── vision/
    │   ├── run_analysis.py          # Vision 분석 파이프라인
    │   ├── bass_system.py           # BassTabSystem 통합 클래스
    │   ├── detection.py             # YOLO 프렛 탐지, MediaPipe 손 탐지
    │   ├── tracking.py              # 코너 추적 (Optical Flow)
    │   ├── calibration.py           # 캘리브레이션 UI
    │   ├── bass_geometry.py         # 현/프렛 좌표 계산
    │   └── filters.py               # One Euro Filter
    ├── matching/
    │   ├── match_pipeline.py        # 매칭 파이프라인
    │   └── video_audio_matcher.py   # 타임스탬프 기반 매칭
    ├── notation/
    │   ├── bass_musicxml_generator.py  # MusicXML 생성
    │   ├── position_predictor.py       # 포지션 예측 (매칭 실패 시)
    │   └── quantizer.py                # 박자 양자화
    └── utils/
        ├── file_manager.py          # 경로 관리 (FilePathManager)
        ├── image_utils.py           # 이미지 유틸
        └── json_utils.py            # JSON 유틸
```

### 데이터 디렉토리 구조

```
data/
├── input/                  # 입력 영상 파일
├── processed/
│   ├── videos/             # 오디오 제거된 영상
│   ├── audios/             # 추출된 전체 오디오
│   └── sessions/           # 악기별 분리 오디오 (bass, other)
├── analysis/
│   ├── vision/             # Vision 분석 결과 JSON
│   ├── audio/              # 오디오 분석 결과 JSON
│   └── matched/            # 매칭 결과 JSON
├── models/
│   └── weights.pt          # YOLO 모델 가중치
└── output/                 # 최종 MusicXML 출력
```

---

## 설치

**Python 3.12 권장**

### 1. 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. MediaPipe 별도 설치 (필수)

MediaPipe 0.10.14는 Windows cp312 빌드가 PyPI에 등록되어 있지 않아 whl 파일을 직접 다운로드하여 설치해야 합니다.

1. 아래 링크에서 `mediapipe-0.10.14-cp312-cp312-win_amd64.whl` 파일을 다운로드합니다.
   - https://github.com/google-ai-edge/mediapipe/releases/tag/v0.10.14
   - https://pypi.org/project/mediapipe/0.10.14/#files
2. 다운로드한 whl 파일을 `data/package/` 폴더에 넣습니다.
3. 아래 명령으로 설치합니다.
```bash
pip install data/package/mediapipe-0.10.14-cp312-cp312-win_amd64.whl
```

> 다른 버전의 MediaPipe를 사용할 경우 호환성 문제가 발생할 수 있습니다.

### 3. ffmpeg 설치

ffmpeg은 시스템에 별도로 설치 및 PATH 등록이 필요합니다.

- 다운로드: https://www.gyan.dev/ffmpeg/builds/ → `ffmpeg-release-full-build` 권장

---

## 사용법

1. `data/input/` 폴더에 분석할 베이스 기타 연주 영상을 넣습니다.
2. `data/models/weights.pt`에 YOLO 모델 가중치 파일을 준비합니다.
3. 아래 명령으로 실행합니다.

```bash
python main.py
```

4. 파일명과 BPM을 입력합니다.

```
파일 탐색 (ex: movie.mp4): my_bass_video.mp4
BPM 입력 (기본값: 120): 95
```

5. Vision 분석이 시작되면 캘리브레이션 UI가 열립니다. 영상에서 **베이스 넥의 4개 코너(너트~마지막 프렛, 위아래)**를 클릭하여 지정합니다.
6. 분석이 완료되면 `data/output/` 폴더에 `.musicxml` 파일이 생성됩니다.

### 실행 중 단축키

| 키 | 동작 |
|---|---|
| `ESC` | 분석 중단 |
| `SPACE` | 캘리브레이션 재설정 |

---

## 전체 파이프라인

```
[입력: mp4]
    │
    ▼
① A/V 분리 (ffmpeg)
    │── processed/videos/ (영상)
    └── processed/audios/ (전체 오디오 wav)
    │
    ▼
② 세션 분리 (Demucs htdemucs)
    └── processed/sessions/ (bass.wav, other.wav)
    │
    ▼
③ Vision 분석 (YOLO + MediaPipe + Optical Flow)
    │  - 사용자 캘리브레이션으로 넥 영역 지정
    │  - Pre-Scan으로 프렛 위치 학습
    │  - 프레임마다 손가락 끝 좌표 추적
    │  - One Euro Filter로 좌표 노이즈 제거
    │  - 현 번호 / 프렛 번호 계산
    └── analysis/vision/*.json
    │
    ▼
④ 오디오 분석 (librosa pyin)
    │  - 피치(주파수) 분석 → MIDI 변환
    │  - 연속 프레임 병합 → 음표 리스트
    │  - BPM 기반 beat 위치 계산
    │  - 16분음표 그리드 양자화
    └── analysis/audio/*.json
    │
    ▼
⑤ 비디오-오디오 매칭
    │  - 타임스탬프 기반 vision ↔ audio 매칭
    └── analysis/matched/*.json
    │
    ▼
⑥ MusicXML 생성 (music21)
    │  - 오디오 음표 기준으로 vision 포지션 통합
    │  - 매칭 실패 시 마지막 포지션 기반 예측 (빨간색 표시)
    │  - 쉼표 자동 삽입
    └── output/*.musicxml
```

---

## 출력 포맷

생성된 `.musicxml` 파일은 MuseScore, Guitar Pro 등 악보 프로그램에서 열 수 있습니다.

- 각 음표에 `현:프렛` 형태의 탭 정보가 가사(Lyric)로 포함됩니다. (예: `A:3`)
- Vision에서 추적하지 못해 예측된 포지션은 **빨간색**으로 표시됩니다.
- 손가락 번호 정보가 있을 경우 함께 표기됩니다.

---

## 주요 파라미터

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| BPM | 120 | 곡의 빠르기 (사용자 입력) |
| fmin_hz | 40.0 Hz | 피치 분석 최소 주파수 (베이스 범위) |
| fmax_hz | 300.0 Hz | 피치 분석 최대 주파수 |
| onset_grid | 0.25 beat | 음표 시작 양자화 단위 (16분음표) |
| Demucs 모델 | htdemucs | 4-stem 분리 모델 |

---

## 참고 사항

- 손 감지 실패 시, 설정된 허용 프레임 수 이내라면 마지막 유효 포지션을 재사용합니다.
- 현재 4현 베이스 기타를 기준으로 설계되었습니다.
- YOLO 가중치(`weights.pt`)는 별도로 학습하거나 준비해야 합니다.
