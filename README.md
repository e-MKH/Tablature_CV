# Tablature_CV

**Audio-visual transcription of bass guitar performance video into a MusicXML score with tablature positions.**

`Tablature_CV` takes a video of someone playing bass, detects *which notes* were played from the
audio and *where on the fretboard* they were played from the video, then merges both into a
MusicXML score annotated with string/fret positions.

> **Status: runnable.** The full 7-stage pipeline in `main.py` runs end-to-end and produces a
> MusicXML file. This is the main repository of the project; started as a capstone and continued
> as an ongoing team project.

---

## Team

| GitHub |
|---|
| [@dayeong0222](https://github.com/dayeong0222) |
| [@seonghoon00](https://github.com/seonghoon00) |
| [@e-MKH](https://github.com/e-MKH) |
| [@LotteryTicketz](https://github.com/LotteryTicketz) |

---

## What it does

Pitch alone cannot determine fingering: on a 4-string bass the same pitch is playable at several
string/fret combinations. This project resolves that ambiguity by matching each audio note
against the fretting-hand position observed in the video.

- **Audio** decides *what* was played (pitch, onset, duration).
- **Video** decides *where* it was played (string, fret, finger).
- Notes the video cannot confirm are filled in by a position predictor and marked as uncertain.

**Scope of this implementation**
- Instrument: 4-string bass (E–A–D–G, standard tuning).
- The fretting hand is analyzed. **Right-hand (picking) technique is not analyzed.**
- Tempo is **entered by the user**, not estimated (see [Limitations](#known-limitations)).
- Output: MusicXML (rendered with music21). Tab positions are written as note lyrics
  (e.g. `E:5`), not as a native tablature staff.

---

## Tech stack

| Role | Library / Tool |
|---|---|
| A/V separation | ffmpeg |
| Source separation | Demucs (`htdemucs`) |
| Fret localization | YOLO instance segmentation (Ultralytics, `weights.pt`) — only `fret` boxes used at inference |
| Fretboard tracking | OpenCV Lucas–Kanade optical flow |
| Fingertip tracking | MediaPipe Hands + One Euro Filter |
| Pitch & note extraction | librosa (pYIN) |
| Score generation | music21 → MusicXML |
| Path management | pathlib (`FilePathManager`) |

---

## Pipeline

`main.py` runs seven stages in order.

```
mp4 ─▶ ① A/V split ─┬─▶ audio ─▶ ② Demucs ─▶ bass.wav ─▶ ④ pYIN + quantize ─▶ audio.json ─┐
       (ffmpeg)     │                                    (user BPM)                      ├─▶ ⑤ match ─▶ matched.json
                    └─▶ video ─▶ ③ vision (calibrate → pre-scan → track) ─▶ video.json ──┤
                                                                                         └─▶ ⑥ MusicXML ─▶ {name}.musicxml
```

| # | Module | What it does | Key parameters |
|---|---|---|---|
| ① | `preprocessing/separator.py` | Splits the input into a WAV and a silent video | 44.1 kHz, mono, `pcm_s16le`; video stream copied |
| ② | `preprocessing/session_separator.py` | Demucs 4-stem run; keeps `bass` and `other` (saved as `_guitar.wav`) | `htdemucs` |
| ③ | `vision/run_analysis.py` → `bass_system.py` | Fretboard calibration, fret mapping, fingertip → string/fret | see below |
| ④ | `audio_analysis/audio_pipeline.py` | pYIN pitch → note segmentation → beat conversion → quantization | 40–300 Hz, 10 ms step, min note 0.1 s, onset grid 1/16 |
| ⑤ | `matching/match_pipeline.py` | Matches audio notes to video frames and reports statistics | ±0.15 s window, ±1 semitone |
| ⑥ | `notation/bass_musicxml_generator.py` | Builds the score; predicts positions for unmatched notes | 4/4, grid 1/32, predicted notes in red with `?` |

> Stage ⑥ re-runs the matching internally; it reads `video.json` and `audio.json`, not
> `matched.json`. The matched JSON is a statistics/diagnostic artifact.

### Vision stage (③) internals

1. **Manual calibration** — on the first frame the user clicks four fretboard corners
   (nut top, nut bottom, body top, body bottom). `SPACE` confirms, `ESC` cancels.
2. **Pre-scan** — the YOLO segmentation model is run and only its `fret` boxes are kept in four 1-second windows (0 %, 25 %, 50 %, 75 % of the
   video). Fret centers are projected onto the neck axis as ratios and merged into a fret map.
   If nothing is detected, a mathematical fret layout (21 frets) is used instead.
3. **Corner tracking** — the four corners are tracked with Lucas–Kanade optical flow.
   If tracking is lost, analysis **stops at that frame**.
4. **Fingertips** — MediaPipe Hands extracts index/middle/ring/pinky tips of the target hand
   (`TARGET_HAND = "Right"` in `config/vision_config.py`; MediaPipe assumes a mirrored image, so
   this label corresponds to the fretting hand in an unmirrored front-facing video).
   A second, enhanced-preprocessing pass runs if the first detection fails.
   Coordinates are smoothed with a One Euro Filter.
5. **String / fret** — a fingertip within 25 px of a string line selects that string; its
   projected ratio is matched to the fret map (±2 %). Each hit is stored with its MIDI number.
6. **Memory mode** — if the hand is lost, the last valid positions are reused for up to 30 frames
   and flagged `is_tracking: false`.

The analysis window is interactive: `ESC` stops early, `SPACE` pauses for recalibration.

### Vision model: why a segmentation model is used only for frets

`weights.pt` is a YOLO **instance segmentation** model with four classes:
`board`, `fret`, `nut`, `string`. It is the model trained in this project; the current
inference path deliberately uses only one of its classes.

| | |
|---|---|
| **Initial design** | Polygon instance segmentation (rather than bounding boxes) for thin, slanted instrument parts. 650 images — 250 bass images (strings) + 400 guitar images (board/fret) whose images were reused from an existing 6-string set and re-annotated by the team. Per image: board ×1, fret ×20+, string ×4. Frets split into normal/occluded cases, weighted toward occluded in val/test. |
| **Result** | fret 96–98 %, nut 100 %, board 69–75 %, **string 47–48 %** mAP50. Even at `conf = 0.1`, only 2 of 4 strings were found. |
| **Why strings failed** | Strings are 1–3 px wide, so confidence is structurally low; four parallel strings suppress each other under NMS; 1–3 px polygons cannot be annotated consistently. |
| **Final design** | String detection removed from inference. The weights still contain the `string` class, but the pipeline keeps only `fret` boxes. Strings are interpolated geometrically between the four manually clicked corners (`STRING_MARGIN = 0.12`), and Pre-Scan stores fret positions as ratios along the neck. |
| **Trade-off** | String placement no longer depends on string-detection accuracy, but it now depends on the four manually initialized corner points. |

### Matching (⑤) rules

For each audio note: collect video entries within ±0.15 s of the onset → keep those whose
string/fret MIDI is within ±1 semitone of the audio pitch → pick the smallest pitch difference,
preferring `is_tracking: true` on ties. Unmatched notes are assigned by
`notation/position_predictor.py`, which minimizes movement from the previous position
(string changes weighted 5× relative to fret changes).

---

## Repository structure

```
Tablature_CV/
├── main.py                      entry point (7-stage orchestrator)
├── requirements.txt
├── data/
│   ├── input/                   source videos
│   ├── models/                  weights.pt (YOLO segmentation: board/fret/nut/string)
│   ├── processed/               videos/ audios/ sessions/
│   ├── analysis/                vision/ audio/ matched/  (JSON)
│   └── output/                  {name}.musicxml
└── src/
    ├── config/                  constants.py, paths.py, vision_config.py
    ├── preprocessing/           separator.py, session_separator.py
    ├── vision/                  run_analysis, bass_system, calibration, detection,
    │                            tracking, filters, bass_geometry
    ├── audio_analysis/          audio_pipeline, bass_pitch_analyzer, note_detector
    ├── matching/                match_pipeline, video_audio_matcher
    ├── notation/                bass_musicxml_generator, position_predictor, quantizer
    ├── utils/                   file_manager, image_utils, json_utils
    └── legacy/                  earlier single-file versions (final/, audio/, video/)
```

All logic lives in `src/`; `main.py` only calls the stages in order.

---

## Getting started

### Requirements

- **Python 3.12** (Windows, x64)
- **MediaPipe 0.10.14** — installed manually from a wheel (step 2)
- **ffmpeg** binary on `PATH` (step 3)
- A desktop session — stage ③ opens OpenCV windows for calibration and preview

### 1. Python packages

```bash
pip install -r requirements.txt
```

`requirements.txt` pins torch 2.5.0, ultralytics 8.4, demucs 4.0.1, librosa 0.11, music21 9.9, etc.
MediaPipe is intentionally **not** listed; see the next step.

### 2. MediaPipe 0.10.14 (required, manual install)

The vision stage depends on the legacy `mp.solutions.hands` API and is tested with MediaPipe
**0.10.14**. A Windows cp312 build of this version cannot be installed with a plain
`pip install mediapipe==0.10.14`, so install it from the wheel file:

1. Download `mediapipe-0.10.14-cp312-cp312-win_amd64.whl` from either:
   - https://github.com/google-ai-edge/mediapipe/releases/tag/v0.10.14
   - https://pypi.org/project/mediapipe/0.10.14/#files
2. Put the file in `data/package/` (wheel files are git-ignored, so it is not in the repository).
3. Install it:

```bash
pip install data/package/mediapipe-0.10.14-cp312-cp312-win_amd64.whl
```

> Other MediaPipe versions may break the hand-tracking stage.

### 3. ffmpeg (required)

Stage ① calls the `ffmpeg` command directly through `subprocess`, so the ffmpeg **binary** must be
installed on the system and registered on `PATH`.

- Download: https://www.gyan.dev/ffmpeg/builds/ → `ffmpeg-release-full-build` recommended
- Verify:

```bash
ffmpeg -version
```

> The `ffmpeg` entry in `requirements.txt` is a Python package, not the ffmpeg program.
> Installing requirements alone does **not** provide the binary.

### 4. Model weights

Place the YOLO segmentation weights at `data/models/weights.pt` (`.pt` files are git-ignored).

### Usage

```bash
python main.py
# 파일 탐색 (ex: movie.mp4): drowning.mp4
# BPM 입력 (기본값: 120): 111
```

1. Put the video in `data/input/`.
2. Enter the file name and the song's BPM.
3. Click the four fretboard corners in the calibration window, then press `SPACE`.
4. The score is written to `data/output/{name}.musicxml`.

---

## Results

MusicXML was generated end-to-end for all three test clips.

**Note-level coverage** — the share of audio-detected notes for which the vision stream produced
a matching (string, fret) candidate:

| Clip | Coverage |
|---|---|
| Runaway | 93.42 % |
| Drowning | 87.60 % |
| Welcome to the Show | 77.64 % |

- **This is coverage, not accuracy.** It shows whether vision produced a candidate, not whether
  that candidate was the correct string/fret. No ground-truth dataset was available for
  positional accuracy.
- The unmatched notes (6–22 % per clip) are filled in by the movement-minimizing position
  predictor and marked in red with `?`, so predicted positions are never shown as verified.

---

## Known limitations

- **No tempo estimation.** BPM is typed in by the user (default 120) and assumed constant;
  4/4 is fixed.
- **Manual calibration.** Every run requires clicking four corners; recalibration is manual.
- **Tracking loss ends the run.** When optical-flow corner tracking fails, the vision JSON stops
  early while the audio JSON covers the full song. Comparing the two end times is the most
  reliable way to detect this.
- **No `string` detection at inference.** Strings are laid out geometrically between the
  calibrated corners (see [Vision model](#vision-model-why-a-segmentation-model-is-used-only-for-frets)).
- **Silent fret-index shifts.** The neck axis is a linear approximation (no homography), and fret
  numbers come from detection order — one missed low fret shifts every fret number after it
  without raising an error.
- **Picking hand ignored.** No onset, technique, or plucking-string information comes from the
  right hand.
- **Guitar stem unused.** Stage ② saves `other` as `_guitar.wav`, but no guitar transcription
  is performed.

---

## Related repository

**[Tab_CV_Bass](https://github.com/e-MKH/Tab_CV_Bass)** was an experimental rewrite exploring
right-hand technique recognition (slap / pop / fingerstyle). It uses a *separate, later* pair of
models (`board / fret_zone / left_hand / nut` segmentation + a right-hand box detector) that are
not used here. Its usable results have been merged into this repository; what remains there is
unapplied experimental logic kept for reference.
