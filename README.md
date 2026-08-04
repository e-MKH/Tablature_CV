# Tab_CV_Bass

**Audio-visual transcription of bass guitar performance video into tablature.**

`Tab_CV_Bass` takes a video of someone playing bass and reconstructs what was played — not
just *which notes*, but *where and how* they were played on the instrument. It combines
YOLOv11 instance segmentation, MediaPipe hand tracking, and a Demucs + librosa audio chain.

> **Status: research in progress.**
> The audio pipeline is implemented and verified end-to-end. The right-hand vision pipeline
> is implemented but **has never been executed**. The project's central research question —
> whether right-hand playing technique can be classified from video — is still open.

---

## Why vision?

Pitch alone cannot determine fingering. On a 4-string bass the same pitch is playable at
several string/fret combinations, so an audio-only transcriber can output *a* note but not
*the* tab position the player actually used. Video supplies the missing constraint.

The predecessor project (`Tablature_CV`, a capstone) proved this pipeline shape works, but
its right-hand analysis only detected onsets — it could not tell **slap** from **pop** from
**fingerstyle**. `Tab_CV_Bass` is a full rewrite built around that unsolved problem.

**Core research question:** *Can right-hand playing technique (slap / pop / fingerstyle) be
detected from video?* The answer determines the rest of the architecture. If it succeeds,
audio is demoted from "the thing that decides pitch" to "a verification channel."

**Scope**
- 4-string bass (confirmed). 5+ string support under consideration. Guitar support removed.
- Primary output: tablature. `.mscz` bass tab export is optional, not a priority.

---

## Team

| GitHub |
|---|
| [@dayeong0222](https://github.com/dayeong0222) |
| [@seonghoon00](https://github.com/seonghoon00) |
| [@e-MKH](https://github.com/e-MKH) |
| [@LotteryTicketz](https://github.com/LotteryTicketz) |

---

## Tech stack

| Role | Library / Tool |
|---|---|
| A/V extraction | ffmpeg |
| Source separation | Demucs (`htdemucs`, `--two-stems=bass`) |
| Instrument segmentation | YOLOv11 instance segmentation (Ultralytics) |
| Right-hand detection | YOLO detection (Ultralytics) |
| Fingertip tracking | MediaPipe Hands |
| Tempo / beat tracking | librosa (HPSS + `beat_track`) |
| Pitch & note extraction | librosa (pYIN) |
| Video / geometry processing | OpenCV |
| Annotation | Roboflow (polygon, modal masking) |
| Path management | pathlib (`FilePathManager`) |
| Score generation | *planned* — music21 / tablature renderer |

Trained classes (`weights_bass.pt`): `board`, `fret_zone`, `left_hand`, `nut`.
`string` is deliberately **not** a class — see [Design decisions](#design-decisions).

---

## Current status

| Area | State |
|---|---|
| Audio pipeline (stages 1–5) | ✅ Implemented and verified on real footage |
| Vision pipeline (stages 6–7) | ⚠️ Implemented, **0 executions**, heuristics unvalidated |
| Trained weights (`weights_bass.pt`, `weights_hand.pt`) | ✅ Available |
| Annotation & augmentation methodology | ✅ Established and documented |
| Right-hand technique classifier | ❌ Not started — *the core problem* |
| Left-hand fret detection | ❌ Not started (`left_hand` class trained, no consumer code) |
| Audio ↔ vision temporal fusion | ❌ Not started |
| Tablature rendering | ❌ Not started (fingering optimizer parked in `legacy/core_change/`) |

---

## Pipeline

```
input                preprocessing            core processing                    output
mp4 ──▶ ① audio extract ──▶ ② Demucs ──▶ ③ BPM ──▶ ④ pYIN notes ──────▶ ⑤ {name}_audio_final.json
  │        (ffmpeg)         (two-stems)   (HPSS)    (C1–C3, grouped)
  └──────────────────────────────────▶ ⑥ right-hand vision ──────────▶ ⑦ {name}_vision_final.json
                                          (YOLO ×2 + MediaPipe)
```

| # | Module | Tech | Key parameters |
|---|---|---|---|
| ① | `legacy/separator.py` | ffmpeg subprocess | 48 kHz, mono, `pcm_s16le` |
| ② | `audio/demucs_runner.py` | Demucs `htdemucs` | `--two-stems=bass`, `--shifts=10`, result caching, `TORCHAUDIO_USE_TORCHCODEC=0` |
| ③ | `audio/bpm_estimator.py` | librosa HPSS + `beat_track` | `tightness=50`, fixed tempo, no octave correction |
| ④ | `audio/note_extractor.py` | `librosa.pyin` | `fmin=C1`, `fmax=C3`, min duration 0.05 s, MIDI grouping |
| ⑤ | `audio/report_writer.py` | JSON | metadata + `beat_grid` + `notes` |
| ⑥ | `vision/right_hand_pipeline.py` | YOLOv11 seg + YOLO det + MediaPipe Hands | `conf=0.3`, left-quarter hand rule, `STRING_RATIOS`, 35 px threshold, 20 px ROI pad, ≥3 `fret_zone` required |
| ⑦ | `vision/report_writer.py` | JSON | metadata + per-frame records |

**Not in the pipeline yet:** technique classification (expected between ⑥ and ⑦), left-hand
fret stage, ⑤+⑦ temporal fusion, tab rendering.

### Vision stage internals

1. **Fretboard** — segment `nut` and `fret_zone`. Each zone's contour → convex hull →
   `minAreaRect` → 4 corners sorted near/far and top/bottom relative to the nut
   (`smart_sort_and_label_v6`).
2. **String lines** — all top corners and all bottom corners are fitted separately with
   `cv2.fitLine(DIST_HUBER)`, giving a top and a bottom reference line.
3. **Virtual strings** — the four strings are interpolated between those lines at
   `STRING_RATIOS = [0.9, 2.85, 4.85, 6.8] / 8`. No string is ever detected directly.
4. **Right hand** — a `weights_hand.pt` box whose center falls in the **left quarter** of the
   frame is taken as the picking hand; MediaPipe Hands runs inside that ROI and returns five
   fingertips (thumb, index, middle, ring, pinky).
5. **Candidate strings** — any fingertip within **35 px** of a virtual string line registers
   that string as a candidate for the frame.

There is no temporal smoothing. The only stabilization is carry-forward state: `last_nut`
persists when the nut is missed, and the previous string lines persist when fewer than three
`fret_zone` instances are detected in a frame.

---

## Repository structure

```
Tab_CV_Bass/
├── main.py                      ★ the only entry point (7-stage orchestrator)
├── requirements.txt             ⚠️ currently unreliable — see Known issues
├── data/
│   ├── input/                   source videos
│   ├── models/                  weights_bass.pt (seg), weights_hand.pt (right-hand box)
│   ├── output/                  wav / json / mp4 artifacts (git-ignored)
│   └── reference/               research papers (current + old/)
└── src/
    ├── audio/                   demucs_runner, bpm_estimator, note_extractor, report_writer
    ├── vision/                  fretboard_detector, hand_detector, right_hand_pipeline, report_writer
    ├── utils/                   file_path_manager (pathlib-based path resolution)
    ├── test/                    standalone validation scripts (the one place direct execution is allowed)
    └── legacy/                  archived experiments and preserved alternatives
        └── core_change/         originals of the modules merged in 2026-06
```

**Architecture rule (non-negotiable):** `main.py` only orchestrates. All logic lives in
`src/`, and no module under `src/` is meant to be run on its own — `src/test/` is the sole
exception.

---

## Getting started

### Requirements

- Python 3.11+ (developed on Miniconda, Windows)
- **ffmpeg** available on `PATH`
- PyTorch (current dev environment: torch 2.10.0 CPU / torchaudio 2.10.0 / torchvision 0.25.0)

Core packages: `ultralytics`, `opencv-python`, `mediapipe`, `demucs`, `librosa`, `soundfile`,
`numpy`, `torch`.

> `requirements.txt` is currently **not usable** — it was frozen mid-experiment, is saved in
> UTF-16, contains local `file://` paths and abandoned experiment artifacts, and is missing
> `demucs`. Install the packages above manually until it is regenerated.

### Installation

```bash
# 1. dependencies
pip install ultralytics opencv-python mediapipe demucs librosa soundfile

# 2. ffmpeg — must be callable as `ffmpeg` from the shell
ffmpeg -version

# 3. weights — place both files in data/models/
#    weights_bass.pt   (YOLOv11 instance segmentation: board / fret_zone / left_hand / nut)
#    weights_hand.pt   (YOLO detection: right-hand bounding box)
```

### Usage

```bash
python main.py
# > 파일 탐색 (ex: video.mp4):  drowning.mp4
```

Place the video in `data/input/` first. Both reports are written to `data/output/`.

### Output

`{name}_audio_final.json`

```json
{
  "metadata": { "file": "drowning", "bpm": 128.0, "analysis_date": "2026-08-04" },
  "beat_grid": [0.51, 0.98, 1.45],
  "notes": [{ "start_s": 1.2, "end_s": 1.45, "pitch": 43, "velocity": 0.87 }]
}
```

`{name}_vision_final.json`

```json
{
  "metadata": { "file": "drowning", "fps": 30.0, "total_frames": 4500, "analysis_date": "2026-08-04" },
  "frames": [
    {
      "frame": 120,
      "time_s": 4.0,
      "hand_box": [210.4, 388.1, 402.7, 566.9],
      "fingertips": { "thumb": [268, 431], "index": [301, 470] },
      "candidate_strings": [3, 4]
    }
  ]
}
```

`hand_box` is `null` and `fingertips` is empty when no picking hand is found in the frame.

---

## Design decisions

Each of these was made against measured failure, not preference.

**No `string` class.** Direct string segmentation was trained in the predecessor project and
reached only **47–48 % mAP50** — at `conf=0.1` it still found just 2 of 4 strings. Three
causes: thin objects produce weak confidence, parallel strings suppress each other under NMS,
and 1–3 px polygon annotation cannot be labeled consistently. Rather than fight it, strings
were removed from the class set and are now derived geometrically from the classes that *do*
detect reliably (board/nut/fret at 96–100 %). Residual error is meant to be corrected
downstream by audio pitch matching.

**Modal masking, not amodal.** Labeling occluded regions would teach the model that skin
pixels are instrument pixels. Amodal is also impractical for thin linear objects, YOLO has no
unknown-pixel label, and occluded geometry can be recovered in post-processing instead.

**Photometric augmentation only.** Rotation, blur, and noise destroy thin linear features and
break the spatial relationships between instrument components. Only brightness, exposure,
saturation, and hue are used, at a conservative 2× (1,001 source images → 1,702 train).

**Zones, not frets.** Following the TapToTab approach: the question is *which fret interval a
finger occupies*, not the exact pixel position of a fret wire. This simplifies both labeling
and inference.

**YOLO + MediaPipe hybrid for the right hand.** YOLO answers "where is the picking hand"
(context, robust to motion), MediaPipe answers "where exactly are the fingertips." Right-hand
annotation is bounding-box only, on the hypothesis that position, angle, and coarse pose
carry enough signal to separate techniques.

**Simplification with a preserved recovery path.** The June 2026 integration deliberately
dropped 4-stem separation, BPM octave correction, and OneEuroFilter smoothing to get a thin
pipeline running end-to-end. Every dropped component is preserved in runnable form under
`src/legacy/` and `src/test/` so it can be restored if measurement demands it.

---

## Known issues and unverified assumptions

**Blocking**
- The vision pipeline has never run. `weights_bass.pt`'s `model.names` must be confirmed to
  contain exactly `'nut'` and `'fret_zone'` before the first execution.
- `main.py` imports `extract_audio_from_video` from `src.legacy.separator` — a structural
  violation. `separator.py` should be promoted to `src/audio/` or a new `src/preprocessing/`.
- `requirements.txt` needs a full rewrite.

**Unvalidated heuristics** (all three are first-run checks)
- `STRING_RATIOS` was ported from an older project's empirical tuning and has not been
  re-validated on current footage, instruments, or camera angles.
- "Left quarter of the frame = picking hand" assumes a frontal camera setup.
- The 35 px candidate-string threshold, and whether unsmoothed per-frame fingertips are
  stable enough at all. If jitter appears, OneEuroFilter is the first thing to restore.

**Latent risks**
- BPM octave correction was removed, so half-tempo errors (e.g. 83 reported for a ~166 BPM
  track) can recur. The corrected implementation is preserved at `legacy/librosa_beat.py`.
- Fretboard analysis depends on `last_nut`; if the nut is never detected in the opening
  frames, every downstream stage silently produces nothing for those frames.
- Variable tempo is unsupported — fixed BPM is assumed. `beat_this` (ISMIR 2024) is the
  reserved solution; `madmom` was rejected for Python 3.11 incompatibility.
- Quantitative metrics for the current segmentation model have not been recorded.

---

## Roadmap

1. **First full run** of `main.py` on `drowning.mp4`; validate the three vision heuristics
   against real output.
2. **Qualitative analysis** of fingertip trajectories in `*_vision_final.json` — do they
   contain signal that separates slap / pop / fingerstyle? Then choose an approach
   (rule-based vs. temporal classifier).
3. Cleanup pass: promote `separator.py`, regenerate `requirements.txt`, split `legacy/` into
   dead code vs. preserved candidates.
4. Left-hand fret detection (consumes the trained `left_hand` class).
5. Audio ↔ vision temporal fusion, then tablature rendering.
6. Optional: 5+ string support, `.mscz` export, ground-truth-based quantitative evaluation.

---

## Background

Grew out of **Tablature_CV**, a capstone project transcribing bass and guitar video to
MusicXML (84.94 % audio-vision matching, later 87.6 %). Those numbers are **not** carried
over as baselines — `Tab_CV_Bass` is a full rewrite with a different class set, a different
core problem, and a narrower instrument scope. What was carried over is the reasoning: the
string-detection failure data is precisely what made the current geometric approach the
obvious choice.

Key references are kept in `data/reference/` (HTDemucs, MT3, beat tracking, bass/guitar
string detection, multimodal guitar dataset, ConvGRU action recognition).