import json
from pathlib import Path
from music21 import stream, note, tempo, meter, instrument

def load_notes(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bpm = data["bpm"]
    notes = data["notes"]

    # onset_beat_q 기준으로 정렬
    notes_sorted = sorted(notes, key=lambda n: (n.get("onset_beat_q", 0.0), n["id"]))
    return bpm, notes_sorted


def build_score_from_notes(bpm: int, notes):
    """
    JSON의 notes 리스트를 music21 Score 객체로 변환
    - onset_beat_q, duration_beats_q 기준으로 리듬 구성
    """
    sc = stream.Score()
    part = stream.Part()
    part.id = "BassPart"

    # 악기 설정 (MuseScore에서 악기로 인식)
    part.insert(0, instrument.ElectricBass())

    # 박자표, 템포
    ts = meter.TimeSignature("4/4")
    part.insert(0, ts)

    mm = tempo.MetronomeMark(number=bpm)  # quarter = bpm
    part.insert(0, mm)

    prev_end_beat = 0.0

    for n in notes:
        onset = float(n.get("onset_beat_q", n["onset_beat"]))
        dur = float(n.get("duration_beats_q", n["duration_beats"]))
        midi_num = float(n["midi"])

        # midi가 소수라서 반올림
        midi_int = int(round(midi_num))

        # 앞에 빈 공간 있으면 rest 추가
        gap = onset - prev_end_beat
        if gap > 0.01:  # 너무 미세한 건 무시
            r = note.Rest()
            r.quarterLength = gap
            part.append(r)

        # 실제 음표
        nt = note.Note(midi_int)
        nt.quarterLength = dur
        part.append(nt)

        prev_end_beat = onset + dur

    sc.append(part)
    return sc


def main():
    import argparse

    parser = argparse.ArgumentParser(description="audio_notes.json → MIDI/MusicXML 변환")
    parser.add_argument(
        "--input-json",
        required=True,
        help="audio_pipeline에서 생성한 JSON 경로",
    )
    parser.add_argument(
        "--out-midi",
        default="output.mid",
        help="저장할 MIDI 파일 경로 (기본: output.mid)",
    )
    parser.add_argument(
        "--out-mxml",
        default="output.musicxml",
        help="저장할 MusicXML 파일 경로 (기본: output.musicxml)",
    )
    args = parser.parse_args()

    json_path = Path(args.input_json)
    if not json_path.exists():
        raise FileNotFoundError(json_path)

    bpm, notes = load_notes(json_path)
    print(f"[INFO] BPM: {bpm}, 노트 개수: {len(notes)}")

    sc = build_score_from_notes(bpm, notes)

    # MIDI 저장
    midi_path = Path(args.out_midi)
    sc.write("midi", fp=str(midi_path))
    print(f"[INFO] MIDI 저장 완료 → {midi_path}")

    # MusicXML 저장
    mxml_path = Path(args.out_mxml)
    sc.write("musicxml", fp=str(mxml_path))
    print(f"[INFO] MusicXML 저장 완료 → {mxml_path}")


if __name__ == "__main__":
    main()