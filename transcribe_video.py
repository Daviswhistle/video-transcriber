#!/usr/bin/env python3
"""
Local video/audio transcription for long files.

Default path:
  MP4 -> faster-whisper (direct decode via PyAV) -> TXT/SRT/Markdown/JSON

Optional speaker diarization:
  --diarize -> temporary 16 kHz mono WAV via ffmpeg -> pyannote community-1
              -> speaker labels reconciled with Whisper timestamps

Designed for Windows / WSL / Linux and long recordings.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-community-1"


@dataclass
class SegmentRow:
    start: float
    end: float
    text: str
    speaker: Optional[str] = None


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def format_clock(seconds: float, *, srt: bool = False) -> str:
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    if srt:
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def clean_text(text: str) -> str:
    # Conservative cleanup only: normalize whitespace without rewriting content.
    return re.sub(r"\s+", " ", text).strip()


def detect_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_compute_type(device: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "float16" if device == "cuda" else "int8"


def transcribe_faster_whisper(args: argparse.Namespace) -> tuple[list[SegmentRow], dict[str, Any]]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit(
            "faster-whisper가 설치되어 있지 않습니다.\n"
            "  pip install -r requirements.txt"
        ) from exc

    device = detect_device(args.device)
    compute_type = resolve_compute_type(device, args.compute_type)

    print(f"[1/3] 모델 로드: {args.model} / device={device} / compute={compute_type}")
    try:
        model = WhisperModel(args.model, device=device, compute_type=compute_type)
    except Exception as exc:
        if args.device == "auto" and device == "cuda":
            eprint(f"CUDA 모델 로드 실패: {exc}")
            eprint("CPU int8로 자동 재시도합니다.")
            device = "cpu"
            compute_type = "int8"
            model = WhisperModel(args.model, device=device, compute_type=compute_type)
        else:
            raise

    transcribe_kwargs: dict[str, Any] = {
        "language": None if args.language == "auto" else args.language,
        "beam_size": args.beam_size,
        "vad_filter": not args.no_vad,
        "word_timestamps": args.word_timestamps,
        "condition_on_previous_text": args.condition_on_previous_text,
    }

    if not args.no_vad:
        transcribe_kwargs["vad_parameters"] = {
            "min_silence_duration_ms": args.vad_min_silence_ms
        }
    if args.initial_prompt:
        transcribe_kwargs["initial_prompt"] = args.initial_prompt
    if args.hotwords:
        transcribe_kwargs["hotwords"] = args.hotwords

    print(f"[2/3] 전사 시작: {args.input}")
    segments_gen, info = model.transcribe(str(args.input), **transcribe_kwargs)

    duration = float(getattr(info, "duration", 0.0) or 0.0)
    rows: list[SegmentRow] = []
    last_bucket = -1

    for seg in segments_gen:
        row = SegmentRow(
            start=float(seg.start),
            end=float(seg.end),
            text=clean_text(seg.text),
        )
        if row.text:
            rows.append(row)

        if duration > 0:
            pct = min(100.0, 100.0 * float(seg.end) / duration)
            bucket = int(pct // 5)
            if bucket > last_bucket:
                last_bucket = bucket
                print(f"  진행 {pct:5.1f}%  ({format_clock(seg.end)} / {format_clock(duration)})")
        elif len(rows) % 100 == 0:
            print(f"  {len(rows)}개 구간 처리")

    metadata = {
        "engine": "faster-whisper",
        "model": args.model,
        "device": device,
        "compute_type": compute_type,
        "language": getattr(info, "language", args.language),
        "language_probability": getattr(info, "language_probability", None),
        "duration_seconds": duration or (rows[-1].end if rows else 0.0),
        "vad": not args.no_vad,
    }

    del model
    gc.collect()
    return rows, metadata


def require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit(
            "--diarize에는 ffmpeg가 필요합니다.\n"
            "Windows: winget install Gyan.FFmpeg\n"
            "WSL/Ubuntu: sudo apt update && sudo apt install -y ffmpeg"
        )
    return ffmpeg


def extract_temp_wav(input_path: Path, wav_path: Path) -> None:
    ffmpeg = require_ffmpeg()
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True)


def diarize_pyannote(
    args: argparse.Namespace, rows: list[SegmentRow], transcription_device: str
) -> dict[str, Any]:
    try:
        import torch
        from pyannote.audio import Pipeline
        from pyannote.audio.pipelines.utils.hook import ProgressHook
    except ImportError as exc:
        raise SystemExit(
            "화자 분리 의존성이 설치되어 있지 않습니다.\n"
            "  pip install -r requirements-diarize.txt"
        ) from exc

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit(
            "--diarize에는 Hugging Face 토큰이 필요합니다.\n"
            "환경변수 HF_TOKEN을 설정하거나 --hf-token으로 넘겨주세요.\n"
            "또한 pyannote/speaker-diarization-community-1 사용 조건에 동의해야 합니다."
        )

    # Keep the official, fixed checkpoint ID rather than accepting arbitrary model IDs.
    model_id = PYANNOTE_MODEL_ID

    print("[3/3] 화자 분리 준비: 16 kHz mono WAV 임시 추출")
    with tempfile.TemporaryDirectory(prefix="video_transcriber_") as tmpdir:
        wav_path = Path(tmpdir) / "audio_16k_mono.wav"
        extract_temp_wav(args.input, wav_path)

        print(f"      pyannote 모델 로드: {model_id}")
        pipeline = Pipeline.from_pretrained(model_id, token=hf_token)

        diar_device = "cuda" if transcription_device == "cuda" and torch.cuda.is_available() else "cpu"
        pipeline.to(torch.device(diar_device))

        diar_kwargs: dict[str, int] = {}
        if args.min_speakers is not None:
            diar_kwargs["min_speakers"] = args.min_speakers
        if args.max_speakers is not None:
            diar_kwargs["max_speakers"] = args.max_speakers

        print(f"      화자 분리 실행: device={diar_device}")
        with ProgressHook() as hook:
            output = pipeline(str(wav_path), hook=hook, **diar_kwargs)

    # community-1 exposes exclusive diarization specifically for reconciliation
    # with transcription timestamps. Fall back to normal diarization if needed.
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = output.speaker_diarization

    speaker_turns: list[tuple[float, float, str]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        speaker_turns.append((float(turn.start), float(turn.end), str(speaker)))

    assign_speakers_by_overlap(rows, speaker_turns)

    return {
        "diarization": True,
        "diarization_engine": "pyannote.audio",
        "diarization_model": model_id,
        "diarization_device": diar_device,
        "speaker_count": len({s for _, _, s in speaker_turns}),
        "speaker_turns": [
            {"start": a, "end": b, "speaker": s} for a, b, s in speaker_turns
        ],
    }


def assign_speakers_by_overlap(
    rows: list[SegmentRow], speaker_turns: list[tuple[float, float, str]]
) -> None:
    if not rows or not speaker_turns:
        return

    # Sweep through turns. Long recordings remain efficient because we only inspect
    # diarization turns that can overlap each ASR segment.
    j = 0
    n = len(speaker_turns)

    for row in rows:
        while j < n and speaker_turns[j][1] <= row.start:
            j += 1

        best_speaker: Optional[str] = None
        best_overlap = 0.0
        k = max(0, j - 1)

        while k < n and speaker_turns[k][0] < row.end:
            t0, t1, speaker = speaker_turns[k]
            overlap = max(0.0, min(row.end, t1) - max(row.start, t0))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
            k += 1

        if best_speaker is None:
            # Conservative fallback for boundary gaps: nearest diarization midpoint.
            midpoint = (row.start + row.end) / 2.0
            candidates = []
            for idx in (max(0, j - 1), min(n - 1, j)):
                t0, t1, speaker = speaker_turns[idx]
                distance = abs(midpoint - ((t0 + t1) / 2.0))
                candidates.append((distance, speaker))
            if candidates and min(candidates)[0] <= 2.0:
                best_speaker = min(candidates)[1]

        row.speaker = best_speaker or "SPEAKER_UNKNOWN"


def write_txt(path: Path, rows: Iterable[SegmentRow]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            speaker = f" {r.speaker}" if r.speaker else ""
            f.write(f"[{format_clock(r.start)} - {format_clock(r.end)}]{speaker} {r.text}\n")


def write_srt(path: Path, rows: Iterable[SegmentRow]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            text = f"{r.speaker}: {r.text}" if r.speaker else r.text
            f.write(f"{i}\n")
            f.write(f"{format_clock(r.start, srt=True)} --> {format_clock(r.end, srt=True)}\n")
            f.write(text + "\n\n")


def paragraphize(rows: list[SegmentRow], gap_seconds: float) -> list[list[SegmentRow]]:
    if not rows:
        return []

    groups: list[list[SegmentRow]] = [[rows[0]]]
    for row in rows[1:]:
        prev = groups[-1][-1]
        speaker_changed = bool(row.speaker and prev.speaker and row.speaker != prev.speaker)
        gap = row.start - prev.end
        if speaker_changed or gap >= gap_seconds:
            groups.append([row])
        else:
            groups[-1].append(row)
    return groups


def write_markdown(path: Path, rows: list[SegmentRow], metadata: dict[str, Any], gap_seconds: float) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# 녹취록 — {metadata.get('source_name', '')}\n\n")
        f.write("> 이 문서는 음성인식 결과의 공백만 정규화하고 문단을 묶은 것입니다. ")
        f.write("내용을 요약·의역하거나 임의로 교정하지 않았습니다.\n\n")

        for group in paragraphize(rows, gap_seconds):
            start = group[0].start
            speaker = group[0].speaker
            heading = f"**[{format_clock(start)}]"
            if speaker:
                heading += f" {speaker}"
            heading += "**"
            text = " ".join(r.text for r in group)
            f.write(heading + "\n\n")
            f.write(text + "\n\n")


def write_json(path: Path, rows: list[SegmentRow], metadata: dict[str, Any]) -> None:
    payload = {
        "metadata": metadata,
        "segments": [asdict(r) for r in rows],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def output_dir_for(args: argparse.Namespace) -> Path:
    if args.output_dir:
        out = Path(args.output_dir).expanduser().resolve()
    else:
        out = args.input.parent / f"{args.input.stem}_transcript"
    out.mkdir(parents=True, exist_ok=True)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="긴 MP4/오디오 파일을 로컬에서 Whisper로 녹취합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="입력 MP4/MKV/MOV/MP3/WAV 등")
    parser.add_argument("--output-dir", help="결과 폴더")
    parser.add_argument("--model", default="large-v3", help="Whisper 모델")
    parser.add_argument("--language", default="ko", help="ko, en, ja 또는 auto")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--compute-type",
        default="auto",
        help="auto, float16, int8_float16, int8 등",
    )
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--no-vad", action="store_true", help="Silero VAD 비활성화")
    parser.add_argument("--vad-min-silence-ms", type=int, default=500)
    parser.add_argument("--word-timestamps", action="store_true")
    parser.add_argument(
        "--condition-on-previous-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="이전 구간 텍스트를 다음 구간 조건으로 사용",
    )
    parser.add_argument(
        "--initial-prompt",
        help="고유명사/전문용어 힌트. 예: 'Novo Nordisk, semaglutide, CagriSema'",
    )
    parser.add_argument("--hotwords", help="인식 우선 힌트 단어/구문")
    parser.add_argument(
        "--paragraph-gap",
        type=float,
        default=2.0,
        help="읽기용 Markdown에서 새 문단으로 나눌 무음 간격(초)",
    )

    # Optional diarization
    parser.add_argument("--diarize", action="store_true", help="pyannote로 화자 분리")
    parser.add_argument("--hf-token", help="Hugging Face token. 가능하면 HF_TOKEN 환경변수 권장")
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="동일 입력/옵션의 .raw.json이 있으면 전사를 재사용",
    )

    args = parser.parse_args()
    args.input = args.input.expanduser().resolve()
    if not args.input.exists() or not args.input.is_file():
        parser.error(f"입력 파일을 찾을 수 없습니다: {args.input}")
    if args.min_speakers and args.max_speakers and args.min_speakers > args.max_speakers:
        parser.error("--min-speakers는 --max-speakers보다 클 수 없습니다.")
    return args


def output_paths(out_dir: Path, stem: str, suffix: str = "") -> dict[str, Path]:
    return {
        "txt": out_dir / f"{stem}{suffix}.txt",
        "srt": out_dir / f"{stem}{suffix}.srt",
        "md": out_dir / f"{stem}{suffix}.md",
        "json": out_dir / f"{stem}{suffix}.json",
    }


def save_outputs(
    paths: dict[str, Path],
    rows: list[SegmentRow],
    metadata: dict[str, Any],
    paragraph_gap: float,
) -> None:
    write_txt(paths["txt"], rows)
    write_srt(paths["srt"], rows)
    write_markdown(paths["md"], rows, metadata, paragraph_gap)
    write_json(paths["json"], rows, metadata)


def transcription_signature(args: argparse.Namespace) -> dict[str, Any]:
    """Options that materially affect the Whisper transcript."""
    return {
        "model": args.model,
        "language": args.language,
        "device": args.device,
        "compute_type": args.compute_type,
        "beam_size": args.beam_size,
        "vad": not args.no_vad,
        "vad_min_silence_ms": args.vad_min_silence_ms,
        "word_timestamps": args.word_timestamps,
        "condition_on_previous_text": args.condition_on_previous_text,
        "initial_prompt": args.initial_prompt,
        "hotwords": args.hotwords,
    }


def source_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "source": str(path),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def load_raw_checkpoint(
    path: Path, args: argparse.Namespace
) -> Optional[tuple[list[SegmentRow], dict[str, Any]]]:
    if not path.exists() or not args.resume:
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = dict(payload["metadata"])
        segments = payload["segments"]
    except Exception as exc:
        eprint(f"기존 체크포인트를 읽지 못해 새로 전사합니다: {exc}")
        return None

    fingerprint = source_fingerprint(args.input)
    for key, value in fingerprint.items():
        if metadata.get(key) != value:
            return None
    if metadata.get("transcription_signature") != transcription_signature(args):
        return None

    try:
        rows = [
            SegmentRow(
                start=float(item["start"]),
                end=float(item["end"]),
                text=str(item["text"]),
                speaker=item.get("speaker"),
            )
            for item in segments
        ]
    except Exception as exc:
        eprint(f"기존 체크포인트 형식이 올바르지 않아 새로 전사합니다: {exc}")
        return None

    print(f"[1/3] 기존 전사 체크포인트 재사용: {path}")
    print("[2/3] Whisper 전사 건너뜀")
    return rows, metadata


def preflight_diarization(args: argparse.Namespace) -> None:
    """Fail before a long transcription when diarization prerequisites are missing.

    In particular, verify the gated Hugging Face model can actually be downloaded
    with the supplied token. This prevents wasting hours on ASR before discovering
    that Community-1 access was never accepted.
    """
    require_ffmpeg()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise SystemExit(
            "--diarize에는 Hugging Face 토큰이 필요합니다.\n"
            ".env 또는 환경변수 HF_TOKEN을 설정하거나 --hf-token으로 넘겨주세요.\n"
            f"먼저 https://huggingface.co/{PYANNOTE_MODEL_ID} 에서 사용자 조건에 동의하고 "
            "접근 권한을 받은 뒤 Read 토큰을 만드세요."
        )

    try:
        import torch  # noqa: F401
        import pyannote.audio  # noqa: F401
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "화자 분리 의존성이 설치되어 있지 않습니다.\n"
            "  pip install -r requirements-diarize.txt"
        ) from exc

    print("[0/3] 화자 분리 사전 점검: FFmpeg / Hugging Face 접근권한 확인")
    try:
        # config.yaml is tiny but gated, so this validates both the token and
        # acceptance of the model's user conditions before expensive ASR starts.
        hf_hub_download(
            repo_id=PYANNOTE_MODEL_ID,
            filename="config.yaml",
            token=hf_token,
        )
    except Exception as exc:
        raise SystemExit(
            "Hugging Face의 pyannote Community-1 모델에 접근할 수 없습니다.\n\n"
            "화자 분리 전에 아래 순서를 먼저 완료해야 합니다.\n"
            f"  1. https://huggingface.co/{PYANNOTE_MODEL_ID} 접속\n"
            "  2. 사용자 조건에 동의하여 gated model 접근 승인\n"
            "  3. https://huggingface.co/settings/tokens 에서 Read 토큰 생성\n"
            "  4. .env에 HF_TOKEN=hf_... 저장\n\n"
            f"원래 오류: {type(exc).__name__}: {exc}"
        ) from exc


def main() -> int:
    args = parse_args()
    out_dir = output_dir_for(args)
    stem = args.input.stem

    # IMPORTANT: validate cheap prerequisites before spending time on ASR.
    if args.diarize:
        preflight_diarization(args)

    raw_paths = output_paths(out_dir, stem, ".raw")
    checkpoint = load_raw_checkpoint(raw_paths["json"], args) if args.diarize else None

    if checkpoint is not None:
        rows, metadata = checkpoint
    else:
        rows, metadata = transcribe_faster_whisper(args)
        metadata.update(source_fingerprint(args.input))
        metadata["source_name"] = args.input.name
        metadata["transcription_signature"] = transcription_signature(args)

    # Persist the ASR result immediately. If diarization later fails (model access,
    # OOM, network, etc.), the expensive transcription is still available.
    if args.diarize:
        raw_metadata = dict(metadata)
        raw_metadata["diarization"] = False
        raw_metadata["checkpoint"] = "pre-diarization"
        if checkpoint is None:
            save_outputs(raw_paths, rows, raw_metadata, args.paragraph_gap)
            print("      전사 체크포인트 저장:")
            print(f"        JSON: {raw_paths['json']}")

        try:
            diar_meta = diarize_pyannote(args, rows, str(metadata.get("device", "cpu")))
            metadata.update(diar_meta)
        except BaseException:
            eprint("\n화자 분리에 실패했지만 전사 결과는 보존했습니다:")
            eprint(f"  TXT : {raw_paths['txt']}")
            eprint(f"  SRT : {raw_paths['srt']}")
            eprint(f"  MD  : {raw_paths['md']}")
            eprint(f"  JSON: {raw_paths['json']}")
            raise
    else:
        metadata["diarization"] = False
        print("[3/3] 화자 분리 건너뜀")

    paths = output_paths(out_dir, stem)
    save_outputs(paths, rows, metadata, args.paragraph_gap)

    print("\n완료")
    print(f"  TXT : {paths['txt']}")
    print(f"  SRT : {paths['srt']}")
    print(f"  MD  : {paths['md']}")
    print(f"  JSON: {paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
