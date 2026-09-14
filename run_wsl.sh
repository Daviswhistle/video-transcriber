#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "사용법: ./run_wsl.sh /mnt/c/path/video.mp4 [--diarize] [기타 옵션...]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="$1"
shift || true

# Friendly aliases for a common PowerShell-style spelling mistake.
ARGS=()
for arg in "$@"; do
  case "$arg" in
    -Diarize|--Diarize) ARGS+=("--diarize") ;;
    *) ARGS+=("$arg") ;;
  esac
done

# Load local secrets/settings automatically if present.
if [[ -f "$SCRIPT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  python3 -m venv "$SCRIPT_DIR/.venv"
  "$PYTHON" -m pip install -U pip
fi

# Keep the venv synchronized with the repository after updates.
"$PYTHON" -m pip install -q -r "$SCRIPT_DIR/requirements.txt"

FORCE_CPU=0
for ((i=0; i<${#ARGS[@]}; i++)); do
  if [[ "${ARGS[$i]}" == "--device" ]] && (( i + 1 < ${#ARGS[@]} )) && [[ "${ARGS[$((i+1))]}" == "cpu" ]]; then
    FORCE_CPU=1
    break
  fi
done

# CTranslate2/faster-whisper GPU inference needs CUDA 12 cuBLAS + cuDNN 9.
# WSL exposes the Windows NVIDIA driver, but these user-space libraries may still
# be missing. Install them in the venv and prepend their paths automatically.
if [[ "$FORCE_CPU" -eq 0 ]] && command -v nvidia-smi >/dev/null 2>&1; then
  "$PYTHON" -m pip install -q -r "$SCRIPT_DIR/requirements-gpu-wsl.txt"
  CUDA_LIB_DIRS="$("$PYTHON" - <<'PY'
from pathlib import Path
import nvidia.cublas
import nvidia.cudnn
print(f"{Path(next(iter(nvidia.cublas.__path__))) / 'lib'}:{Path(next(iter(nvidia.cudnn.__path__))) / 'lib'}")
PY
)"
  export LD_LIBRARY_PATH="${CUDA_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

DIARIZE=0
for arg in "${ARGS[@]}"; do
  if [[ "$arg" == "--diarize" ]]; then
    DIARIZE=1
    break
  fi
done

if [[ "$DIARIZE" -eq 1 ]]; then
  # Check FFmpeg BEFORE a potentially long transcription.
  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "--diarize에는 ffmpeg가 필요합니다." >&2
    echo "WSL/Ubuntu: sudo apt update && sudo apt install -y ffmpeg" >&2
    exit 2
  fi

  "$PYTHON" -m pip install -r "$SCRIPT_DIR/requirements-diarize.txt"
fi

exec "$PYTHON" "$SCRIPT_DIR/transcribe_video.py" "$INPUT" "${ARGS[@]}"
