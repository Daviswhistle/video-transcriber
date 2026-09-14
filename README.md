# video-transcriber

긴 MP4/오디오 파일을 로컬에서 **Whisper 녹취 + 선택적 화자 분리**까지 처리하는 도구입니다.

- `faster-whisper` `large-v3` 기본 사용
- MP4를 먼저 WAV로 바꾸지 않고 바로 전사
- TXT / SRT / Markdown / JSON 출력
- `pyannote.audio` `speaker-diarization-community-1` 화자 분리 옵션
- 긴 전사 후 화자 분리가 실패해도 `.raw.*` 체크포인트 보존
- 같은 파일/같은 전사 옵션이면 다음 실행에서 `.raw.json`을 자동 재사용
- WSL에서 CUDA 12용 cuBLAS/cuDNN 경로 자동 설정
- `--diarize` 사용 전 FFmpeg/Hugging Face 접근권한을 **전사 시작 전에 검사**

## ⚠️ 화자 분리를 쓸 사람은 이것부터

`--diarize`는 Hugging Face의 gated 모델인 [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)을 사용합니다.

**토큰만 만들면 되는 것이 아닙니다. 먼저 모델 페이지에서 사용자 조건에 동의하여 접근 권한을 받아야 합니다.** 현재 모델 페이지는 파일 접근 전에 연락처 공유 조건 동의를 요구합니다.

순서는 반드시 다음과 같습니다.

1. Hugging Face 계정으로 로그인
2. [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) 페이지 열기
3. 모델의 사용자 조건을 확인하고 **접근 동의/승인** 완료
4. [Hugging Face Access Tokens](https://huggingface.co/settings/tokens)에서 **Read** 토큰 생성
5. 이 레포의 `.env.example`을 `.env`로 복사하고 토큰 입력

```bash
cp .env.example .env
vi .env
```

```dotenv
HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
```

이 도구는 `--diarize` 실행 시 **실제 모델 파일 접근을 사전 검사**합니다. 승인되지 않은 계정/잘못된 토큰이면 긴 Whisper 전사를 시작하지 않고 즉시 종료합니다.

---

## WSL / Ubuntu 권장 설치

### 1. 시스템 패키지

기본 전사에는 별도 FFmpeg가 필수는 아닙니다. 화자 분리를 쓸 경우 설치합니다.

```bash
sudo apt update
sudo apt install -y python3-venv ffmpeg
```

NVIDIA GPU를 쓴다면 WSL에서 먼저 확인합니다.

```bash
nvidia-smi
```

> WSL에서는 Windows의 NVIDIA 드라이버를 사용합니다. 이 레포 때문에 WSL 내부에 별도의 NVIDIA 드라이버를 설치할 필요는 없습니다.

### 2. 실행 권한

```bash
chmod +x run_wsl.sh
```

### 3. 기본 전사

```bash
./run_wsl.sh "/mnt/c/path/to/video.mp4"
```

### 4. 화자 분리 포함

`.env`에 `HF_TOKEN`을 넣고 모델 접근 승인을 받은 뒤:

```bash
./run_wsl.sh "/mnt/c/path/to/video.mp4" --diarize
```

화자가 두 명임을 알고 있다면:

```bash
./run_wsl.sh "/mnt/c/path/to/video.mp4" \
  --diarize --min-speakers 2 --max-speakers 2
```

실수 방지를 위해 WSL 래퍼는 `-Diarize`와 `--Diarize`도 `--diarize`로 교정합니다.

---

## Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

기본 전사:

```powershell
.\run_windows.ps1 "D:\video\meeting.mp4"
```

화자 분리:

```powershell
winget install Gyan.FFmpeg
Copy-Item .env.example .env
# .env의 HF_TOKEN 수정 후
.\run_windows.ps1 "D:\video\meeting.mp4" -Diarize
```

> 네이티브 Windows에서 NVIDIA GPU를 쓸 경우 CTranslate2가 요구하는 CUDA/cuDNN 런타임이 시스템에 별도로 준비되어 있어야 합니다. WSL 경로가 설정이 더 단순합니다.

---

## 결과물

입력 파일이 `meeting.mp4`라면 기본 출력 폴더는 `meeting_transcript/`입니다.

```text
meeting_transcript/
├── meeting.txt
├── meeting.srt
├── meeting.md
├── meeting.json
├── meeting.raw.txt      # --diarize 사용 시 전사 직후 체크포인트
├── meeting.raw.srt
├── meeting.raw.md
└── meeting.raw.json
```

- `*.txt`: 구간별 타임스탬프
- `*.srt`: 자막
- `*.md`: 사람이 읽기 편한 문단형 녹취
- `*.json`: 메타데이터 + 구조화 구간
- `*.raw.*`: 화자 분리 전 원본 Whisper 결과

Markdown은 내용을 요약하거나 의역하지 않습니다. 공백 정규화와 문단 묶기만 합니다.

### 체크포인트/재시작

`--diarize`에서는 Whisper 전사가 끝나는 즉시 `.raw.*`를 저장합니다. 이후 pyannote, 네트워크, GPU 등의 문제로 화자 분리가 실패해도 전사 결과는 남습니다.

같은 입력 파일이고 전사 옵션도 같다면 다음 `--diarize` 실행에서 기존 `.raw.json`을 자동으로 읽어 **Whisper 전사를 건너뜁니다.** 강제로 다시 전사하려면:

```bash
./run_wsl.sh input.mp4 --diarize --no-resume
```

---

## 직접 Python으로 실행

```bash
python transcribe_video.py input.mp4
```

영어:

```bash
python transcribe_video.py input.mp4 --language en
```

언어 자동 감지:

```bash
python transcribe_video.py input.mp4 --language auto
```

전문용어 힌트:

```bash
python transcribe_video.py input.mp4 \
  --initial-prompt "Novo Nordisk, Wegovy, semaglutide, CagriSema"
```

CPU 강제:

```bash
python transcribe_video.py input.mp4 --device cpu --compute-type int8
```

GPU 강제:

```bash
python transcribe_video.py input.mp4 --device cuda --compute-type float16
```

---

## WSL CUDA 오류

다음과 같은 오류가 대표적입니다.

```text
RuntimeError: Library libcublas.so.12 is not found or cannot be loaded
```

`run_wsl.sh`는 NVIDIA GPU가 감지되면 `requirements-gpu-wsl.txt`의 CUDA 12 cuBLAS/cuDNN 패키지를 설치하고 해당 라이브러리 경로를 `LD_LIBRARY_PATH`에 자동 추가합니다.

수동 확인:

```bash
source .venv/bin/activate
python - <<'PY'
import ctypes
for lib in ["libcublas.so.12", "libcudnn.so.9"]:
    ctypes.CDLL(lib)
    print("OK:", lib)
PY
```

---

## 기본값

- Whisper 모델: `large-v3`
- 언어: `ko`
- 장치: `auto`
- NVIDIA GPU: `float16`
- CPU: `int8`
- VAD: 활성화
- beam size: `5`
- 화자 분리: 기본 비활성화

10GB라는 **파일 크기 자체보다 영상 재생 시간과 GPU 성능이 더 큰 병목**입니다. 기본 전사는 PyAV를 통해 MP4를 직접 읽기 때문에 거대한 중간 WAV를 만들지 않습니다. 화자 분리 단계에서만 임시 16 kHz mono WAV를 생성하고 완료 후 삭제합니다.

## 참고

- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- pyannote.audio: https://github.com/pyannote/pyannote-audio
- pyannote Community-1: https://huggingface.co/pyannote/speaker-diarization-community-1
