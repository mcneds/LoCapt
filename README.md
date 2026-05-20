# LoCapt

LoCapt is a local live captioning and translation tool for Windows. It can listen to your microphone, system audio, or both, then transcribe and optionally translate speech using local Faster-Whisper models.

The main goal is to make meetings, videos, calls, and language learning easier without needing a cloud transcription service.

---

## Features

* Local speech-to-text using Faster-Whisper
* Optional translation output
* Microphone capture
* System audio capture
* Separate mic and system caption panes when both sources are enabled
* Optional source-language text beside translated text
* Collapsible settings panel
* Cached device settings between sessions
* Configurable model size, compute type, and worker/thread settings
* Simple Windows Python setup

---

## Requirements

* Windows 10 or Windows 11
* Python 3.10, 3.11, or 3.12 recommended
* A working microphone, system audio device, or both
* Enough disk space for Whisper models

Recommended hardware:

* CPU-only: small or medium model
* NVIDIA GPU: medium, large-v2, or large-v3 depending on VRAM
* 8 GB RAM minimum
* 16 GB+ RAM recommended for larger models

---

## Project Files

Typical project layout:

```text
LoCapt/
├─ LoCapt.py
├─ requirements.txt
├─ README.md
├─ .venv/
└─ locapt_device_cache.json
```

`locapt_device_cache.json` is created automatically. It stores the last selected microphone/system devices and related settings so you do not need to re-select them every time.

---

## Setup

Open PowerShell in the project folder.

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation scripts, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the project does not have a `requirements.txt` yet, install the common dependencies manually:

```powershell
pip install faster-whisper sounddevice numpy scipy
```

Depending on the current UI version, you may also need:

```powershell
pip install PyQt6
```

---

## Faster-Whisper Model Downloads

LoCapt uses Faster-Whisper models. Models are downloaded automatically the first time you run a model name, as long as your internet connection allows Hugging Face downloads.

Common model choices:

```text
tiny
base
small
medium
large-v2
large-v3
```

Recommended starting points:

| Hardware           |                   Recommended Model | Notes                        |
| ------------------ | ----------------------------------: | ---------------------------- |
| Slow CPU / low RAM |                    `tiny` or `base` | Fastest, least accurate      |
| Decent CPU         |                             `small` | Good balance                 |
| Strong CPU         |                            `medium` | Better accuracy, slower      |
| NVIDIA GPU         | `medium`, `large-v2`, or `large-v3` | Best accuracy if VRAM allows |

The first run may take a while because the model has to download. After that, it loads from the local cache.

---

## Manual Model Pre-download

You can force a model to download before running LoCapt.

Activate your virtual environment, then run:

```powershell
python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"
```

Replace `small` with the model you want:

```powershell
python -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cpu', compute_type='int8')"
```

For GPU use, try:

```powershell
python -c "from faster_whisper import WhisperModel; WhisperModel('medium', device='cuda', compute_type='float16')"
```

---

## Model Storage Location

Faster-Whisper downloads models through Hugging Face and stores them in the local Hugging Face cache.

Usually this is somewhere under:

```text
C:\Users\YOUR_USERNAME\.cache\huggingface\hub
```

You can change the cache location by setting `HF_HOME` before running the app:

```powershell
$env:HF_HOME="C:\AIModels\huggingface"
python LoCapt.py
```

To make that permanent for your Windows user:

```powershell
setx HF_HOME "C:\AIModels\huggingface"
```

Restart PowerShell after using `setx`.

---

## Running LoCapt

From the project folder:

```powershell
.\.venv\Scripts\Activate.ps1
python LoCapt.py
```

Or run directly with the venv Python:

```powershell
.\.venv\Scripts\python.exe LoCapt.py
```

---

## Audio Sources

LoCapt can listen to:

* Microphone audio
* System audio
* Both microphone and system audio at the same time

When both are enabled, LoCapt shows them separately so mic speech and system audio do not get mixed into one caption stream.

### Microphone Mode

Use this for:

* Your own speech
* Language practice
* Calls where you only need your side transcribed

### System Mode

Use this for:

* YouTube
* Teams
* Zoom
* Browser audio
* Any audio playing through Windows

### Both Mode

Use this for:

* Meetings
* Calls
* Language learning
* Comparing your spoken response against the system audio

---

## Device Sweep / Device Selection

If LoCapt does not hear audio, use the sweep/test feature to detect working devices.

Recommended process:

1. Select microphone mode and sweep/test mic devices.
2. Select system mode and sweep/test system devices.
3. Confirm the correct input devices are selected.
4. Start listening.

Once working devices are selected, LoCapt saves them to the local cache file so you should not need to sweep every time.

---

## Device Cache

LoCapt stores the last used audio devices in a small local cache file.

Example:

```text
locapt_device_cache.json
```

This file may save:

* Last microphone device
* Last system audio device
* Last source mode
* Last language/translation options
* Last display preferences

If your audio devices change or LoCapt opens the wrong device, delete the cache file and restart the app.

```powershell
Remove-Item .\locapt_device_cache.json
```

Then run LoCapt again and re-select/sweep devices.

---

## Translation Display Options

LoCapt can show translated text by itself, or show the source-language text beside the translation.

This can be useful for learning because you can compare what was said with the translated version.

There are separate settings for:

* Mic source text beside translation
* System source text beside translation

That means you can enable it for one source and disable it for the other.

---

## Settings Panel

The settings panel can be collapsed once your devices and model options are configured.

The app does not auto-collapse settings when listening starts. Use the small collapse button manually when you want more caption space.

---

## Performance Settings

### Model size

Larger models are more accurate but slower.

Suggested order for testing:

```text
base → small → medium → large-v3
```

If captions lag badly, lower the model size.

### Compute type

Common CPU option:

```text
int8
```

Common GPU option:

```text
float16
```

If GPU mode fails, use CPU mode first to confirm the app works.

### Threads / workers

Increasing thread count does not always improve performance. More threads can help until the CPU is saturated, but too many threads can make performance worse because of overhead and contention.

Good starting points:

| CPU Type     | Suggested Threads |
| ------------ | ----------------: |
| 4-core CPU   |                 4 |
| 6-core CPU   |            4 to 6 |
| 8-core CPU   |            6 to 8 |
| 12-core+ CPU |           8 to 12 |

If changing the thread value appears to do nothing, the current pipeline may be bottlenecked by audio chunking, model inference, queue handling, or the selected backend rather than raw CPU thread count.

---

## Troubleshooting

### The model download is slow

The first run downloads the selected model. Use a smaller model like `base` or `small` to test the app first.

### The model download fails

Try:

```powershell
pip install --upgrade huggingface-hub faster-whisper
```

Then manually pre-download:

```powershell
python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"
```

If you are on a corporate network, the download may be blocked by proxy or SSL rules. Download the model on another network or configure the required proxy/certificate settings.

### No microphone audio is detected

Check:

* Windows microphone permissions
* Correct input device selected
* Mic is not muted
* Another app is not taking exclusive control
* Run the mic sweep/test again

### No system audio is detected

Check:

* Audio is playing
* Correct output device is selected in Windows
* System capture device was swept/tested
* Bluetooth headphones are not switching modes
* Try another output device

### Stop button crashes the app

This usually means one of the audio threads, queues, or device streams is still being accessed while shutdown is happening.

Try:

1. Start the app again.
2. Select only mic or only system mode.
3. Test stop.
4. Then test both mode.

If it only crashes in both mode, the shutdown issue is likely in the dual-stream cleanup logic.

### Captions are delayed

Try:

* Smaller model
* Shorter audio chunks
* CPU `int8`
* GPU `float16`, if available
* Fewer simultaneous sources
* Lower thread count if CPU usage is maxed out

### Translation quality is poor

Try:

* Larger model
* Confirm the source language is correct
* Avoid noisy audio
* Use headphones to prevent mic echo

### Wrong device loads every time

Delete the cache:

```powershell
Remove-Item .\locapt_device_cache.json
```

Then restart LoCapt and select devices again.

---

## Updating Dependencies

Activate the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then update packages:

```powershell
pip install --upgrade faster-whisper sounddevice numpy scipy
```

If using PyQt:

```powershell
pip install --upgrade PyQt6
```

---

## Creating `requirements.txt`

If needed, generate a requirements file from your current environment:

```powershell
pip freeze > requirements.txt
```

A minimal starting `requirements.txt` may look like:

```text
faster-whisper
sounddevice
numpy
scipy
PyQt6
```

Only keep `PyQt6` if the current version of LoCapt uses it.

---

## Notes

* Larger models are not always better for live captions if they cause too much delay.
* Device caching improves convenience, but deleting the cache is the easiest fix after changing headsets, docks, monitors, or audio drivers.
* System audio capture is usually more sensitive to Windows device changes than microphone capture.
* For learning, enabling source text beside translation can make the output more useful than translation alone.

---

## License

Add your chosen license here.

Example:

```text
MIT License
```

---

## TODO

Possible future improvements:

* Installer or one-click launcher
* Automatic model download UI
* Better GPU detection
* Better crash-safe stop/start handling
* Saved profiles for different audio setups
* Export transcript to text file
* Hotkeys for start/stop and collapse settings
