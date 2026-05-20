
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import numpy as np
import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel


TARGET_SAMPLE_RATE = 16000
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
REQUIRED_MODEL_FILES = ("config.json", "model.bin")
SETTINGS_CACHE_FILE = APP_DIR / "locapt_settings_cache.json"


@dataclass
class SourceState:
    name: str
    audio_queue: queue.Queue = field(default_factory=queue.Queue)
    text_queue: queue.Queue = field(default_factory=queue.Queue)
    latest_rms: float = 0.0
    latest_peak: float = 0.0
    level_history: list[float] = field(default_factory=lambda: [0.0] * 160)
    last_chunk_seconds: float = 0.0
    processing_started_at: float = 0.0
    is_processing_chunk: bool = False
    dropped_chunks: int = 0
    text_widget: tk.Text | None = None
    visualizer: tk.Canvas | None = None
    level_var: tk.StringVar | None = None
    delay_var: tk.StringVar | None = None
    delay_bar: ttk.Progressbar | None = None
    frame: ttk.LabelFrame | None = None


def get_local_models() -> list[str]:
    models = []
    if MODELS_DIR.is_dir():
        for folder in sorted(MODELS_DIR.iterdir()):
            if folder.is_dir() and all((folder / name).exists() for name in REQUIRED_MODEL_FILES):
                models.append(str(folder.resolve()))
    return models


def short_name(path: str) -> str:
    return Path(path).name


def safe_float(value: str, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def safe_int(value: str, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def clear_queue(q: queue.Queue):
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass


def load_settings_cache() -> dict:
    try:
        if SETTINGS_CACHE_FILE.exists():
            data = json.loads(SETTINGS_CACHE_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def write_settings_cache(data: dict):
    try:
        SETTINGS_CACHE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def cached_label(cache: dict, cache_key: str, labels: list[str], fallback: str) -> str:
    saved = cache.get(cache_key, {})
    if not isinstance(saved, dict):
        return fallback

    saved_label = saved.get("label")
    if saved_label in labels:
        return saved_label

    saved_name = str(saved.get("name", "")).strip().lower()
    if saved_name:
        for label in labels:
            if saved_name in label.lower():
                return label

    return fallback


def device_name_from_label(label: str) -> str:
    main = label.split("|", 1)[0].strip()
    if ":" in main:
        return main.split(":", 1)[1].strip()
    return main


def resample_audio(audio: np.ndarray, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if source_rate == target_rate or len(audio) == 0:
        return audio.astype(np.float32, copy=False)

    duration = len(audio) / float(source_rate)
    target_len = max(1, int(duration * target_rate))
    old_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    new_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(new_x, old_x, audio).astype(np.float32)


def mono_from_float32_bytes(data: bytes, channels: int) -> np.ndarray:
    audio = np.frombuffer(data, dtype=np.float32)
    if channels > 1 and len(audio) >= channels:
        usable = (len(audio) // channels) * channels
        audio = audio[:usable].reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32, copy=False)


def mono_from_int16_bytes(data: bytes, channels: int) -> np.ndarray:
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1 and len(audio) >= channels:
        usable = (len(audio) // channels) * channels
        audio = audio[:usable].reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32, copy=False)


def device_is_loopback(dev: dict) -> bool:
    if bool(dev.get("isLoopbackDevice", False)):
        return True
    name = str(dev.get("name", "")).lower()
    return "loopback" in name or "wasapi" in name and "loopback" in name


def audio_score(rms: float, peak: float) -> float:
    # RMS catches sustained speech. Peak catches short tap/clap tests.
    return max(float(rms), float(peak) * 0.35)


class LoCaptApp:
    def __init__(self):
        self.stop_event = threading.Event()
        self.scan_stop_event = threading.Event()
        self.model_lock = threading.Lock()

        self.model = None
        self.loaded_model_key = None
        self.system_stream = None

        self.system_state = SourceState("System translate")
        self.mic_state = SourceState("Microphone")
        self.active_states: list[SourceState] = []

        self.running = False

        self.root = tk.Tk()
        self.root.title("LoCapt - Local Live Captions")
        self.root.geometry("1240x780")
        self.root.minsize(980, 650)

        self.models = get_local_models() or ["tiny"]
        self.mic_labels, self.mic_map = self.get_mic_choices()
        self.loopback_labels, self.loopback_map = self.get_loopback_choices()
        self.settings_cache = load_settings_cache()

        default_mic = self.mic_labels[0] if self.mic_labels else "Default microphone"
        default_loopback = self.loopback_labels[0] if self.loopback_labels else "No loopback device found"
        cached_mic = cached_label(self.settings_cache, "last_mic", self.mic_labels, default_mic)
        cached_loopback = cached_label(self.settings_cache, "last_loopback", self.loopback_labels, default_loopback)

        self.model_var = tk.StringVar(value=short_name(self.models[0]))
        self.capture_mode_var = tk.StringVar(value="System translate")
        self.mic_var = tk.StringVar(value=cached_mic)
        self.loopback_var = tk.StringVar(value=cached_loopback)
        self.language_var = tk.StringVar(value="de")
        self.task_var = tk.StringVar(value="Translate to English")
        self.chunk_var = tk.StringVar(value="3.0")
        self.overlap_var = tk.StringVar(value="0.25")
        self.threads_var = tk.StringVar(value="4")
        self.silence_var = tk.StringVar(value="0.020")
        self.max_lag_var = tk.StringVar(value="6.0")
        self.scan_seconds_var = tk.StringVar(value="0.8")
        self.mic_scan_seconds_var = tk.StringVar(value="0.8")
        self.auto_catchup_var = tk.BooleanVar(value=True)
        self.topmost_var = tk.BooleanVar(value=True)

        self.system_show_source_var = tk.BooleanVar(
            value=bool(self.settings_cache.get("system_show_source_with_translation", False))
        )
        self.mic_show_source_var = tk.BooleanVar(
            value=bool(self.settings_cache.get("mic_show_source_with_translation", False))
        )

        self.build_ui()
        self.apply_topmost()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self.pump_text)
        self.root.after(100, self.update_level_ui)
        self.root.after(1000, self.watchdog)

    def build_ui(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(expand=True, fill="both")

        self.settings_frame = ttk.LabelFrame(outer, text="Settings", padding=10)
        self.settings_frame.pack(fill="x", pady=(0, 8))
        controls = self.settings_frame

        for col in range(8):
            controls.columnconfigure(col, weight=1)

        ttk.Label(controls, text="Model").grid(row=0, column=0, sticky="w")
        self.model_combo = ttk.Combobox(
            controls,
            textvariable=self.model_var,
            values=[short_name(m) for m in self.models],
            state="readonly",
            width=28,
        )
        self.model_combo.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(0, 8))

        ttk.Label(controls, text="Capture").grid(row=0, column=2, sticky="w")
        self.capture_combo = ttk.Combobox(
            controls,
            textvariable=self.capture_mode_var,
            values=["System translate", "Microphone", "Both: system translate and microphone"],
            state="readonly",
            width=28,
        )
        self.capture_combo.grid(row=1, column=2, columnspan=2, sticky="ew", padx=(0, 8))
        self.capture_combo.bind("<<ComboboxSelected>>", lambda e: self.update_device_visibility())

        ttk.Label(controls, text="System translate source").grid(row=0, column=4, sticky="w")
        self.loopback_combo = ttk.Combobox(
            controls,
            textvariable=self.loopback_var,
            values=self.loopback_labels,
            state="readonly",
            width=52,
        )
        self.loopback_combo.grid(row=1, column=4, columnspan=4, sticky="ew")
        self.loopback_combo.bind("<<ComboboxSelected>>", lambda e: self.save_device_cache())

        ttk.Label(controls, text="Microphone device").grid(row=2, column=4, sticky="w", pady=(8, 0))
        self.mic_combo = ttk.Combobox(
            controls,
            textvariable=self.mic_var,
            values=self.mic_labels,
            state="readonly",
            width=52,
        )
        self.mic_combo.grid(row=3, column=4, columnspan=4, sticky="ew")
        self.mic_combo.bind("<<ComboboxSelected>>", lambda e: self.save_device_cache())

        ttk.Label(controls, text="Language").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.language_var,
            values=["de", "en", "fr", "es", "it", "auto"],
            width=10,
        ).grid(row=3, column=0, sticky="ew", padx=(0, 8))

        ttk.Label(controls, text="Task").grid(row=2, column=1, sticky="w", pady=(8, 0))
        ttk.Combobox(
            controls,
            textvariable=self.task_var,
            values=["Translate to English", "Transcribe same language"],
            state="readonly",
            width=24,
        ).grid(row=3, column=1, sticky="ew", padx=(0, 8))

        ttk.Label(controls, text="Chunk s").grid(row=2, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.chunk_var, width=8).grid(row=3, column=2, sticky="ew", padx=(0, 8))

        ttk.Label(controls, text="Silence RMS").grid(row=2, column=3, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.silence_var, width=10).grid(row=3, column=3, sticky="ew", padx=(0, 8))

        advanced = ttk.Frame(controls)
        advanced.grid(row=4, column=0, columnspan=8, sticky="ew", pady=(8, 0))
        ttk.Label(advanced, text="Overlap s").pack(side="left")
        ttk.Entry(advanced, textvariable=self.overlap_var, width=7).pack(side="left", padx=(4, 12))
        ttk.Label(advanced, text="Threads").pack(side="left")
        ttk.Entry(advanced, textvariable=self.threads_var, width=7).pack(side="left", padx=(4, 12))
        ttk.Label(advanced, text="Max lag s").pack(side="left")
        ttk.Entry(advanced, textvariable=self.max_lag_var, width=7).pack(side="left", padx=(4, 12))
        ttk.Checkbutton(advanced, text="Auto catch up", variable=self.auto_catchup_var).pack(side="left", padx=(0, 12))
        ttk.Checkbutton(advanced, text="Always on top", variable=self.topmost_var, command=self.apply_topmost).pack(side="left", padx=(0, 12))

        source_options = ttk.Frame(controls)
        source_options.grid(row=5, column=0, columnspan=8, sticky="ew", pady=(6, 0))
        ttk.Checkbutton(
            source_options,
            text="System: source + English",
            variable=self.system_show_source_var,
            command=self.save_device_cache,
        ).pack(side="left", padx=(0, 16))
        ttk.Checkbutton(
            source_options,
            text="Mic: source + English",
            variable=self.mic_show_source_var,
            command=self.save_device_cache,
        ).pack(side="left")

        self.buttons_frame = ttk.Frame(outer)
        self.buttons_frame.pack(fill="x", pady=(0, 8))
        buttons = self.buttons_frame

        self.start_button = ttk.Button(buttons, text="Start", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Catch up", width=10, command=self.catch_up_now).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="↻", width=3, command=self.refresh_devices_models).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Clear", width=8, command=self.clear_text).pack(side="right")

        self.scan_frame = ttk.LabelFrame(outer, text="Find active devices", padding=8)
        self.scan_frame.pack(fill="x", pady=(0, 8))
        scan_frame = self.scan_frame
        scan_frame.columnconfigure(3, weight=1)

        ttk.Label(scan_frame, text="Sweep s").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(scan_frame, textvariable=self.scan_seconds_var, width=5).grid(row=0, column=1, sticky="w")
        self.scan_button = ttk.Button(scan_frame, text="Sweep output", width=12, command=self.start_device_sweep)
        self.scan_button.grid(row=0, column=2, sticky="w", padx=(8, 8))
        self.scan_status_var = tk.StringVar(value="Play system audio, then click Sweep output. Loudest loopback device will be selected.")
        ttk.Label(scan_frame, textvariable=self.scan_status_var).grid(row=0, column=3, sticky="ew")

        self.sweep_tree = ttk.Treeview(scan_frame, columns=("score", "device"), show="headings", height=3)
        self.sweep_tree.heading("score", text="RMS / Peak")
        self.sweep_tree.heading("device", text="Loopback device")
        self.sweep_tree.column("score", width=150, stretch=False)
        self.sweep_tree.column("device", width=850, stretch=True)
        self.sweep_tree.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(6, 8))
        self.sweep_tree.bind("<<TreeviewSelect>>", self.on_sweep_select)

        ttk.Label(scan_frame, text="Mic sweep s").grid(row=2, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(scan_frame, textvariable=self.mic_scan_seconds_var, width=5).grid(row=2, column=1, sticky="w")
        self.mic_scan_button = ttk.Button(scan_frame, text="Sweep mic", width=12, command=self.start_mic_sweep)
        self.mic_scan_button.grid(row=2, column=2, sticky="w", padx=(8, 8))
        self.mic_scan_status_var = tk.StringVar(value="Talk, tap the mic, or make sound near the mic, then click Sweep mic.")
        ttk.Label(scan_frame, textvariable=self.mic_scan_status_var).grid(row=2, column=3, sticky="ew")

        self.mic_sweep_tree = ttk.Treeview(scan_frame, columns=("score", "device"), show="headings", height=3)
        self.mic_sweep_tree.heading("score", text="RMS / Peak")
        self.mic_sweep_tree.heading("device", text="Microphone device")
        self.mic_sweep_tree.column("score", width=150, stretch=False)
        self.mic_sweep_tree.column("device", width=850, stretch=True)
        self.mic_sweep_tree.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        self.mic_sweep_tree.bind("<<TreeviewSelect>>", self.on_mic_sweep_select)

        self.status_var = tk.StringVar(value="Ready. Use Both mode to split system audio and microphone into separate caption panes.")
        self.status_label = ttk.Label(outer, textvariable=self.status_var)
        self.status_label.pack(fill="x", pady=(0, 4))

        self.top_controls_visible = True
        self.settings_toggle_row = ttk.Frame(outer)
        self.settings_toggle_row.pack(fill="x", pady=(0, 4))
        self.settings_toggle_button = ttk.Button(
            self.settings_toggle_row,
            text="Hide settings ▲",
            width=16,
            command=self.toggle_top_controls,
        )
        self.settings_toggle_button.pack(anchor="center")

        self.panes = ttk.Panedwindow(outer, orient="horizontal")
        self.panes.pack(expand=True, fill="both")

        self.system_pane = self.create_source_pane(self.system_state)
        self.mic_pane = self.create_source_pane(self.mic_state)
        self.update_device_visibility()

    def toggle_top_controls(self):
        if self.top_controls_visible:
            self.settings_frame.pack_forget()
            self.scan_frame.pack_forget()
            self.settings_toggle_button.configure(text="Show settings ▼")
            self.top_controls_visible = False
        else:
            self.settings_frame.pack(fill="x", pady=(0, 8), before=self.buttons_frame)
            self.scan_frame.pack(fill="x", pady=(0, 8), before=self.status_label)
            self.settings_toggle_button.configure(text="Hide settings ▲")
            self.top_controls_visible = True

    def create_source_pane(self, state: SourceState):
        frame = ttk.LabelFrame(self.panes, text=state.name, padding=8)
        state.frame = frame

        top = ttk.Frame(frame)
        top.pack(fill="x", pady=(0, 6))

        state.level_var = tk.StringVar(value="RMS: 0.0000 | Peak: 0.0000 | Silence cutoff: 0.0200")
        ttk.Label(top, textvariable=state.level_var, width=55).pack(side="left", padx=(0, 8))

        lag_frame = ttk.LabelFrame(top, text="Live delay", padding=4)
        lag_frame.pack(side="right")
        state.delay_var = tk.StringVar(value="~0.0s behind")
        ttk.Label(lag_frame, textvariable=state.delay_var, width=18).pack(anchor="w")
        state.delay_bar = ttk.Progressbar(lag_frame, maximum=15.0, value=0.0, length=160)
        state.delay_bar.pack(anchor="w", pady=(4, 0))

        state.visualizer = tk.Canvas(frame, height=70, bg="#111111", highlightthickness=1, highlightbackground="#777777")
        state.visualizer.pack(fill="x", pady=(0, 8))

        state.text_widget = tk.Text(frame, wrap="word", font=("Segoe UI", 18), undo=False)
        state.text_widget.pack(expand=True, fill="both")
        return frame

    def update_device_visibility(self):
        mode = self.capture_mode_var.get()
        if mode == "Microphone":
            self.mic_combo.configure(state="readonly")
            self.loopback_combo.configure(state="disabled")
            self.show_single_pane(self.mic_pane)
        elif mode == "Both: system translate and microphone":
            self.mic_combo.configure(state="readonly")
            self.loopback_combo.configure(state="readonly")
            self.show_both_panes()
        else:
            self.mic_combo.configure(state="disabled")
            self.loopback_combo.configure(state="readonly")
            self.show_single_pane(self.system_pane)

    def remove_all_panes(self):
        try:
            self.panes.forget(self.system_pane)
        except Exception:
            pass
        try:
            self.panes.forget(self.mic_pane)
        except Exception:
            pass

    def show_single_pane(self, pane):
        self.remove_all_panes()
        self.panes.add(pane, weight=1)

    def show_both_panes(self):
        self.remove_all_panes()
        self.panes.add(self.system_pane, weight=1)
        self.panes.add(self.mic_pane, weight=1)

    def get_mic_choices(self):
        labels = []
        mapping = {}
        try:
            with pyaudio.PyAudio() as p:
                default_input_index = None
                try:
                    default_input_index = int(p.get_default_input_device_info().get("index"))
                except Exception:
                    default_input_index = None

                for i in range(p.get_device_count()):
                    dev = p.get_device_info_by_index(i)
                    channels = int(dev.get("maxInputChannels", 0) or 0)
                    if channels <= 0:
                        continue
                    if device_is_loopback(dev):
                        continue

                    index = int(dev.get("index", i))
                    rate = int(float(dev.get("defaultSampleRate", 48000) or 48000))
                    name = str(dev.get("name", f"Device {index}"))
                    default_mark = " default" if index == default_input_index else ""
                    label = f"{index}: {name} | in:{channels} | {rate} Hz{default_mark}"
                    labels.append(label)
                    mapping[label] = dev
        except Exception:
            pass

        if not labels:
            labels = ["Default microphone"]
            mapping[labels[0]] = None
        return labels, mapping

    def get_loopback_choices(self):
        labels = []
        mapping = {}
        try:
            with pyaudio.PyAudio() as p:
                for dev in p.get_loopback_device_info_generator():
                    index = int(dev["index"])
                    channels = int(dev.get("maxInputChannels", 0))
                    rate = int(float(dev.get("defaultSampleRate", 48000)))
                    name = str(dev.get("name", f"Loopback {index}"))
                    label = f"{index}: {name} | in:{channels} | {rate} Hz"
                    labels.append(label)
                    mapping[label] = dev
        except Exception as e:
            labels = [f"No loopback device found: {e}"]
            mapping[labels[0]] = None
        if not labels:
            labels = ["No loopback device found"]
            mapping[labels[0]] = None
        return labels, mapping

    def refresh_devices_models(self):
        self.models = get_local_models() or ["tiny"]
        self.model_combo.configure(values=[short_name(m) for m in self.models])
        if self.model_var.get() not in [short_name(m) for m in self.models]:
            self.model_var.set(short_name(self.models[0]))

        self.mic_labels, self.mic_map = self.get_mic_choices()
        self.loopback_labels, self.loopback_map = self.get_loopback_choices()
        self.mic_combo.configure(values=self.mic_labels)
        self.loopback_combo.configure(values=self.loopback_labels)

        if self.mic_var.get() not in self.mic_labels:
            self.mic_var.set(cached_label(self.settings_cache, "last_mic", self.mic_labels, self.mic_labels[0]))

        if self.loopback_var.get() not in self.loopback_labels:
            self.loopback_var.set(cached_label(self.settings_cache, "last_loopback", self.loopback_labels, self.loopback_labels[0]))

        self.save_device_cache()
        self.status_var.set("Refreshed devices and models.")

    def selected_model_path(self):
        selected = self.model_var.get()
        for model in self.models:
            if short_name(model) == selected:
                return model
        return selected

    def selected_language(self):
        lang = self.language_var.get().strip().lower()
        return None if lang in ("", "auto", "none") else lang

    def selected_task(self):
        return "translate" if self.task_var.get() == "Translate to English" else "transcribe"

    def apply_topmost(self):
        self.root.attributes("-topmost", bool(self.topmost_var.get()))

    def save_device_cache(self):
        cache = dict(getattr(self, "settings_cache", {}) or {})

        mic_label = self.mic_var.get()
        loopback_label = self.loopback_var.get()

        cache["last_mic"] = {
            "label": mic_label,
            "name": device_name_from_label(mic_label),
        }
        cache["last_loopback"] = {
            "label": loopback_label,
            "name": device_name_from_label(loopback_label),
        }

        cache["system_show_source_with_translation"] = bool(self.system_show_source_var.get())
        cache["mic_show_source_with_translation"] = bool(self.mic_show_source_var.get())

        self.settings_cache = cache
        write_settings_cache(cache)

    def catch_up_now(self):
        for state in self.active_states or [self.system_state, self.mic_state]:
            clear_queue(state.audio_queue)
            state.dropped_chunks += 1
            state.text_queue.put("\n[catch up: skipped old buffered audio]\n")

    def clear_text(self):
        for state in [self.system_state, self.mic_state]:
            if state.text_widget:
                state.text_widget.delete("1.0", "end")

    def current_capture_states(self):
        mode = self.capture_mode_var.get()
        if mode == "Microphone":
            return [self.mic_state]
        if mode == "Both: system translate and microphone":
            return [self.system_state, self.mic_state]
        return [self.system_state]

    def should_show_source_with_translation(self, state: SourceState):
        if self.selected_task() != "translate":
            return False
        if state is self.system_state:
            return bool(self.system_show_source_var.get())
        if state is self.mic_state:
            return bool(self.mic_show_source_var.get())
        return False

    def start(self):
        if self.running:
            return

        self.save_device_cache()

        self.stop_event.clear()
        self.active_states = self.current_capture_states()
        for state in self.active_states:
            clear_queue(state.audio_queue)
            clear_queue(state.text_queue)
            state.latest_rms = 0.0
            state.latest_peak = 0.0
            state.processing_started_at = 0.0
            state.is_processing_chunk = False

        self.running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")

        if self.system_state in self.active_states:
            threading.Thread(target=self.system_audio_thread, daemon=True).start()
        if self.mic_state in self.active_states:
            threading.Thread(target=self.mic_audio_thread, daemon=True).start()

        for state in self.active_states:
            threading.Thread(target=self.transcribe_thread, args=(state,), daemon=True).start()

    def stop(self):
        self.stop_event.set()
        self.running = False

        # Do NOT close self.system_stream here.
        # The system audio thread may currently be inside stream.read().
        # Closing it from the UI thread can crash the process.
        # Let system_audio_thread notice stop_event and close the stream itself.

        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status_var.set("Stopping...")

    def close(self):
        self.save_device_cache()
        self.scan_stop_event.set()
        self.stop()
        self.root.destroy()

    def mic_audio_thread(self):
        stream = None
        try:
            dev = self.mic_map.get(self.mic_var.get())
            if not dev:
                raise RuntimeError("No microphone selected. Click ↻, then Sweep mic while making sound near the mic.")

            index = int(dev["index"])
            channels = max(1, min(2, int(dev.get("maxInputChannels", 1) or 1)))
            source_rate = int(float(dev.get("defaultSampleRate", 48000) or 48000))
            frames = max(256, int(source_rate * 0.25))

            self.status_var.set(f"Opening microphone: {dev.get('name', index)}")

            with pyaudio.PyAudio() as p:
                stream = p.open(
                    format=pyaudio.paInt16,
                    channels=channels,
                    rate=source_rate,
                    frames_per_buffer=frames,
                    input=True,
                    input_device_index=index,
                )

                self.status_var.set(f"Listening to microphone: {dev.get('name', index)} | {source_rate} Hz")

                while not self.stop_event.is_set():
                    try:
                        data = stream.read(frames, exception_on_overflow=False)
                    except Exception:
                        if self.stop_event.is_set():
                            break
                        raise

                    mono = mono_from_int16_bytes(data, channels)
                    self.push_audio(self.mic_state, mono, source_rate)

        except Exception as e:
            if not self.stop_event.is_set():
                self.mic_state.text_queue.put(f"\n[mic audio error] {e}\n")
                self.stop_event.set()

        finally:
            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass

    def system_audio_thread(self):
        stream = None
        try:
            dev = self.loopback_map.get(self.loopback_var.get())
            if not dev:
                raise RuntimeError("No WASAPI loopback device selected. Click ↻, then Sweep while audio is playing.")

            index = int(dev["index"])
            channels = max(1, int(dev.get("maxInputChannels", 2)))
            source_rate = int(float(dev.get("defaultSampleRate", 48000)))
            frames = max(256, int(source_rate * 0.25))

            self.status_var.set(f"Opening WASAPI loopback: {dev.get('name', index)}")

            with pyaudio.PyAudio() as p:
                stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=channels,
                    rate=source_rate,
                    frames_per_buffer=frames,
                    input=True,
                    input_device_index=index,
                )
                self.system_stream = stream

                self.status_var.set(f"Listening to WASAPI loopback: {dev.get('name', index)} | {source_rate} Hz")

                while not self.stop_event.is_set():
                    try:
                        data = stream.read(frames, exception_on_overflow=False)
                    except Exception:
                        if self.stop_event.is_set():
                            break
                        raise

                    mono = mono_from_float32_bytes(data, channels)
                    self.push_audio(self.system_state, mono, source_rate)

        except Exception as e:
            if not self.stop_event.is_set():
                self.system_state.text_queue.put(f"\n[system audio error] {e}\n")
                self.stop_event.set()

        finally:
            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass

            if self.system_stream is stream:
                self.system_stream = None

    def push_audio(self, state: SourceState, mono, source_rate):
        rms = float(np.sqrt(np.mean(mono * mono))) if len(mono) else 0.0
        peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
        state.latest_rms = rms
        state.latest_peak = peak
        state.level_history.append(min(peak, 1.0))
        if len(state.level_history) > 240:
            state.level_history = state.level_history[-240:]
        state.audio_queue.put(resample_audio(mono, source_rate))

    def load_model_if_needed(self):
        model_key = self.selected_model_path()
        threads = safe_int(self.threads_var.get(), 4)
        if self.model is not None and self.loaded_model_key == (model_key, threads):
            return
        self.model = None
        self.loaded_model_key = None

        self.model = WhisperModel(model_key, device="cpu", compute_type="int8", cpu_threads=threads)
        self.loaded_model_key = (model_key, threads)

    def run_whisper(self, audio: np.ndarray, task: str):
        segments, _ = self.model.transcribe(
            audio,
            language=self.selected_language(),
            task=task,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=0.65,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def format_source_and_translation(self, source_text: str, translation_text: str):
        source_text = source_text.strip()
        translation_text = translation_text.strip()

        if source_text and translation_text:
            return f"{source_text}  →  {translation_text}"
        if translation_text:
            return translation_text
        if source_text:
            return source_text
        return ""

    def transcribe_thread(self, state: SourceState):
        try:
            with self.model_lock:
                if self.model is None or self.loaded_model_key != (self.selected_model_path(), safe_int(self.threads_var.get(), 4)):
                    state.text_queue.put(f"Loading model: {self.selected_model_path()}\n")
                    self.load_model_if_needed()
                    state.text_queue.put("Model loaded.\n\n")
        except Exception as e:
            state.text_queue.put(f"\n[model error] {e}\n")
            self.stop_event.set()
            return

        buffer = np.zeros(0, dtype=np.float32)
        while not self.stop_event.is_set():
            chunk_seconds = max(1.0, safe_float(self.chunk_var.get(), 3.0))
            overlap_seconds = max(0.0, safe_float(self.overlap_var.get(), 0.25))
            silence_rms = max(0.0, safe_float(self.silence_var.get(), 0.020))
            max_lag = max(1.0, safe_float(self.max_lag_var.get(), 6.0))
            chunk_samples = int(TARGET_SAMPLE_RATE * chunk_seconds)
            keep_samples = min(int(TARGET_SAMPLE_RATE * overlap_seconds), chunk_samples // 2)

            if self.auto_catchup_var.get() and self.queued_audio_seconds(state) > max_lag:
                old = self.queued_audio_seconds(state)
                clear_queue(state.audio_queue)
                buffer = np.zeros(0, dtype=np.float32)
                state.dropped_chunks += 1
                state.text_queue.put(f"\n[auto catch up: skipped {old:.1f}s of old audio]\n")

            try:
                while len(buffer) < chunk_samples and not self.stop_event.is_set():
                    buffer = np.concatenate([buffer, state.audio_queue.get(timeout=0.25)])
            except queue.Empty:
                continue

            audio = buffer[:chunk_samples]
            buffer = buffer[max(0, chunk_samples - keep_samples):] if keep_samples > 0 else buffer[chunk_samples:]

            rms = float(np.sqrt(np.mean(audio * audio))) if len(audio) else 0.0
            if rms < silence_rms:
                continue

            try:
                start = time.time()
                state.processing_started_at = start
                state.is_processing_chunk = True

                with self.model_lock:
                    if self.should_show_source_with_translation(state):
                        source_text = self.run_whisper(audio, task="transcribe")
                        translation_text = self.run_whisper(audio, task="translate")
                        result = self.format_source_and_translation(source_text, translation_text)
                    else:
                        result = self.run_whisper(audio, task=self.selected_task())

                state.last_chunk_seconds = time.time() - start
                state.is_processing_chunk = False

                if result:
                    state.text_queue.put(result + "\n")
            except Exception as e:
                state.is_processing_chunk = False
                state.text_queue.put(f"\n[transcription error] {e}\n")

    def queued_audio_seconds(self, state: SourceState):
        return state.audio_queue.qsize() * 0.25

    def start_mic_sweep(self):
        if self.mic_scan_button.cget("text") == "Stop":
            self.scan_stop_event.set()
            self.mic_scan_button.configure(text="Sweep mic")
            return
        self.scan_stop_event.clear()
        self.mic_scan_button.configure(text="Stop")
        self.clear_mic_sweep_results()
        threading.Thread(target=self.mic_sweep_thread, daemon=True).start()

    def clear_mic_sweep_results(self):
        for item in self.mic_sweep_tree.get_children():
            self.mic_sweep_tree.delete(item)

    def mic_sweep_thread(self):
        devices = [(label, dev) for label, dev in self.mic_map.items()]
        if not devices:
            self.root.after(0, lambda: self.mic_scan_status_var.set("No microphone devices found."))
            self.root.after(0, lambda: self.mic_scan_button.configure(text="Sweep mic"))
            return

        scan_seconds = max(0.25, safe_float(self.mic_scan_seconds_var.get(), 0.8))
        results = []
        for idx, (label, dev) in enumerate(devices, start=1):
            if self.scan_stop_event.is_set():
                break
            self.root.after(0, lambda idx=idx, total=len(devices), label=label: self.mic_scan_status_var.set(f"Sweeping mic {idx}/{total}: {label}"))
            try:
                rms, peak = self.measure_mic_device(dev, scan_seconds)
                results.append((rms, peak, label))
                self.root.after(0, lambda rms=rms, peak=peak, label=label: self.add_mic_sweep_result(rms, peak, label))
            except Exception as e:
                self.root.after(0, lambda label=label, e=e: self.add_mic_sweep_result(0.0, 0.0, f"{label} [failed: {e}]"))

        if results and not self.scan_stop_event.is_set():
            results.sort(key=lambda x: audio_score(x[0], x[1]), reverse=True)
            rms, peak, label = results[0]
            self.root.after(0, lambda: self.select_mic_sweep_winner(label, rms, peak))
        self.root.after(0, lambda: self.mic_scan_button.configure(text="Sweep mic"))

    def measure_mic_device(self, dev, seconds):
        if not dev:
            return 0.0, 0.0

        index = int(dev["index"])
        channels = max(1, min(2, int(dev.get("maxInputChannels", 1) or 1)))
        source_rate = int(float(dev.get("defaultSampleRate", 48000) or 48000))
        frames = max(256, int(source_rate * 0.10))
        chunks = []

        with pyaudio.PyAudio() as p:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=source_rate,
                frames_per_buffer=frames,
                input=True,
                input_device_index=index,
            )
            try:
                end = time.time() + seconds
                while time.time() < end and not self.scan_stop_event.is_set():
                    data = stream.read(frames, exception_on_overflow=False)
                    chunks.append(mono_from_int16_bytes(data, channels))
            finally:
                stream.stop_stream()
                stream.close()

        if not chunks:
            return 0.0, 0.0
        audio = np.concatenate(chunks)
        return float(np.sqrt(np.mean(audio * audio))), float(np.max(np.abs(audio)))

    def add_mic_sweep_result(self, rms, peak, label):
        self.mic_sweep_tree.insert("", "end", values=(f"{rms:.4f} / {peak:.4f}", label))

    def select_mic_sweep_winner(self, label, rms, peak):
        if label in self.mic_labels:
            self.mic_var.set(label)
            if self.capture_mode_var.get() == "System translate":
                self.capture_mode_var.set("Microphone")
            self.update_device_visibility()
            self.save_device_cache()
        self.mic_scan_status_var.set(f"Selected loudest mic: RMS {rms:.4f}, Peak {peak:.4f}")

    def on_mic_sweep_select(self, event=None):
        selected = self.mic_sweep_tree.selection()
        if not selected:
            return
        values = self.mic_sweep_tree.item(selected[0], "values")
        if len(values) >= 2 and values[1] in self.mic_labels:
            self.mic_var.set(values[1])
            if self.capture_mode_var.get() == "System translate":
                self.capture_mode_var.set("Microphone")
            self.update_device_visibility()
            self.save_device_cache()
            self.mic_scan_status_var.set(f"Selected: {values[1]}")

    def start_device_sweep(self):
        if self.scan_button.cget("text") == "Stop":
            self.scan_stop_event.set()
            self.scan_button.configure(text="Sweep")
            return
        self.scan_stop_event.clear()
        self.scan_button.configure(text="Stop")
        self.clear_sweep_results()
        threading.Thread(target=self.device_sweep_thread, daemon=True).start()

    def clear_sweep_results(self):
        for item in self.sweep_tree.get_children():
            self.sweep_tree.delete(item)

    def device_sweep_thread(self):
        devices = [(label, dev) for label, dev in self.loopback_map.items() if dev]
        if not devices:
            self.root.after(0, lambda: self.scan_status_var.set("No WASAPI loopback devices found."))
            self.root.after(0, lambda: self.scan_button.configure(text="Sweep"))
            return

        scan_seconds = max(0.25, safe_float(self.scan_seconds_var.get(), 0.8))
        results = []
        for idx, (label, dev) in enumerate(devices, start=1):
            if self.scan_stop_event.is_set():
                break
            self.root.after(0, lambda idx=idx, total=len(devices), label=label: self.scan_status_var.set(f"Sweeping {idx}/{total}: {label}"))
            try:
                rms, peak = self.measure_loopback_device(dev, scan_seconds)
                results.append((rms, peak, label))
                self.root.after(0, lambda rms=rms, peak=peak, label=label: self.add_sweep_result(rms, peak, label))
            except Exception as e:
                self.root.after(0, lambda label=label, e=e: self.add_sweep_result(0.0, 0.0, f"{label} [failed: {e}]"))

        if results and not self.scan_stop_event.is_set():
            results.sort(key=lambda x: audio_score(x[0], x[1]), reverse=True)
            rms, peak, label = results[0]
            self.root.after(0, lambda: self.select_sweep_winner(label, rms, peak))
        self.root.after(0, lambda: self.scan_button.configure(text="Sweep"))

    def measure_loopback_device(self, dev, seconds):
        index = int(dev["index"])
        channels = max(1, int(dev.get("maxInputChannels", 2)))
        source_rate = int(float(dev.get("defaultSampleRate", 48000)))
        frames = max(256, int(source_rate * 0.10))
        chunks = []

        with pyaudio.PyAudio() as p:
            stream = p.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=source_rate,
                frames_per_buffer=frames,
                input=True,
                input_device_index=index,
            )
            try:
                end = time.time() + seconds
                while time.time() < end and not self.scan_stop_event.is_set():
                    data = stream.read(frames, exception_on_overflow=False)
                    chunks.append(mono_from_float32_bytes(data, channels))
            finally:
                stream.stop_stream()
                stream.close()

        if not chunks:
            return 0.0, 0.0
        audio = np.concatenate(chunks)
        return float(np.sqrt(np.mean(audio * audio))), float(np.max(np.abs(audio)))

    def add_sweep_result(self, rms, peak, label):
        self.sweep_tree.insert("", "end", values=(f"{rms:.4f} / {peak:.4f}", label))

    def select_sweep_winner(self, label, rms, peak):
        if label in self.loopback_labels:
            self.loopback_var.set(label)
            if self.capture_mode_var.get() == "Microphone":
                self.capture_mode_var.set("System translate")
            self.update_device_visibility()
            self.save_device_cache()
        self.scan_status_var.set(f"Selected loudest: RMS {rms:.4f}, Peak {peak:.4f}")

    def on_sweep_select(self, event=None):
        selected = self.sweep_tree.selection()
        if not selected:
            return
        values = self.sweep_tree.item(selected[0], "values")
        if len(values) >= 2 and values[1] in self.loopback_labels:
            self.loopback_var.set(values[1])
            if self.capture_mode_var.get() == "Microphone":
                self.capture_mode_var.set("System translate")
            self.update_device_visibility()
            self.save_device_cache()
            self.scan_status_var.set(f"Selected: {values[1]}")

    def pump_text(self):
        for state in [self.system_state, self.mic_state]:
            try:
                while True:
                    msg = state.text_queue.get_nowait()
                    if state.text_widget:
                        state.text_widget.insert("end", msg)
                        state.text_widget.see("end")
            except queue.Empty:
                pass
        self.root.after(100, self.pump_text)

    def update_level_ui(self):
        cutoff = safe_float(self.silence_var.get(), 0.020)
        for state in [self.system_state, self.mic_state]:
            if state.level_var:
                state.level_var.set(f"RMS: {state.latest_rms:.4f} | Peak: {state.latest_peak:.4f} | Silence cutoff: {cutoff:.4f}")
            self.draw_visualizer(state)
            self.update_delay_indicator(state)
        self.root.after(100, self.update_level_ui)

    def draw_visualizer(self, state: SourceState):
        canvas = state.visualizer
        if canvas is None:
            return
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        canvas.delete("all")

        mid = height // 2
        canvas.create_line(0, mid, width, mid, fill="#333333")

        history = state.level_history[-max(20, min(len(state.level_history), width // 3)):]
        if not history:
            return

        bar_count = len(history)
        gap = 2
        bar_w = max(2, (width - gap * (bar_count - 1)) / bar_count)
        x = 0
        for value in history:
            scaled = min(1.0, value / 0.75)
            bar_h = max(2, scaled * (height - 10))
            y0 = mid - bar_h / 2
            y1 = mid + bar_h / 2
            canvas.create_rectangle(x, y0, x + bar_w, y1, fill="#20c45a", outline="")
            x += bar_w + gap

        cutoff = safe_float(self.silence_var.get(), 0.020)
        cutoff_y = mid - min(1.0, cutoff / 0.75) * (height - 10) / 2
        canvas.create_line(0, cutoff_y, width, cutoff_y, fill="#ccaa33", dash=(4, 4))

    def estimated_delay_seconds(self, state: SourceState):
        chunk_s = max(1.0, safe_float(self.chunk_var.get(), 3.0))
        queued = self.queued_audio_seconds(state)
        processing_elapsed = 0.0
        if state.is_processing_chunk and state.processing_started_at > 0:
            processing_elapsed = max(0.0, time.time() - state.processing_started_at)
        return queued + chunk_s + processing_elapsed

    def update_delay_indicator(self, state: SourceState):
        if state.delay_var is None or state.delay_bar is None:
            return
        delay = self.estimated_delay_seconds(state) if self.running and state in self.active_states else 0.0
        if delay < 4:
            label = f"~{delay:.1f}s behind (good)"
        elif delay < 8:
            label = f"~{delay:.1f}s behind"
        else:
            label = f"~{delay:.1f}s behind (stale)"
        state.delay_var.set(label)
        state.delay_bar.configure(value=min(delay, 15.0))

    def watchdog(self):
        if self.running and not self.stop_event.is_set():
            parts = []
            for state in self.active_states:
                parts.append(
                    f"{state.name}: queued {self.queued_audio_seconds(state):.1f}s, process {state.last_chunk_seconds:.1f}s, drops {state.dropped_chunks}"
                )
                if self.auto_catchup_var.get() and self.queued_audio_seconds(state) > max(1.0, safe_float(self.max_lag_var.get(), 6.0)):
                    clear_queue(state.audio_queue)
                    state.dropped_chunks += 1
                    state.text_queue.put("\n[auto catch up: skipped old buffered audio]\n")
            self.status_var.set(" | ".join(parts))

        if self.stop_event.is_set() and self.running:
            self.running = False
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
        self.root.after(1000, self.watchdog)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    LoCaptApp().run()

