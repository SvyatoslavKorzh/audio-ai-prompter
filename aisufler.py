import os
import io
import re
import json
import wave
import time
import datetime
import threading
import numpy as np
import tkinter as tk
from tkinter import ttk
import keyboard
import pyaudiowpatch as pyaudio
from openai import OpenAI

# ================= НАСТРОЙКИ =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_URL = "https://api.groq.com/openai/v1"
KEY_FILE = os.path.join(BASE_DIR, "groq_key.txt")
SETTINGS_FILE = os.path.join(BASE_DIR, "settings.json")

WHISPER_MODEL = "whisper-large-v3-turbo"
HOTKEY = "F9"

VOICE_PROMPT = """
Ты — умный помощник ученика на уроке в Zoom. Тебе передают точную расшифровку вопроса или задания учителя.
Твоя задача:
1. Если в тексте нет вопроса, задания или осмысленной речи (просто музыка, шум, обрывок слова) — ответь ровно одним словом: SKIP
2. Если есть вопрос или задача — дай сразу точный, правильный и краткий ответ (2-4 предложения) на том же языке, на котором задан вопрос (украинском или русском), без вступлений.
"""

MANUAL_PROMPT = """
Ты — умный помощник ученика на уроке в Zoom. Ученик вручную написал тебе вопрос или условие задачи.
Дай сразу точный, правильный и краткий ответ (2-4 предложения) на том же языке, на котором написан запрос (украинском или русском), без лишних вступлений, чтобы можно было сразу зачитать или переписать.
"""

HALLUCINATIONS = [
    "дякую за перегляд",
    "спасибо за просмотр",
    "звучить музика",
    "играет музыка",
    "субтитры создавал",
    "подписывайтесь на канал",
    "продолжение следует",
    "аплодисменты",
    "оплески",
]

# Описания для 3 уровней моделей (подтягиваются только реально доступные на ключе)
MODEL_META = {
    "llama-3.3-70b-versatile": (
        "Llama 3.3 70B — Основная",
        "Умная универсальная модель на 70B. Отлично понимает украинский и русский.\n"
        "Лимит Free: ~1 000 зап/день | 30 зап/мин.",
    ),
    "openai/gpt-oss-120b": (
        "GPT-OSS 120B — Основная",
        "Самая умная открытая модель (120B). Лучше всех для сложных вопросов.\n"
        "Лимит Free: ~1 000 зап/день | 30 зап/мин.",
    ),
    "meta-llama/llama-4-maverick-17b-128e-instruct": (
        "Llama 4 Maverick — Основная",
        "Старшая Llama 4. Хороший баланс ума и скорости по всем предметам.\n"
        "Лимит Free: ~1 000 зап/день | 30 зап/мин.",
    ),
    "qwen/qwen3-32b": (
        "Qwen 3 32B — Основная",
        "Сильна в математике, физике и логике (размышления скрыты).\n"
        "Лимит Free: ~1 000 зап/день | 60 зап/мин.",
    ),
    "openai/gpt-oss-20b": (
        "GPT-OSS 20B — Быстрая",
        "Облегчённая и быстрая модель (20B). Проще основной, но отвечает быстрее.\n"
        "Лимит Free: ~1 000 зап/день | 30 зап/мин.",
    ),
    "meta-llama/llama-4-scout-17b-16e-instruct": (
        "Llama 4 Scout — Быстрая",
        "Свежая быстрая модель. Хорошо подходит для обычных школьных вопросов.\n"
        "Лимит Free: ~1 000 зап/день | 30 зап/мин | 500K ток/день.",
    ),
    "llama-3.1-8b-instant": (
        "Llama 3.1 8B — Турбо",
        "Самая простая и мгновенная модель (8B). Огромный запас бесплатных запросов.\n"
        "Лимит Free: ~14 400 зап/день | 30 зап/мин | 500K ток/день.",
    ),
}

# Кандидаты на 3 слота: [1: Основная, 2: Быстрая, 3: Турбо]
TIER_CANDIDATES = [
    [
        "llama-3.3-70b-versatile",
        "openai/gpt-oss-120b",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "qwen/qwen3-32b",
    ],
    [
        "openai/gpt-oss-20b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "qwen/qwen3-32b",
    ],
    [
        "llama-3.1-8b-instant",
    ],
]
# =============================================

client = None
LLM_MODEL = "llama-3.1-8b-instant"
available_models_list = []

is_recording = False
recorded_chunks = []
buffer_lock = threading.Lock()

audio_lock = threading.Lock()
pa_instance = None
audio_stream = None
selected_device_name = "DEFAULT"
active_device_label = "По умолчанию (Windows)"
sample_rate = 16000
channels = 1

usage_lock = threading.Lock()
usage_data = {
    "date": "",
    "requests": 0,
    "tokens": 0,
    "audio_sec": 0.0,
    "groq_limits": {},  # {model_id: {"rem_req": "...", "lim_req": "..."}}
}


def today_str():
    return datetime.date.today().isoformat()


def load_saved_key():
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def save_key(key):
    with open(KEY_FILE, "w", encoding="utf-8") as f:
        f.write(key.strip())


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def init_usage():
    global usage_data
    saved = load_settings().get("usage", {})
    today = today_str()
    with usage_lock:
        if saved.get("date") == today:
            usage_data = {
                "date": today,
                "requests": int(saved.get("requests", 0)),
                "tokens": int(saved.get("tokens", 0)),
                "audio_sec": float(saved.get("audio_sec", 0.0)),
                "groq_limits": dict(saved.get("groq_limits", {})),
            }
        else:
            usage_data = {
                "date": today,
                "requests": 0,
                "tokens": 0,
                "audio_sec": 0.0,
                "groq_limits": {},
            }


def save_settings():
    with usage_lock:
        usage_copy = dict(usage_data)
    data = {
        "model": LLM_MODEL,
        "audio_device": selected_device_name,
        "usage": usage_copy,
    }
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def record_usage(model_id=None, tokens_used=0, audio_seconds=0.0, headers=None, count_request=False):
    today = today_str()
    with usage_lock:
        if usage_data["date"] != today:
            usage_data["date"] = today
            usage_data["requests"] = 0
            usage_data["tokens"] = 0
            usage_data["audio_sec"] = 0.0
            usage_data["groq_limits"] = {}

        if count_request:
            usage_data["requests"] += 1
        if tokens_used:
            usage_data["tokens"] += int(tokens_used)
        if audio_seconds:
            usage_data["audio_sec"] += float(audio_seconds)

        if model_id and headers is not None:
            rem_req = headers.get("x-ratelimit-remaining-requests")
            lim_req = headers.get("x-ratelimit-limit-requests")
            if rem_req is not None and lim_req is not None:
                usage_data["groq_limits"][model_id] = {
                    "rem_req": str(rem_req),
                    "lim_req": str(lim_req),
                }
    save_settings()


def get_usage_bar_text():
    with usage_lock:
        req = usage_data["requests"]
        tok = usage_data["tokens"]
        aud = int(round(usage_data["audio_sec"]))
        lim_info = usage_data["groq_limits"].get(LLM_MODEL)

    tok_str = f"{tok / 1000:.1f}K" if tok >= 1000 else str(tok)
    base = f"Расход сегодня: {req} зап. | {tok_str} ток. | {aud}с аудио"
    if lim_info:
        base += f"   •   Осталось: {lim_info['rem_req']}/{lim_info['lim_req']}"
    return base


def build_model_entry(model_id, tier_index=None):
    if model_id in MODEL_META:
        title, desc = MODEL_META[model_id]
        return {"id": model_id, "title": title, "desc": desc}
    labels = ["Основная", "Быстрая", "Турбо"]
    tag = labels[tier_index] if tier_index is not None and tier_index < 3 else "Доп."
    return {
        "id": model_id,
        "title": f"{model_id} — {tag}",
        "desc": "Доступная модель Groq API.\nЛимиты отображаются в строке расхода после первого запроса.",
    }


def init_groq_client(api_key):
    """Проверяет ключ и собирает ровно 3 модели: основную + 2 более быстрые"""
    global client, LLM_MODEL, available_models_list
    test_client = OpenAI(base_url=LLM_URL, api_key=api_key)
    api_ids = [m.id for m in test_client.models.list().data]

    picked_ids = []
    curated = []

    # Подбираем по одной лучшей модели на каждый из 3 уровней
    for tier_idx, candidates in enumerate(TIER_CANDIDATES):
        for cid in candidates:
            if cid in api_ids and cid not in picked_ids:
                picked_ids.append(cid)
                curated.append(build_model_entry(cid, tier_idx))
                break

    # Если каких-то слотов не хватило (Groq обновил названия) — добиваем до 3 доступными чат-моделями
    if len(curated) < 3:
        for mid in api_ids:
            if mid not in picked_ids and not any(
                x in mid.lower() for x in ["whisper", "guard", "tts", "audio", "compound"]
            ):
                picked_ids.append(mid)
                curated.append(build_model_entry(mid, len(curated)))
            if len(curated) >= 3:
                break

    available_models_list = curated
    available_ids = [m["id"] for m in curated]

    saved_model = load_settings().get("model")
    if saved_model in available_ids:
        LLM_MODEL = saved_model
    elif available_ids:
        LLM_MODEL = available_ids[0]
    else:
        LLM_MODEL = "llama-3.1-8b-instant"

    client = test_client


def get_model_info(model_id):
    for m in available_models_list:
        if m["id"] == model_id:
            return m
    return build_model_entry(model_id)


def get_loopback_devices():
    """Список всех устройств вывода звука в Windows"""
    devices = [("DEFAULT", "По умолчанию (как в Windows)")]
    temp_pa = pyaudio.PyAudio()
    try:
        for i in range(temp_pa.get_device_count()):
            dev = temp_pa.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice"):
                name = dev["name"]
                clean_name = name.replace(" [Loopback]", "")
                devices.append((name, clean_name))
    finally:
        temp_pa.terminate()
    return devices


def audio_callback(in_data, frame_count, time_info, status):
    if is_recording:
        with buffer_lock:
            recorded_chunks.append(in_data)
    return (None, pyaudio.paContinue)


def stop_audio_stream():
    global pa_instance, audio_stream
    with audio_lock:
        if audio_stream is not None:
            try:
                audio_stream.stop_stream()
                audio_stream.close()
            except Exception:
                pass
            audio_stream = None
        if pa_instance is not None:
            try:
                pa_instance.terminate()
            except Exception:
                pass
            pa_instance = None


def start_audio_stream(device_name="DEFAULT"):
    """Запуск или переключение перехвата системного звука"""
    global pa_instance, audio_stream, sample_rate, channels, selected_device_name, active_device_label
    stop_audio_stream()

    with audio_lock:
        pa_instance = pyaudio.PyAudio()
        target_dev = None

        if device_name and device_name != "DEFAULT":
            for i in range(pa_instance.get_device_count()):
                dev = pa_instance.get_device_info_by_index(i)
                if dev.get("isLoopbackDevice") and dev["name"] == device_name:
                    target_dev = dev
                    break

        if target_dev is None:
            target_dev = pa_instance.get_default_wasapi_loopback()
            selected_device_name = "DEFAULT"
            active_device_label = f"По умолчанию ({target_dev['name'].replace(' [Loopback]', '')})"
        else:
            selected_device_name = target_dev["name"]
            active_device_label = target_dev["name"].replace(" [Loopback]", "")

        sample_rate = int(target_dev["defaultSampleRate"])
        channels = target_dev["maxInputChannels"]

        audio_stream = pa_instance.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=target_dev["index"],
            frames_per_buffer=1024,
            stream_callback=audio_callback,
        )
        audio_stream.start_stream()


def ask_for_api_key(parent=None, force_new=False):
    """Окно ввода и сохранения API-ключа Groq"""
    if not force_new:
        saved = load_saved_key()
        if saved:
            try:
                init_groq_client(saved)
                return True
            except Exception:
                pass

    dialog = tk.Toplevel(parent) if parent else tk.Tk()
    dialog.title("Настройка Groq API")
    dialog.geometry("460x215+100+100")
    dialog.configure(bg="#181818")
    dialog.attributes("-topmost", True)
    dialog.resizable(False, False)

    result = {"ok": False}

    tk.Label(
        dialog,
        text="Вставь свой API-ключ Groq (с console.groq.com):",
        fg="#ffffff",
        bg="#181818",
        font=("Segoe UI", 10, "bold"),
        pady=12,
    ).pack()

    entry = tk.Entry(
        dialog,
        width=48,
        font=("Consolas", 10),
        bg="#262626",
        fg="#ffffff",
        insertbackground="#ffffff",
        relief="flat",
    )
    entry.pack(ipady=6, padx=20)
    entry.insert(0, load_saved_key())
    entry.focus_set()

    def paste_key(e):
        if e.keycode == 86:
            entry.event_generate("<<Paste>>")
            return "break"

    entry.bind("<Control-KeyPress>", paste_key)

    status_lbl = tk.Label(
        dialog,
        text="Ключ сохранится в groq_key.txt рядом со скриптом",
        fg="#888888",
        bg="#181818",
        font=("Segoe UI", 9),
    )
    status_lbl.pack(pady=(8, 4))

    def on_save(event=None):
        key = entry.get().strip()
        if not key.startswith("gsk_"):
            status_lbl.config(text="❌ Ключ должен начинаться на gsk_...", fg="#ff6b6b")
            return
        status_lbl.config(text="⏳ Проверяю ключ...", fg="#ffffff")
        dialog.update()
        try:
            init_groq_client(key)
            save_key(key)
            result["ok"] = True
            dialog.destroy()
        except Exception:
            status_lbl.config(text="❌ Неверный ключ или нет интернета", fg="#ff6b6b")

    entry.bind("<Return>", on_save)

    tk.Button(
        dialog,
        text="Сохранить и продолжить",
        command=on_save,
        bg="#2d5af0",
        fg="#ffffff",
        font=("Segoe UI", 10, "bold"),
        relief="flat",
        padx=16,
        pady=6,
        cursor="hand2",
    ).pack(pady=8)

    if parent:
        dialog.grab_set()
        parent.wait_window(dialog)
    else:
        dialog.mainloop()

    return result["ok"]


def open_settings_dialog(parent, on_applied_callback):
    """Окно настроек: звук, 3 модели и статистика расхода"""
    win = tk.Toplevel(parent)
    win.title("Настройки AI Суфлёра")
    win.geometry("500x460+90+90")
    win.configure(bg="#181818")
    win.attributes("-topmost", True)
    win.resizable(False, False)

    style = ttk.Style(win)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure(
        "Dark.TCombobox",
        fieldbackground="#262626",
        background="#333333",
        foreground="#ffffff",
        arrowcolor="#ffffff",
        borderwidth=0,
    )

    # --- БЛОК 1: ЗВУК ВЫХОДА ---
    tk.Label(
        win,
        text="🔊 Устройство вывода звука (откуда слушать Zoom):",
        fg="#ffffff",
        bg="#181818",
        font=("Segoe UI", 10, "bold"),
        anchor="w",
    ).pack(fill="x", padx=16, pady=(14, 4))

    audio_row = tk.Frame(win, bg="#181818")
    audio_row.pack(fill="x", padx=16)

    devices_data = []
    device_var = tk.StringVar()
    device_combo = ttk.Combobox(
        audio_row,
        textvariable=device_var,
        state="readonly",
        style="Dark.TCombobox",
        font=("Segoe UI", 9),
    )
    device_combo.pack(side="left", fill="x", expand=True, ipady=3)

    def refresh_devices():
        nonlocal devices_data
        devices_data = get_loopback_devices()
        labels = [d[1] for d in devices_data]
        device_combo["values"] = labels
        current_idx = 0
        for idx, (raw_name, _) in enumerate(devices_data):
            if raw_name == selected_device_name:
                current_idx = idx
                break
        device_combo.current(current_idx)

    tk.Button(
        audio_row,
        text="🔄",
        command=refresh_devices,
        bg="#2b2b2b",
        fg="#ffffff",
        font=("Segoe UI", 9),
        relief="flat",
        padx=8,
        pady=3,
        cursor="hand2",
    ).pack(side="right", padx=(6, 0))

    refresh_devices()

    # --- БЛОК 2: 3 МОДЕЛИ ---
    tk.Label(
        win,
        text="🧠 Модель нейросети (3 варианта: основная + 2 быстрее):",
        fg="#ffffff",
        bg="#181818",
        font=("Segoe UI", 10, "bold"),
        anchor="w",
    ).pack(fill="x", padx=16, pady=(14, 4))

    ordered_models = list(available_models_list)
    model_titles = [
        f"★ {m['title']} (выбрана)" if m["id"] == LLM_MODEL else m["title"]
        for m in ordered_models
    ]

    model_var = tk.StringVar()
    model_combo = ttk.Combobox(
        win,
        textvariable=model_var,
        values=model_titles,
        state="readonly",
        style="Dark.TCombobox",
        font=("Segoe UI", 9),
    )
    model_combo.pack(fill="x", padx=16, ipady=3)

    selected_m_idx = 0
    for idx, m in enumerate(ordered_models):
        if m["id"] == LLM_MODEL:
            selected_m_idx = idx
            break
    if ordered_models:
        model_combo.current(selected_m_idx)

    info_box = tk.Label(
        win,
        text="",
        fg="#cccccc",
        bg="#222222",
        font=("Segoe UI", 9),
        wraplength=445,
        justify="left",
        anchor="nw",
        padx=12,
        pady=8,
    )
    info_box.pack(fill="x", padx=16, pady=(8, 10))

    def update_model_desc(event=None):
        idx = model_combo.current()
        if 0 <= idx < len(ordered_models):
            m = ordered_models[idx]
            with usage_lock:
                lim = usage_data["groq_limits"].get(m["id"])
            lim_line = (
                f"\nОсталось сегодня (данные Groq): {lim['rem_req']} из {lim['lim_req']} зап."
                if lim
                else "\nОстаток от Groq появится после первого запроса этой моделью."
            )
            info_box.config(text=f"ID: {m['id']}\n{m['desc']}{lim_line}")

    model_combo.bind("<<ComboboxSelected>>", update_model_desc)
    update_model_desc()

    # --- БЛОК 3: СТАТИСТИКА РАСХОДА ---
    tk.Label(
        win,
        text="📊 Расход за сегодня:",
        fg="#ffffff",
        bg="#181818",
        font=("Segoe UI", 10, "bold"),
        anchor="w",
    ).pack(fill="x", padx=16, pady=(4, 4))

    stats_lbl = tk.Label(
        win,
        text="",
        fg="#cccccc",
        bg="#222222",
        font=("Segoe UI", 9),
        justify="left",
        anchor="nw",
        padx=12,
        pady=8,
    )
    stats_lbl.pack(fill="x", padx=16)

    def refresh_stats_box():
        with usage_lock:
            req = usage_data["requests"]
            tok = usage_data["tokens"]
            aud = int(round(usage_data["audio_sec"]))
            dt = usage_data["date"]
        stats_lbl.config(
            text=(
                f"Дата: {dt}\n"
                f"Запросов к ИИ: {req}   |   Токенов: {tok}\n"
                f"Распознано аудио (Whisper): {aud} сек (лимит Free ~7 200 сек/день)"
            )
        )

    refresh_stats_box()

    status_msg = tk.Label(win, text="", fg="#ff6b6b", bg="#181818", font=("Segoe UI", 9))
    status_msg.pack(pady=(4, 0))

    def apply_settings():
        global LLM_MODEL
        m_idx = model_combo.current()
        if 0 <= m_idx < len(ordered_models):
            LLM_MODEL = ordered_models[m_idx]["id"]

        d_idx = device_combo.current()
        if 0 <= d_idx < len(devices_data):
            chosen_dev_name = devices_data[d_idx][0]
            try:
                start_audio_stream(chosen_dev_name)
            except Exception as e:
                status_msg.config(text=f"Ошибка аудиоустройства: {e}")
                return

        save_settings()
        on_applied_callback()
        win.destroy()

    tk.Button(
        win,
        text="Применить настройки",
        command=apply_settings,
        bg="#2d5af0",
        fg="#ffffff",
        font=("Segoe UI", 10, "bold"),
        relief="flat",
        padx=18,
        pady=7,
        cursor="hand2",
    ).pack(pady=(6, 12))

    win.grab_set()
    parent.wait_window(win)


def make_wav_in_memory(raw_bytes):
    audio_np = np.frombuffer(raw_bytes, dtype=np.int16)
    if channels > 1:
        audio_np = audio_np.reshape(-1, channels).mean(axis=1).astype(np.int16)

    duration_sec = len(audio_np) / float(sample_rate)

    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_np.tobytes())
    wav_io.seek(0)
    wav_io.name = "speech.wav"
    return wav_io, duration_sec


def clean_llm_answer(text):
    """Убирает теги <think>...</think> у рассуждающих моделей"""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned or text.strip()


def call_llm_with_stats(system_prompt, user_text):
    """Вызывает LLM и обновляет счётчик расхода + реальные лимиты из заголовков Groq"""
    raw_resp = client.chat.completions.with_raw_response.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        temperature=0.2,
    )
    headers = raw_resp.headers
    parsed = raw_resp.parse()
    tokens_used = parsed.usage.total_tokens if getattr(parsed, "usage", None) else 0
    record_usage(
        model_id=LLM_MODEL,
        tokens_used=tokens_used,
        headers=headers,
        count_request=True,
    )
    return clean_llm_answer(parsed.choices[0].message.content)


def process_audio(raw_data, set_status, refresh_usage_ui):
    audio_check = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
    if len(audio_check) < sample_rate * channels * 0.4:
        set_status("⚠️ Слишком короткое нажатие. Удерживай кнопку, пока идёт вопрос.", "#aaaaaa")
        return

    rms = np.sqrt(np.mean(audio_check ** 2))
    if rms < 40:
        set_status(
            f"🔇 На устройстве «{active_device_label}» была тишина.\n"
            f"Если звук идёт в другие наушники/колонки — выбери их в ⚙️.",
            "#aaaaaa",
        )
        return

    set_status("⏳ Распознаю речь...", "#ffffff")
    try:
        wav_file, duration_sec = make_wav_in_memory(raw_data)
        transcription = client.audio.transcriptions.create(
            file=wav_file,
            model=WHISPER_MODEL,
        )
        record_usage(audio_seconds=duration_sec)
        refresh_usage_ui()

        transcript = transcription.text.strip()

        clean_lower = transcript.lower().strip(".,!?[]() ")
        if not transcript or any(h in clean_lower for h in HALLUCINATIONS):
            set_status("🔇 Речь не распознана (была музыка или тишина).", "#aaaaaa")
            return

        set_status(f"🎙 Вопрос: {transcript}\n\n⏳ Думаю ({LLM_MODEL})...", "#ffffff")

        answer = call_llm_with_stats(VOICE_PROMPT, transcript)
        refresh_usage_ui()

        if answer.upper().startswith("SKIP"):
            set_status(f"💤 В тексте нет вопроса.\n\n🎙 Услышал: {transcript}", "#aaaaaa")
        else:
            set_status(f"💡 ОТВЕТ:\n{answer}\n\n────────────────────\n🎙 Вопрос: {transcript}", "#ffffff")

    except Exception as e:
        set_status(f"Ошибка Groq:\n{e}", "#ff6b6b")


def process_manual_text(query_text, set_status, refresh_usage_ui):
    """Обработка ручного текстового запроса"""
    set_status(f"⌨️ Запрос: {query_text}\n\n⏳ Думаю ({LLM_MODEL})...", "#ffffff")
    try:
        answer = call_llm_with_stats(MANUAL_PROMPT, query_text)
        refresh_usage_ui()
        set_status(f"💡 ОТВЕТ:\n{answer}\n\n────────────────────\n⌨️ Запрос: {query_text}", "#ffffff")
    except Exception as e:
        set_status(f"Ошибка Groq:\n{e}", "#ff6b6b")


def start_app():
    init_usage()

    if not ask_for_api_key():
        return

    saved_dev = load_settings().get("audio_device", "DEFAULT")
    audio_error = ""
    try:
        start_audio_stream(saved_dev)
    except Exception as e:
        audio_error = f"\n⚠️ Ошибка аудио: {e} (выбери устройство в ⚙️)"

    root = tk.Tk()
    root.title("AI Суфлёр (Push-to-Talk)")
    root.geometry("560x460+50+50")
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.94)
    root.configure(bg="#181818")

    # Верхняя панель: кнопка записи + настройки + ключ
    top_frame = tk.Frame(root, bg="#181818")
    top_frame.pack(fill="x", padx=12, pady=(12, 6))

    btn = tk.Button(
        top_frame,
        text=f"🎙 ЗАЖМИ И ДЕРЖИ (или {HOTKEY})",
        bg="#2d5af0",
        fg="#ffffff",
        activebackground="#d9383a",
        activeforeground="#ffffff",
        font=("Segoe UI", 11, "bold"),
        relief="flat",
        pady=10,
        cursor="hand2",
    )
    btn.pack(side="left", fill="x", expand=True)

    def get_ready_text():
        m_title = get_model_info(LLM_MODEL)["title"]
        return (
            f"Готов!\n"
            f"🧠 Модель: {m_title}\n"
            f"🔊 Звук: {active_device_label}\n\n"
            f"• Зажми синюю кнопку (или {HOTKEY}), пока звучит вопрос.\n"
            f"• Или впиши вопрос вручную снизу и нажми Enter."
            f"{audio_error}"
        )

    def set_status(text, color="#ffffff"):
        try:
            root.after(0, lambda: text_label.config(text=text, fg=color))
        except Exception:
            pass

    def refresh_usage_ui():
        try:
            root.after(0, lambda: usage_bar.config(text=get_usage_bar_text()))
        except Exception:
            pass

    def on_settings_applied():
        set_status(get_ready_text(), "#cccccc")
        refresh_usage_ui()

    settings_btn = tk.Button(
        top_frame,
        text="⚙️",
        command=lambda: open_settings_dialog(root, on_settings_applied),
        bg="#2b2b2b",
        fg="#cccccc",
        font=("Segoe UI", 11),
        relief="flat",
        padx=10,
        pady=10,
        cursor="hand2",
    )
    settings_btn.pack(side="right", padx=(6, 0))

    key_btn = tk.Button(
        top_frame,
        text="🔑",
        command=lambda: ask_for_api_key(parent=root, force_new=True),
        bg="#2b2b2b",
        fg="#cccccc",
        font=("Segoe UI", 11),
        relief="flat",
        padx=10,
        pady=10,
        cursor="hand2",
    )
    key_btn.pack(side="right", padx=(6, 0))

    # Нижняя панель: ручной ввод запроса
    bottom_frame = tk.Frame(root, bg="#181818")
    bottom_frame.pack(side="bottom", fill="x", padx=12, pady=(4, 12))

    manual_entry = tk.Entry(
        bottom_frame,
        bg="#262626",
        fg="#ffffff",
        insertbackground="#ffffff",
        font=("Segoe UI", 10),
        relief="flat",
    )
    manual_entry.pack(side="left", fill="x", expand=True, ipady=7, padx=(0, 6))

    def handle_shortcuts(event):
        if event.keycode == 86:  # Ctrl+V в любой раскладке
            manual_entry.event_generate("<<Paste>>")
            return "break"
        elif event.keycode == 67:  # Ctrl+C в любой раскладке
            manual_entry.event_generate("<<Copy>>")
            return "break"

    manual_entry.bind("<Control-KeyPress>", handle_shortcuts)

    # Полоска расхода прямо над полем ввода
    usage_bar = tk.Label(
        root,
        text=get_usage_bar_text(),
        fg="#888888",
        bg="#141414",
        font=("Segoe UI", 8),
        anchor="w",
        padx=12,
        pady=4,
    )
    usage_bar.pack(side="bottom", fill="x", padx=12, pady=(2, 2))

    # Центральная область вывода
    text_label = tk.Label(
        root,
        text=get_ready_text(),
        fg="#cccccc",
        bg="#181818",
        font=("Segoe UI", 11),
        wraplength=530,
        justify="left",
        anchor="nw",
        padx=14,
        pady=10,
    )
    text_label.pack(fill="both", expand=True)

    def send_manual(event=None):
        query = manual_entry.get().strip()
        if not query:
            return
        manual_entry.delete(0, tk.END)
        threading.Thread(
            target=process_manual_text,
            args=(query, set_status, refresh_usage_ui),
            daemon=True,
        ).start()

    manual_entry.bind("<Return>", send_manual)

    send_btn = tk.Button(
        bottom_frame,
        text="➤",
        command=send_manual,
        bg="#2d5af0",
        fg="#ffffff",
        font=("Segoe UI", 10, "bold"),
        relief="flat",
        padx=14,
        pady=5,
        cursor="hand2",
    )
    send_btn.pack(side="right")

    def start_rec(event=None):
        global is_recording, recorded_chunks
        if is_recording:
            return
        with buffer_lock:
            recorded_chunks = []
        is_recording = True
        btn.config(text="🔴 ИДЁТ ЗАПИСЬ... ОТПУСТИ ДЛЯ ОТВЕТА", bg="#d9383a")
        set_status(f"Слушаю: {active_device_label}...", "#ff7b7b")

    def stop_rec(event=None):
        global is_recording
        if not is_recording:
            return
        is_recording = False
        btn.config(text=f"🎙 ЗАЖМИ И ДЕРЖИ (или {HOTKEY})", bg="#2d5af0")
        with buffer_lock:
            raw_data = b"".join(recorded_chunks)
        threading.Thread(
            target=process_audio,
            args=(raw_data, set_status, refresh_usage_ui),
            daemon=True,
        ).start()

    btn.bind("<ButtonPress-1>", start_rec)
    btn.bind("<ButtonRelease-1>", stop_rec)

    def hotkey_watcher():
        while True:
            keyboard.wait(HOTKEY)
            try:
                root.after(0, start_rec)
            except Exception:
                break
            while keyboard.is_pressed(HOTKEY):
                time.sleep(0.05)
            try:
                root.after(0, stop_rec)
            except Exception:
                break

    def on_close():
        stop_audio_stream()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    threading.Thread(target=hotkey_watcher, daemon=True).start()
    root.mainloop()


if __name__ == "__main__":
    start_app()