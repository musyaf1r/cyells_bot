import json
import os
import re
import base64
import io
import time
import ctypes
import hashlib
import threading
import queue
import tkinter as tk
from tkinter import ttk

import mss
import numpy as np
from PIL import Image, ImageDraw
import pyautogui
from google import genai
from google.genai import types
from dotenv import load_dotenv
import keyboard
from deteksi import deteksi, gambar_overlay, ringkas

# =============================================
# KONFIGURASI (dari file .env)
# =============================================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise SystemExit("[!] Isi GOOGLE_API_KEY di file .env. Jangan tempelkan kunci API di kode.")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
DEBUG_KLIK = os.getenv("DEBUG_KLIK", "1") == "1"        # simpan debug_klik.png & debug_ai.jpg
MIN_JAWAB = int(os.getenv("MIN_JAWAB", "1"))            # minimal klik jawaban per halaman sebelum Next
VIDEO_WAIT = float(os.getenv("VIDEO_WAIT", "6"))        # detik menunggu video setelah di-play
AUDIO_WAIT = float(os.getenv("AUDIO_WAIT", "2.5"))      # detik menunggu setelah klik speaker
AI_TIMEOUT = max(5.0, float(os.getenv("AI_TIMEOUT", "45")))  # batas tunggu respons API

MAX_STEPS_PER_PAGE = 15   # batas langkah per halaman sebelum paksa Next
STUCK_LIMIT = 3           # layar sama sekian kali setelah beraksi -> paksa Next
NO_SOAL_LIMIT = 4         # AI bilang "tidak ada soal" sekian kali -> coba Next

google_client = genai.Client(
    api_key=GOOGLE_API_KEY,
    http_options=types.HttpOptions(timeout=int(AI_TIMEOUT * 1000)),
)

# Supaya koordinat benar di Windows dengan display scaling
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

pyautogui.FAILSAFE = True   # mouse ke pojok kiri atas = stop paksa
pyautogui.PAUSE = 0.05

screen_w, screen_h = pyautogui.size()
is_running = False
status_events = queue.Queue()
status_state = {
    "stage": "Siap",
    "detail": "Tekan F8 atau tombol Mulai untuk menjalankan bot.",
    "level": "ready",
    "started_at": None,
}


def set_status(stage, detail="", level="working"):
    """Kirim status kerja ke panel UI tanpa mengganggu thread bot."""
    started_at = time.monotonic() if level == "working" else None
    status_events.put(("status", stage, detail, level, started_at))

# =============================================
# STATE PER HALAMAN
# =============================================
class Halaman:
    def __init__(self):
        self.reset()

    def reset(self):
        self.riwayat = []
        self.langkah = 0
        self.jawab = 0                 # jumlah klik jawaban yang sudah dilakukan
        self.butuh = MIN_JAWAB         # minimal jawaban sebelum Next boleh
        self.diputar = False           # video sudah di-play
        self.audio_diklik = False      # speaker sudah diklik
        self.tipe = "UNKNOWN"
        self.last_hash = None
        self.sama = 0
        self.tanpa_soal = 0

# =============================================
# SCREENSHOT
# =============================================
def capture_screen(simpan=False):
    """Return (base64 JPEG ber-ID, hash layar, elemen terdeteksi)."""
    set_status("Mengambil screenshot", "Membaca tampilan layar saat ini.")
    with mss.MSS() as sct:
        monitor = {"top": 0, "left": 0, "width": screen_w, "height": screen_h}
        sct_img = sct.grab(monitor)
        asli = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

    kecil = asli.resize((64, 36)).convert("L")
    layar_hash = hashlib.md5(kecil.tobytes()).hexdigest()
    set_status("Mendeteksi elemen", "Mencari tombol, opsi, gambar, teks, dan kolom kosong.")
    hasil = deteksi(np.asarray(asli)[:, :, ::-1].copy())
    elems = hasil["elems"]

    skala = min(1.0, 1280 / asli.width)
    if skala < 1:
        img = asli.resize((1280, int(asli.height * skala)), Image.LANCZOS)
    else:
        img = asli.copy()
    img = gambar_overlay(img, elems, skala)

    if simpan and DEBUG_KLIK:
        img.save("debug_ai.jpg", quality=85)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    set_status("Screenshot siap", f"{len(elems)} elemen diberi ID.", "ready")
    return base64.b64encode(buffer.getvalue()).decode("utf-8"), layar_hash, elems


def hash_screen():
    """Ambil hash layar tanpa menjalankan deteksi, overlay, atau encoding gambar."""
    with mss.MSS() as sct:
        monitor = {"top": 0, "left": 0, "width": screen_w, "height": screen_h}
        sct_img = sct.grab(monitor)
    img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
    kecil = img.resize((64, 36)).convert("L")
    return hashlib.md5(kecil.tobytes()).hexdigest()

# =============================================
# TANYA AI
# =============================================
def extract_json(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = raw.replace("```json", "").replace("```", "").strip()
    awal = raw.find("{")
    akhir = raw.rfind("}")
    if awal == -1 or akhir == -1:
        raise json.JSONDecodeError("JSON tidak ditemukan", raw, 0)
    return json.loads(raw[awal:akhir + 1])


def buat_prompt_jawaban(hal, elems):
    hist = "; ".join(hal.riwayat[-8:]) if hal.riwayat else "(belum ada)"
    daftar_elemen = ringkas(elems) or "(belum ada elemen terdeteksi)"
    return (
        "Kamu menyelesaikan quiz bahasa Inggris dengan screenshot dan ID elemen yang terdeteksi. "
        "Analisis soal, tentukan jawaban, lalu pilih SATU aksi berikutnya dari ID aktif. "
        "Jangan pernah membuat koordinat, box, x/y, atau ID yang tidak tercantum. "
        "Titik mouse dihitung program lokal dari pusat kotak ID; kamu tidak menggerakkan mouse.\n"
        "Nomor overlay menunjuk elemen: box=kontrol/chip, img=gambar, teks=baris opsi, "
        "blank=kolom kosong, next=tombol Next.\n"
        f"Tipe sebelumnya={hal.tipe}; jawaban sudah dilakukan={hal.jawab}; "
        f"minimal sebelum Next={hal.butuh}; langkah={hal.langkah}; riwayat={hist}.\n"
        f"Daftar ID aktif: {daftar_elemen}\n\n"
        "Aturan:\n"
        "- Jawaban pilihan: klik ID teks/gambar yang sesuai dengan jawaban.\n"
        "- Dropdown tertutup: klik kontrol saja; setelah screenshot baru pilih teks opsi.\n"
        "- Word bank: drag dari ID kata ke ID blank.\n"
        "- Isian teks: ketik jawaban pada ID kolom.\n"
        "- Audio: klik ID speaker sekali sebagai answer:false, lalu evaluasi ulang setelah screenshot baru.\n"
        "- Video: klik ID play jika belum berjalan; jika sudah berjalan gunakan ID next.\n"
        "- Popup: klik Continue/OK sebagai answer:false.\n"
        "- Lakukan MAKSIMAL SATU aksi yang mengubah layar. ID hanya berlaku untuk screenshot ini.\n"
        "- Untuk Next, gunakan ID bertipe next yang terlihat. Jangan Next jika belum mencapai minimal jawaban.\n"
        "- Jika loading atau tidak ada soal: ada_soal=false dan actions=[]; jangan menebak klik.\n"
        "Kembalikan JSON saja, dengan bentuk:\n"
        '{"ada_soal":true,"tipe":"RADIO|PICTURE|WORDBANK|DROPDOWN|TEXT|AUDIO|VIDEO|POPUP|UNKNOWN",'
        '"soal":"teks pertanyaan singkat","jawaban":"jawaban final atau null",'
        '"keterangan":"singkat","actions":[{"type":"click","id":4,'
        '"label":"opsi jawaban","answer":true}]}'
    )


def jawab_soal_google(img_base64, hal, elems):
    """Gemini memahami screenshot, menjawab soal, dan memilih satu aksi berbasis ID."""
    set_status(
        "Menganalisis soal dengan Gemini",
        f"Model {GEMINI_MODEL}; batas tunggu {AI_TIMEOUT:g} detik.",
    )
    mulai = time.monotonic()
    gambar = Image.open(io.BytesIO(base64.b64decode(img_base64)))
    response = google_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[gambar, buat_prompt_jawaban(hal, elems)],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
            max_output_tokens=600,
        ),
    )
    raw = response.text
    if not raw:
        raise ValueError("Gemini mengembalikan respons kosong.")
    data = extract_json(raw)
    if not isinstance(data, dict):
        raise ValueError("Respons Gemini harus berupa objek JSON.")
    data.setdefault("ada_soal", False)
    data.setdefault("tipe", "UNKNOWN")
    data.setdefault("jawaban", None)
    data.setdefault("soal", "")
    data.setdefault("keterangan", "")
    data.setdefault("actions", [])
    set_status(
        "Gemini selesai menjawab dan memilih aksi",
        f"{time.monotonic() - mulai:.1f} detik; tipe={data['tipe']}; jawaban={data['jawaban']}",
        "ready",
    )
    print(f"[i] Jawaban dan aksi Gemini: {data}")
    return data

# =============================================
# AKSI MOUSE / KEYBOARD
# =============================================
def cari_elemen(id_elemen, elems, jenis=None):
    """Resolve ID dari screenshot aktif; ID tak valid tidak pernah menjadi koordinat."""
    if isinstance(id_elemen, bool) or not isinstance(id_elemen, int):
        raise ValueError(f"ID elemen harus bilangan bulat, diterima: {id_elemen!r}")
    for elem in elems:
        if elem["id"] == id_elemen and (jenis is None or elem["kind"] == jenis):
            return elem
    jenis_info = f" bertipe {jenis}" if jenis else ""
    raise ValueError(f"ID elemen {id_elemen}{jenis_info} tidak ada pada screenshot saat ini.")


def simpan_debug(x, y):
    if not DEBUG_KLIK:
        return
    try:
        with mss.MSS() as sct:
            s = sct.grab({"top": 0, "left": 0, "width": screen_w, "height": screen_h})
            img = Image.frombytes("RGB", s.size, s.bgra, "raw", "BGRX")
        d = ImageDraw.Draw(img)
        d.ellipse([x - 18, y - 18, x + 18, y + 18], outline=(255, 0, 0), width=4)
        d.line([(x - 30, y), (x + 30, y)], fill=(255, 0, 0), width=2)
        d.line([(x, y - 30), (x, y + 30)], fill=(255, 0, 0), width=2)
        img.save("debug_klik.png")
    except Exception:
        pass


def klik(x, y, label="", dobel=False):
    simpan_debug(x, y)
    pyautogui.moveTo(x, y, duration=0.12)
    if dobel:
        pyautogui.doubleClick(x, y)
    else:
        pyautogui.click(x, y)
    print(f"  [✓] Click {label} → ({x}, {y})")


def drag(fx, fy, tx, ty, label=""):
    """Drag antartitik tengah elemen pada resolusi layar asli."""
    pyautogui.moveTo(fx, fy, duration=0.3)
    time.sleep(0.15)
    pyautogui.mouseDown(button="left")
    time.sleep(0.15)
    pyautogui.moveRel(8, 8, duration=0.1)   # picu event drag-start
    steps = 12
    for i in range(1, steps + 1):
        pyautogui.moveTo(fx + (tx - fx) * i / steps, fy + (ty - fy) * i / steps, duration=0.04)
    time.sleep(0.2)
    pyautogui.moveTo(tx, ty, duration=0.1)
    time.sleep(0.15)
    pyautogui.mouseUp(button="left")
    print(f"  [✓] Drag {label}: ({fx},{fy}) → ({tx},{ty})")


ABAIKAN = re.compile(r"\b(continue|reminder|speaker|audio|play|popup|close|tutup|ok|stay)\b", re.I)


def hitung_jawaban(action):
    """Apakah aksi ini dihitung sebagai 'menjawab'?"""
    if action.get("answer") is False:
        return False
    if ABAIKAN.search(str(action.get("label", ""))):
        return False
    return True


def jalankan_actions(actions, hal, elems):
    """Eksekusi actions. Return True kalau Next diminta DAN diizinkan."""
    for i, action in enumerate(actions):
        atype = action.get("type", "")
        label = action.get("label", "")
        print(f"  [~] Aksi {i+1}/{len(actions)}: {atype} {label}")
        set_status("Menjalankan aksi", f"{atype}: {label or 'tanpa label'}")

        if atype in ("click", "double_click"):
            elem = cari_elemen(action.get("id"), elems)
            klik(elem["cx"], elem["cy"], label=label, dobel=(atype == "double_click"))
            if hitung_jawaban(action):
                hal.jawab += 1
            hal.riwayat.append(f"{atype} {label}".strip())
            time.sleep(action.get("delay_after", 0.2))
            return False

        elif atype == "drag":
            asal = cari_elemen(action.get("from_id"), elems)
            tujuan = cari_elemen(action.get("to_id"), elems)
            drag(asal["cx"], asal["cy"], tujuan["cx"], tujuan["cy"], label=label)
            if hitung_jawaban(action):
                hal.jawab += 1
            hal.riwayat.append(f"drag {label}".strip())
            time.sleep(action.get("delay_after", 0.8))
            return False

        elif atype == "type":
            elem = cari_elemen(action.get("id"), elems)
            klik(elem["cx"], elem["cy"], label="kolom teks")
            time.sleep(0.3)
            teks = str(action.get("text", ""))
            keyboard.write(teks, delay=0.03)
            print(f"  [✓] Ketik: {teks}")
            hal.jawab += 1
            hal.riwayat.append(f"ketik '{teks}'")
            time.sleep(0.5)
            return False

        elif atype == "key":
            tombol = str(action.get("key", "enter"))
            pyautogui.press(tombol)
            print(f"  [✓] Tekan tombol: {tombol}")
            hal.riwayat.append(f"tekan {tombol}")
            time.sleep(0.5)
            return False

        elif atype == "scroll":
            amount = int(action.get("amount", -5))
            pyautogui.scroll(amount * 100)
            hal.riwayat.append("scroll")
            time.sleep(0.5)
            return False

        elif atype == "wait":
            seconds = float(action.get("seconds", 1))
            set_status("Menunggu", f"Jeda {seconds:g} detik sesuai instruksi.")
            time.sleep(seconds)

        elif atype == "audio":
            if hal.audio_diklik:
                print("  [~] Speaker sudah diklik di halaman ini, skip. Langsung pilih jawaban.")
                continue
            elem = cari_elemen(action.get("id"), elems)
            klik(elem["cx"], elem["cy"], label="Audio/Speaker")
            hal.audio_diklik = True
            hal.riwayat.append("klik speaker (SUDAH, jangan klik lagi; langsung pilih jawaban tebakan)")
            set_status("Memutar audio", f"Menunggu audio {AUDIO_WAIT:g} detik.")
            time.sleep(AUDIO_WAIT)
            return False

        elif atype == "play":
            elem = cari_elemen(action.get("id"), elems)
            klik(elem["cx"], elem["cy"], label="Play video")
            hal.diputar = True
            hal.riwayat.append("play video")
            print(f"  [~] Tunggu video {VIDEO_WAIT} detik...")
            set_status("Menunggu video", f"Video berjalan; jeda {VIDEO_WAIT:g} detik.")
            time.sleep(VIDEO_WAIT)
            return False

        elif atype == "replan":
            time.sleep(0.7)
            return False

        elif atype == "next":
            cari_elemen(action.get("id"), elems, jenis="next")
            if hal.tipe == "VIDEO":
                if not hal.diputar:
                    print(f"  [~] Video belum diputar, tunggu {VIDEO_WAIT} detik dulu...")
                    set_status("Menunggu video", f"Menunggu {VIDEO_WAIT:g} detik sebelum lanjut.")
                    time.sleep(VIDEO_WAIT)
                    hal.diputar = True
                return True
            if hal.jawab < hal.butuh:
                print(f"  [!] Next DITAHAN: baru {hal.jawab} jawaban, butuh minimal {hal.butuh}.")
                set_status("Next ditahan", f"Jawaban {hal.jawab}/{hal.butuh}; pilih jawaban lagi.", "warning")
                hal.riwayat.append(f"NEXT DITOLAK: baru {hal.jawab} jawaban, harus klik jawaban dulu (tebak saja)")
                return False
            return True

        else:
            print(f"  [!] Tipe action tidak dikenal: {atype}, skip...")

    return False


def pindah_halaman(hal, elems, sebelum):
    """Klik elemen Next yang terdeteksi lalu cek apakah halaman benar-benar berpindah."""
    tombol = next((elem for elem in elems if elem["kind"] == "next"), None)
    if tombol is None:
        print("[!] Tombol Next tidak terdeteksi; tidak ada koordinat tebakan yang digunakan.")
        hal.riwayat.append("Next tidak dijalankan: elemen next tidak terdeteksi")
        set_status("Next tidak terdeteksi", "Pastikan tombol Next terlihat di layar.", "warning")
        return False
    klik(tombol["cx"], tombol["cy"], label=f'Next (ID {tombol["id"]})')
    print("  [~] Tunggu halaman berikutnya...")
    set_status("Menunggu halaman", "Memeriksa apakah tombol Next memindahkan halaman.")
    time.sleep(3)
    set_status("Memeriksa perubahan layar", "Membandingkan tampilan sebelum dan sesudah Next.")
    sesudah = hash_screen()

    if sesudah == sebelum:
        hal.butuh = min(hal.butuh + 1, 6)
        hal.riwayat.append(
            "NEXT ditekan tapi halaman tidak berpindah (masih ada yang harus dijawab / semua kartu harus dibalik)"
        )
        hal.last_hash = None
        hal.sama = 0
        print(f"  [!] Halaman tidak berpindah. Minimal jawaban dinaikkan jadi {hal.butuh}.")
        return False

    hal.reset()
    return True

# =============================================
# LOOP UTAMA BOT
# =============================================
def proses_bot():
    global is_running
    print(
        f"\n[+] BOT MULAI | Layar: {screen_w}x{screen_h} | "
        f"Gemini Vision: {GEMINI_MODEL}"
    )
    print(f"[i] MIN_JAWAB={MIN_JAWAB} VIDEO_WAIT={VIDEO_WAIT}s | F8 pause | F9 keluar | mouse ke pojok kiri-atas = stop paksa\n")
    set_status("Bot berjalan", f"Gemini menjawab dan memilih ID: {GEMINI_MODEL}.")

    hal = Halaman()

    while is_running:
        try:
            print("[~] Screenshot layar...")
            img_b64, layar_hash, elems = capture_screen(simpan=True)

            hal.sama = hal.sama + 1 if layar_hash == hal.last_hash else 0
            hal.last_hash = layar_hash
            if hal.riwayat and hal.sama >= STUCK_LIMIT:
                print("[!] Layar tidak berubah, paksa Next...")
                set_status("Layar tidak berubah", "Mencoba Next setelah beberapa langkah tanpa perubahan.", "warning")
                pindah_halaman(hal, elems, layar_hash)
                hal.sama = 0
                continue

            hal.langkah += 1
            if hal.langkah > MAX_STEPS_PER_PAGE:
                print("[!] Terlalu banyak langkah di halaman ini, paksa Next...")
                set_status("Batas langkah tercapai", "Mencoba Next agar bot tidak macet.", "warning")
                pindah_halaman(hal, elems, layar_hash)
                hal.langkah = 0
                continue

            print(f"[~] Gemini menganalisis gambar soal (langkah {hal.langkah})...")
            data = jawab_soal_google(img_b64, hal, elems)

            tipe = str(data.get("tipe") or "UNKNOWN").upper()

            if not data.get("ada_soal") and tipe not in ("VIDEO", "POPUP"):
                hal.langkah -= 1
                hal.tanpa_soal += 1
                if hal.tanpa_soal >= NO_SOAL_LIMIT:
                    print("[!] Lama tidak ada soal terdeteksi, coba Next...")
                    pindah_halaman(hal, elems, layar_hash)
                    hal.tanpa_soal = 0
                else:
                    print("[!] Tidak ada soal terdeteksi. Tunggu 2 detik...")
                    set_status("Menunggu soal", f"Belum ada soal; percobaan {hal.tanpa_soal}/{NO_SOAL_LIMIT}.")
                    time.sleep(2)
                continue
            hal.tanpa_soal = 0

            if tipe != "UNKNOWN":
                hal.tipe = tipe

            actions = data.get("actions", []) or []
            print(f"\n[✓] Tipe: {hal.tipe} | {data.get('keterangan', '')} | {len(actions)} aksi | "
                  f"jawab {hal.jawab}/{hal.butuh}\n")

            if actions:
                minta_next = jalankan_actions(actions, hal, elems)
            else:
                # AI tidak memberi aksi: Next hanya kalau syarat terpenuhi
                minta_next = (hal.tipe == "VIDEO") or (hal.jawab >= hal.butuh)
                if not minta_next:
                    hal.riwayat.append("tidak ada aksi; klik jawaban (tebak) dulu")
                    time.sleep(1)

            if minta_next:
                pindah_halaman(hal, elems, layar_hash)

        except pyautogui.FailSafeException:
            print("\n[-] FAILSAFE: mouse di pojok layar. Bot dihentikan.")
            is_running = False
            set_status("Dihentikan oleh failsafe", "Mouse mencapai sudut kiri atas.", "error")

        except json.JSONDecodeError as e:
            print(f"[!] JSON error dari AI: {e}")
            set_status("Error respons AI", f"Format JSON tidak valid: {e}", "error")
            time.sleep(3)

        except ValueError as e:
            print(f"[!] Aksi AI ditolak: {e}")
            hal.riwayat.append(f"AKSI DITOLAK: {e}; gunakan ID dari screenshot terbaru")
            set_status("Aksi ditolak", str(e), "warning")
            time.sleep(1)

        except Exception as e:
            msg = str(e)
            print(f"[!] Error: {msg}")
            jeda = 10 if ("429" in msg or "rate" in msg.lower()) else 3
            set_status("Error — bot akan mencoba lagi", f"{msg}\nCoba lagi dalam {jeda} detik.", "error")
            time.sleep(jeda)

    print("[-] Loop bot berhenti.")


def toggle_bot():
    global is_running
    if not is_running:
        is_running = True
        set_status("Bot berjalan", "Memulai proses; screenshot berikutnya segera diambil.")
        threading.Thread(target=proses_bot, daemon=True).start()
    else:
        is_running = False
        print("\n[-] BOT PAUSE.")
        set_status("Dijeda", "Tekan F8 atau tombol Lanjutkan untuk meneruskan.", "ready")


def keluar():
    global is_running
    is_running = False
    status_events.put(("exit",))


def jalankan_panel():
    root = tk.Tk()
    root.title("Cyells Bot - Status")
    root.geometry("500x235")
    root.resizable(False, False)
    root.configure(padx=18, pady=14)

    ttk.Label(root, text="CYELLS BOT", font=("Segoe UI", 15, "bold")).pack(anchor="w")
    status_label = ttk.Label(root, font=("Segoe UI", 12, "bold"))
    status_label.pack(anchor="w", pady=(10, 2))
    detail_label = ttk.Label(root, wraplength=460, justify="left")
    detail_label.pack(anchor="w", fill="x", pady=(0, 4))
    elapsed_label = ttk.Label(root, text="Siap", foreground="#555555")
    elapsed_label.pack(anchor="w")

    progress = ttk.Progressbar(root, mode="indeterminate", length=460)
    progress.pack(fill="x", pady=(10, 10))

    tombol = ttk.Frame(root)
    tombol.pack(fill="x")
    tombol_jalan = ttk.Button(tombol, text="Mulai", command=toggle_bot)
    tombol_jalan.pack(side="left")
    ttk.Button(tombol, text="Keluar (F9)", command=keluar).pack(side="right")
    progress_running = [False]

    warna = {
        "working": "#1769aa",
        "ready": "#287a3d",
        "warning": "#a05a00",
        "error": "#b42318",
    }
    tahap_memuat = {
        "Mengambil screenshot",
        "Mendeteksi elemen",
        "Menganalisis soal dengan Gemini",
        "Menunggu analisis AI",
        "Mencoba format AI alternatif",
        "Menjalankan aksi",
        "Memutar audio",
        "Menunggu video",
        "Menunggu halaman",
        "Memeriksa perubahan layar",
        "Menunggu",
    }

    def perbarui_panel():
        while True:
            try:
                event = status_events.get_nowait()
            except queue.Empty:
                break
            if event[0] == "exit":
                root.destroy()
                return
            _, stage, detail, level, started_at = event
            status_state.update(
                stage=stage,
                detail=detail,
                level=level,
                started_at=started_at,
            )

        stage = status_state["stage"]
        detail = status_state["detail"]
        level = status_state["level"]
        started_at = status_state["started_at"]
        status_label.configure(text=f"[{level.upper()}] {stage}", foreground=warna.get(level, "#555555"))
        detail_label.configure(text=detail)
        tombol_jalan.configure(text="Jeda (F8)" if is_running else "Mulai (F8)")

        if started_at is not None:
            durasi = time.monotonic() - started_at
            teks_durasi = f"Berjalan {durasi:.0f} detik"
            if stage == "Menunggu analisis AI" and durasi >= 30:
                teks_durasi += " — masih menunggu model/API; periksa koneksi jika terlalu lama"
            elapsed_label.configure(text=teks_durasi)
        else:
            elapsed_label.configure(text="Bot aktif" if is_running else "Siap")

        if level == "working" and stage in tahap_memuat:
            if str(progress["mode"]) != "indeterminate":
                progress.configure(mode="indeterminate")
            if not progress_running[0]:
                progress.start(12)
                progress_running[0] = True
        else:
            if progress_running[0]:
                progress.stop()
                progress_running[0] = False
            progress.configure(mode="determinate", maximum=1, value=1 if level == "ready" else 0)

        root.after(150, perbarui_panel)

    def tutup_panel():
        global is_running
        is_running = False
        keyboard.unhook_all()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", tutup_panel)
    keyboard.add_hotkey("f8", toggle_bot)
    keyboard.add_hotkey("f9", keluar)
    perbarui_panel()
    root.mainloop()
    keyboard.unhook_all()


print("=" * 52)
print("  CYELLS BOT - Multi-Type Quiz (ed.engdis.com)")
print("=" * 52)
print("  F8        -> MULAI / PAUSE")
print("  F9        -> KELUAR")
print("=" * 52)
print()

jalankan_panel()