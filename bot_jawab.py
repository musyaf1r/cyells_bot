import json
import os
import re
import base64
import io
import mss
from PIL import Image
import pyautogui
from groq import Groq
from dotenv import load_dotenv
import keyboard
import time
import threading

# =============================================
# GROQ API KEY - dibaca dari file .env
# =============================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise SystemExit("[!] GROQ_API_KEY belum diisi. Buat file .env (lihat .env.example).")
client = Groq(api_key=GROQ_API_KEY)

# Ukuran layar otomatis
screen_w, screen_h = pyautogui.size()

# Posisi tombol Next (bottom-right, selalu sama)
NEXT_X = 0.972
NEXT_Y = 0.972

is_running = False

# =============================================
# FUNGSI SCREENSHOT
# =============================================
def capture_screen():
    """Tangkap full layar sebagai base64 JPEG."""
    with mss.MSS() as sct:
        monitor = {"top": 0, "left": 0, "width": screen_w, "height": screen_h}
        sct_img = sct.grab(monitor)
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

    if img.width > 1280:
        ratio = 1280 / img.width
        img = img.resize((1280, int(img.height * ratio)), Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

# =============================================
# FUNGSI TANYA AI
# =============================================
def tanya_ai(img_base64):
    prompt = (
        "Kamu adalah bot otomatis untuk mengerjakan soal quiz bahasa Inggris di browser.\n"
        "Analisis screenshot ini dan tentukan:\n"
        "1. Apakah ada soal aktif?\n"
        "2. Tipe soalnya apa?\n"
        "3. Urutan aksi apa yang harus dilakukan untuk menjawab dengan benar?\n\n"

        "=== TIPE SOAL YANG MUNGKIN ADA ===\n"
        "RADIO      : Pilihan ganda dengan radio button (klik radio button yang benar)\n"
        "PICTURE    : Pilih gambar yang benar dari beberapa gambar\n"
        "MATCH      : Drag label/kata dari sisi kanan ke kotak di bawah gambar\n"
        "DROPDOWN   : Klik dropdown dan pilih kata yang benar\n"
        "WORDBANK   : Klik kata dari word bank di bawah, lalu klik bagian blank di kalimat\n"
        "AUDIO      : Ada tombol speaker, klik speaker lalu pilih gambar yang sesuai\n"
        "VIDEO      : Halaman berisi video player (bukan soal). Langsung skip ke Next.\n"
        "             Kalau ada popup 'Reminder' / 'Continue' / 'Stay in this step' → klik 'Continue' dulu\n\n"

        "=== FORMAT JSON WAJIB (HANYA JSON, TIDAK ADA TEKS LAIN) ===\n"
        "{\n"
        '  "ada_soal": true,\n'
        '  "tipe": "RADIO",\n'
        '  "keterangan": "penjelasan singkat apa yang dilakukan",\n'
        '  "actions": [\n'
        '    {"type": "click", "x": 28, "y": 66, "label": "opsi 1"},\n'
        '    {"type": "next"}\n'
        "  ]\n"
        "}\n\n"

        "=== JENIS ACTION ===\n"
        '{"type": "click", "x": X, "y": Y, "label": "keterangan"}\n'
        "  → Klik koordinat. x dan y adalah PERSENTASE layar (0-100)\n\n"
        '{"type": "drag", "from_x": X, "from_y": Y, "to_x": X2, "to_y": Y2, "label": "keterangan"}\n'
        "  → Drag dari satu titik ke titik lain (untuk soal MATCH)\n\n"
        '{"type": "wait", "seconds": 1}\n'
        "  → Tunggu beberapa detik\n\n"
        '{"type": "audio"}\n'
        "  → Klik tombol speaker/audio (otomatis dicari di sekitar atas layar)\n\n"
        '{"type": "next"}\n'
        "  → Klik tombol Next/lanjut (selalu di sudut kanan bawah)\n\n"

        "=== ATURAN PENTING ===\n"
        "- x dan y adalah PERSENTASE (0-100) dari lebar dan tinggi layar\n"
        "- Selalu akhiri dengan {\"type\": \"next\"}\n"
        "- Untuk MATCH: drag label dari kanan ke kotak kosong di bawah gambar\n"
        "- Untuk DROPDOWN: klik dropdown, tunggu, lalu klik opsi yang benar\n"
        "- Untuk WORDBANK: klik setiap kata di word bank, lalu klik blank-nya di kalimat\n"
        "- Untuk AUDIO: gunakan action audio, tunggu 3 detik, lalu klik gambar yang benar\n"
        "- Kalau tidak yakin tipe soalnya, tetap coba jawab semampu mungkin\n"
        "- Kalau tidak ada soal: {\"ada_soal\": false, \"tipe\": null, \"keterangan\": \"tidak ada soal\", \"actions\": []}\n"
        "- HANYA kembalikan JSON, tidak ada teks lain apapun!"
    )

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        temperature=0.1,
        max_tokens=600,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content.strip()
    print(f"[i] Respon AI: {raw}")
    return json.loads(raw)

# =============================================
# FUNGSI AKSI
# =============================================
def klik(x_ratio, y_ratio, label=""):
    x = int(x_ratio * screen_w)
    y = int(y_ratio * screen_h)
    pyautogui.moveTo(x, y, duration=0.25)
    pyautogui.click(x, y)
    print(f"  [✓] Click {label} → ({x}, {y})")

def drag(from_x_r, from_y_r, to_x_r, to_y_r, label=""):
    fx = int(from_x_r * screen_w)
    fy = int(from_y_r * screen_h)
    tx = int(to_x_r  * screen_w)
    ty = int(to_y_r  * screen_h)
    pyautogui.moveTo(fx, fy, duration=0.3)
    time.sleep(0.2)
    pyautogui.dragTo(tx, ty, duration=0.6, button='left')
    print(f"  [✓] Drag {label}: ({fx},{fy}) → ({tx},{ty})")

def klik_next():
    klik(NEXT_X, NEXT_Y, label="Next >")

def klik_audio():
    # Speaker biasanya ada di sekitar tengah-atas konten
    klik(0.27, 0.19, label="Audio/Speaker button")

def jalankan_actions(actions):
    """Eksekusi daftar actions dari AI."""
    for i, action in enumerate(actions):
        atype = action.get("type", "")
        label = action.get("label", f"action {i+1}")
        print(f"  [~] Aksi {i+1}/{len(actions)}: {atype} — {label}")

        if atype == "click":
            x = action.get("x", 50)
            y = action.get("y", 50)
            klik(x / 100, y / 100, label=label)
            time.sleep(action.get("delay_after", 0.6))

        elif atype == "drag":
            fx = action.get("from_x", 50)
            fy = action.get("from_y", 50)
            tx = action.get("to_x", 50)
            ty = action.get("to_y", 50)
            drag(fx/100, fy/100, tx/100, ty/100, label=label)
            time.sleep(action.get("delay_after", 0.8))

        elif atype == "wait":
            secs = action.get("seconds", 1)
            print(f"  [~] Tunggu {secs} detik...")
            time.sleep(secs)

        elif atype == "audio":
            klik_audio()
            print("  [~] Tunggu audio selesai (3 detik)...")
            time.sleep(3)

        elif atype == "next":
            klik_next()
            print("  [~] Tunggu halaman berikutnya (2.5 detik)...")
            time.sleep(2.5)

        else:
            print(f"  [!] Tipe action tidak dikenal: {atype}, skip...")

# =============================================
# LOOP UTAMA BOT
# =============================================
def proses_bot():
    global is_running
    print(f"\n[+] BOT MULAI | Layar: {screen_w}x{screen_h}")
    print("[i] F8 = pause/stop\n")

    while is_running:
        try:
            # 1. Screenshot
            print("[~] Screenshot layar...")
            img_b64 = capture_screen()

            # 2. Tanya AI
            print("[~] Analisis AI (qwen vision)...")
            data = tanya_ai(img_b64)

            tipe      = data.get("tipe", "UNKNOWN")
            keterangan = data.get("keterangan", "")
            actions   = data.get("actions", [])

            # 3. Tidak ada soal (tapi VIDEO tetap diproses)
            if not data.get("ada_soal") and tipe != "VIDEO":
                print("[!] Tidak ada soal terdeteksi. Tunggu 2 detik...")
                time.sleep(2)
                continue

            print(f"\n[✓] Halaman terdeteksi!")
            print(f"    Tipe      : {tipe}")
            print(f"    Keterangan: {keterangan}")
            print(f"    Actions   : {len(actions)} langkah\n")

            # 4. Jalankan semua actions
            if actions:
                jalankan_actions(actions)
            else:
                # Fallback kalau AI tidak kasih actions
                print("[!] AI tidak kasih actions, langsung klik Next...")
                klik_next()
                time.sleep(2.5)

        except json.JSONDecodeError as e:
            print(f"[!] JSON error dari AI: {e}")
            time.sleep(3)

        except Exception as e:
            print(f"[!] Error: {e}")
            time.sleep(3)

def toggle_bot():
    global is_running
    if not is_running:
        is_running = True
        threading.Thread(target=proses_bot, daemon=True).start()
    else:
        is_running = False
        print("\n[-] BOT PAUSE.")

print("=" * 52)
print("  CYELLS BOT - Multi-Type Quiz (ed.engdis.com)")
print("=" * 52)
print("  F8        -> MULAI / PAUSE")
print("  Ctrl+C    -> Keluar")
print("=" * 52)
print()

keyboard.add_hotkey('f8', toggle_bot)
keyboard.wait()