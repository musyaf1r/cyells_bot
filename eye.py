import mss
from PIL import Image
import pyautogui
from google import genai

client = genai.Client(api_key="GOOGLE_API_KEY")

# 1. Tangkap area layar (gambar cuma ada di RAM, bukan di harddisk)
monitor = {"top": 100, "left": 100, "width": 1200, "height": 800}

with mss.mss() as sct:
    sct_img = sct.grab(monitor)
    # Variabel 'img' ini cuma wujud di memori sementara
    img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

# 2. Langsung kirim objek RAM tadi ke AI
response = client.models.generate_content(
    model='gemini-2.5-flash',
    contents=[
        img, 
        "Analisis gambar soal ini. Jawab HANYA dengan nomor atau huruf pilihan yang benar."
    ]
)

jawaban_ai = response.text.strip()
print("Jawaban AI:", jawaban_ai)

# 3. Eksekusi klik otomatis
coords = {
    "1": (300, 610),
    "2": (300, 660),
    "3": (300, 710)
}

for opsi, posisi in coords.items():
    if opsi in jawaban_ai:
        pyautogui.click(posisi[0], posisi[1])
        print(f"Berhasil klik opsi {opsi}!")
        break