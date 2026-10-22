"""
deteksi.py - Deteksi elemen layar quiz lewat pengolahan gambar (OpenCV).

Tujuan: AI tidak perlu menebak koordinat piksel. Bot mendeteksi sendiri elemen yang bisa
diklik/di-drag, memberi NOMOR di gambar, dan AI cukup menyebut nomornya. Titik klik dihitung
dari piksel elemen itu sendiri, jadi akurat sampai ke piksel.

Jenis elemen:
  box   : kotak dengan garis tepi (chip word bank, dropdown, kartu, tombol, speaker, opsi)
  img   : gambar/foto (kartu gambar)
  blank : kolom kosong bergaris bawah (target drop word bank)
  teks  : baris teks (opsi radio, item daftar dropdown, dll)
"""
import cv2
import numpy as np
from PIL import ImageDraw, ImageFont

WARNA = {
    "box": (230, 30, 30),
    "img": (200, 0, 200),
    "blank": (255, 140, 0),
    "teks": (30, 90, 230),
}


# ---------------------------------------------------------------------------
# Layout halaman: cari batas konten & tombol Next
# ---------------------------------------------------------------------------
def _run_pertama(idx):
    """Dari array indeks terurut, ambil run berurutan pertama -> (awal, akhir)."""
    awal = akhir = int(idx[0])
    for v in idx[1:]:
        if int(v) == akhir + 1:
            akhir = int(v)
        else:
            break
    return awal, akhir


def cari_layout(img):
    """Return (content_top, footer_top). Dengan fallback kalau pola tidak ketemu."""
    h, w = img.shape[:2]
    x = int(w * 0.5)

    # --- batas atas: garis biru (cyan) di bawah navbar, lalu bar instruksi abu-abu muda ---
    col = img[:, x].astype(int)
    b, g, r = col[:, 0], col[:, 1], col[:, 2]
    cyan = (r < 130) & (g > 150) & (b > 190) & (b - r > 80)
    ys = np.where(cyan[: int(h * 0.4)])[0]
    if len(ys):
        _, stripe_bottom = _run_pertama(ys)
        content_top = stripe_bottom + 50
        # cari baris putih murni pertama setelah bar instruksi
        xs = [int(w * 0.7), int(w * 0.8), int(w * 0.9)]
        for y in range(stripe_bottom + 10, min(h - 3, stripe_bottom + 160)):
            if all(img[y, xx].min() >= 252 and img[y + 2, xx].min() >= 252 for xx in xs):
                content_top = y
                break
    else:
        content_top = int(h * 0.18)

    # --- batas bawah: footer abu-abu muda di dasar layar ---
    ref = img[h - 3, x].astype(int)
    y = h - 3
    while y > h * 0.8 and np.abs(img[y, x].astype(int) - ref).max() <= 10:
        y -= 1
    footer_top = y + 1
    if not (15 <= h - footer_top <= h * 0.12):
        footer_top = int(h * 0.955)

    return int(content_top), int(footer_top)


def cari_next(img, footer_top):
    """Tombol Next = blok hijau terbesar di kanan-bawah. Return box atau None."""
    _, w = img.shape[:2]
    y0 = max(0, footer_top - 4)
    x0 = int(w * 0.7)
    roi = img[y0:, x0:].astype(int)
    b, g, r = roi[..., 0], roi[..., 1], roi[..., 2]
    mask = ((g > 140) & (r < 100) & (b < 140) & (g - r > 60)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    terbaik = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw >= 60 and bh >= 25 and area >= 1500:
            if terbaik is None or area > stats[terbaik][4]:
                terbaik = i
    if terbaik is None:
        return None
    x, y, bw, bh, _ = stats[terbaik]
    return int(x0 + x), int(y0 + y), int(x0 + x + bw), int(y0 + y + bh)


# ---------------------------------------------------------------------------
# Utilitas kotak
# ---------------------------------------------------------------------------
def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua


def _di_dalam(pusat, kotak, pad=0):
    cx, cy = pusat
    x1, y1, x2, y2 = kotak
    return x1 - pad <= cx <= x2 + pad and y1 - pad <= cy <= y2 + pad


# ---------------------------------------------------------------------------
# Deteksi utama
# ---------------------------------------------------------------------------
def deteksi(img, maks=60):
    """
    img: array BGR (ukuran asli layar).
    Return dict:
      elems : list {id, kind, box=(x1,y1,x2,y2), cx, cy}  (koordinat piksel layar asli)
      next  : (cx, cy) tombol Next, atau None
      area  : (content_top, footer_top)
    """
    ct, fb = cari_layout(img)
    next_box = cari_next(img, fb)

    roi = img[ct:fb]
    rh, rw = roi.shape[:2]
    if rh < 40:
        elems = []
        if next_box:
            x1, y1, x2, y2 = next_box
            elems.append({
                "kind": "next",
                "box": next_box,
                "cx": int((x1 + x2) / 2),
                "cy": int((y1 + y2) / 2),
                "id": 1,
            })
        return {
            "elems": elems,
            "next": ((elems[0]["cx"], elems[0]["cy"]) if elems else None),
            "area": (ct, fb),
        }
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # ---------- 1. kotak bergaris tepi ----------
    edges = cv2.Canny(gray, 25, 90)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    kandidat = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w < 22 or h < 18:
            continue
        if w > rw * 0.8 or h > rh * 0.8:
            continue
        if cv2.contourArea(c) / (w * h) < 0.82:
            continue
        kandidat.append((x, y, x + w, y + h))

    kandidat.sort(key=lambda b: -((b[2] - b[0]) * (b[3] - b[1])))
    boxes = []
    for b in kandidat:
        if any(_iou(b, k) > 0.7 for k in boxes):
            continue
        boxes.append(b)

    # klasifikasi gambar (varians piksel tinggi = foto)
    elems = []
    img_boxes = []
    for b in boxes:
        x1, y1, x2, y2 = b
        patch = gray[y1:y2, x1:x2]
        rasio = (x2 - x1) / max(1, y2 - y1)
        if (
            patch.size
            and (x2 - x1) >= 70
            and (y2 - y1) >= 70
            and 0.5 <= rasio <= 2.5
            and float(patch.std()) > 38
        ):
            img_boxes.append(b)
            elems.append(("img", b))
    boxes = [b for b in boxes if not any(_di_dalam(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), ib) for ib in img_boxes)
             or b in img_boxes]

    # ---------- 2. baris teks ----------
    gelap = (gray < 150).astype(np.uint8) * 255
    gelap = cv2.dilate(gelap, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
    tc, _ = cv2.findContours(gelap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    teks = []
    for c in tc:
        x, y, w, h = cv2.boundingRect(c)
        if h < 8 or h > 45 or w < 12 or w > 900:
            continue
        t = (x + 7, y + 1, x + w - 7, y + h - 1)
        if t[2] - t[0] < 6:
            t = (x, y, x + w, y + h)
        pusat = ((t[0] + t[2]) / 2, (t[1] + t[3]) / 2)
        if any(_di_dalam(pusat, ib) for ib in img_boxes):
            continue
        teks.append(t)

    # kotak non-gambar: simpan semuanya; teks di dalam kotak yang hanya berisi <=1 teks dibuang
    non_img = [b for b in boxes if b not in img_boxes]
    for b in non_img:
        elems.append(("box", b))

    teks_final = []
    for t in teks:
        pusat = ((t[0] + t[2]) / 2, (t[1] + t[3]) / 2)
        induk = [b for b in non_img if _di_dalam(pusat, b)]
        buang = False
        for b in induk:
            jumlah = sum(1 for t2 in teks if _di_dalam(((t2[0] + t2[2]) / 2, (t2[1] + t2[3]) / 2), b))
            if jumlah <= 1:
                buang = True
                break
        if not buang:
            teks_final.append(t)
    for t in teks_final:
        elems.append(("teks", t))

    # ---------- 3. kolom kosong (garis bawah) ----------
    bw_ = (gray < 225).astype(np.uint8) * 255
    horiz = cv2.morphologyEx(bw_, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1)))
    lc, _ = cv2.findContours(horiz, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in lc:
        x, y, w, h = cv2.boundingRect(c)
        if not (45 <= w <= 420 and h <= 4):
            continue
        # garis ini bagian dari tepi kotak yang sudah terdeteksi?
        tepi = False
        for (bx1, by1, bx2, by2) in boxes:
            if x >= bx1 - 5 and x + w <= bx2 + 5 and (abs(y - by1) <= 5 or abs(y + h - by2) <= 5):
                tepi = True
                break
        if tepi:
            continue
        if any(_di_dalam((x + w / 2, y), ib) for ib in img_boxes):
            continue
        elems.append(("blank", (x, max(0, y - 26), x + w, y + 4)))

    # ---------- 4. rapikan: geser ke koordinat layar asli, urut baca, beri id ----------
    hasil = []
    for kind, (x1, y1, x2, y2) in elems:
        y1 += ct
        y2 += ct
        hasil.append({"kind": kind, "box": (int(x1), int(y1), int(x2), int(y2))})

    hasil.sort(key=lambda e: (round(((e["box"][1] + e["box"][3]) / 2) / 20), e["box"][0]))
    hasil = hasil[:max(0, maks - 1 if next_box else maks)]
    if next_box:
        x1, y1, x2, y2 = next_box
        hasil.append({
            "kind": "next",
            "box": next_box,
            "cx": int((x1 + x2) / 2),
            "cy": int((y1 + y2) / 2),
        })
    for i, e in enumerate(hasil, 1):
        x1, y1, x2, y2 = e["box"]
        e["id"] = i
        e.setdefault("cx", int((x1 + x2) / 2))
        e.setdefault("cy", int((y1 + y2) / 2))

    next_elem = next((e for e in hasil if e["kind"] == "next"), None)
    nxt = (next_elem["cx"], next_elem["cy"]) if next_elem else None
    return {"elems": hasil, "next": nxt, "area": (ct, fb)}


# ---------------------------------------------------------------------------
# Overlay bernomor untuk AI
# ---------------------------------------------------------------------------
def _font(ukuran=15):
    for nama in ("arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(nama, ukuran)
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=ukuran)
    except Exception:
        return ImageFont.load_default()


def gambar_overlay(pil_img, elems, skala):
    """Gambar kotak + nomor pada pil_img (yang sudah di-resize dengan faktor `skala`)."""
    img = pil_img.convert("RGB")
    d = ImageDraw.Draw(img)
    font = _font(15)
    for e in elems:
        x1, y1, x2, y2 = [v * skala for v in e["box"]]
        warna = WARNA.get(e["kind"], (230, 30, 30))
        d.rectangle([x1, y1, x2, y2], outline=warna, width=2)
        label = str(e["id"])
        try:
            tw = d.textlength(label, font=font)
        except Exception:
            tw = 9 * len(label)
        th = 17
        lx, ly = x1, max(0, y1 - th)
        d.rectangle([lx, ly, lx + tw + 6, ly + th], fill=warna)
        d.text((lx + 3, ly), label, fill=(255, 255, 255), font=font)
    return img


def ringkas(elems):
    """Daftar elemen untuk prompt tanpa koordinat; AI hanya memilih ID."""
    baris = []
    for e in elems:
        x1, y1, x2, y2 = e["box"]
        baris.append(f'{e["id"]}:{e["kind"]} {x2 - x1}x{y2 - y1}')
    return "; ".join(baris)