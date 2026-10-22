import unittest

import cv2
import numpy as np
from PIL import Image

from deteksi import deteksi, gambar_overlay, ringkas


def _halaman():
    img = np.full((600, 800, 3), 255, dtype=np.uint8)
    cv2.line(img, (0, 70), (799, 70), (255, 200, 50), 4)
    img[550:, :] = (235, 235, 235)
    cv2.rectangle(img, (700, 565), (780, 595), (0, 190, 0), -1)
    return img


def _word_bank():
    img = _halaman()
    cv2.putText(img, "Fill the blank with a word", (55, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.line(img, (260, 220), (390, 220), (30, 30, 30), 2)
    for x, word in ((80, "quiet"), (250, "bright")):
        cv2.rectangle(img, (x, 300), (x + 130, 345), (50, 50, 50), 2)
        cv2.putText(img, word, (x + 18, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
    return img


def _picture_choices():
    img = _halaman()
    cv2.putText(img, "Choose the matching picture", (55, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    for x, seed in ((80, 0), (330, 1)):
        cv2.rectangle(img, (x, 185), (x + 190, 370), (20, 20, 20), 3)
        patch = img[195:355, x + 10:x + 180]
        yy, xx = np.indices(patch.shape[:2])
        patch[:, :, 0] = (xx * 3 + seed * 80) % 256
        patch[:, :, 1] = (yy * 2 + seed * 50) % 256
        patch[:, :, 2] = ((xx + yy) * 2 + seed * 20) % 256
    return img


def _open_dropdown():
    img = _halaman()
    cv2.putText(img, "Select the right answer", (55, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.rectangle(img, (75, 175), (380, 215), (40, 40, 40), 2)
    cv2.putText(img, "Choose one", (90, 203), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    cv2.rectangle(img, (75, 216), (380, 252), (225, 240, 250), -1)
    for y, text in ((241, "first option"), (278, "second option"), (315, "third option")):
        cv2.putText(img, text, (95, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
    return img


class DetectorSyntheticImageTests(unittest.TestCase):
    def _kinds(self, img):
        result = deteksi(img)
        elems = result["elems"]
        self.assertEqual([e["id"] for e in elems], list(range(1, len(elems) + 1)))
        self.assertTrue(all(0 <= e["cx"] < img.shape[1] and 0 <= e["cy"] < img.shape[0] for e in elems))
        self.assertIn("next", [e["kind"] for e in elems])
        return {e["kind"] for e in elems}

    def test_word_bank_detects_chips_blank_and_text(self):
        kinds = self._kinds(_word_bank())
        self.assertIn("box", kinds)
        self.assertIn("blank", kinds)
        self.assertIn("teks", kinds)

    def test_picture_choices_are_detected_as_images(self):
        kinds = self._kinds(_picture_choices())
        self.assertIn("img", kinds)
        self.assertIn("teks", kinds)

    def test_open_dropdown_detects_control_and_visible_options(self):
        kinds = self._kinds(_open_dropdown())
        self.assertIn("box", kinds)
        self.assertIn("teks", kinds)

    def test_overlay_numbers_elements_and_prompt_summary_omits_coordinates(self):
        img = _word_bank()
        elems = deteksi(img)["elems"]
        original = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        overlay = gambar_overlay(original, elems, 1)
        summary = ringkas(elems)

        self.assertEqual(overlay.size, original.size)
        self.assertNotEqual(overlay.tobytes(), original.tobytes())
        self.assertIn(":blank ", summary)
        self.assertNotIn("@", summary)


if __name__ == "__main__":
    unittest.main()
