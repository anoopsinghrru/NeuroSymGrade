"""
OCR Perception Front-End (Unit 5).
Converts a scanned/handwritten submission image into raw text lines using a
lightweight deep-learning OCR model (EasyOCR: CRAFT detector + CRNN recognizer),
so the existing TextPreprocessor / NeuralSlotExtractor pipeline can consume
genuinely unstructured, non-digital student input instead of only typed strings.
"""

import re
from typing import List, Optional, Tuple
from PIL import Image, ImageOps


class OCRSubmissionReader:
    """
    Lazily loads an EasyOCR reader and reconstructs reading-ordered text lines
    from a scanned/handwritten derivation image.
    """

    _reader = None  # Shared across instances; model weights are ~60MB and slow to load.

    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False):
        self.languages = languages or ["en"]
        self.gpu = gpu

    def _get_reader(self):
        if OCRSubmissionReader._reader is None:
            import easyocr
            OCRSubmissionReader._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
        return OCRSubmissionReader._reader

    @staticmethod
    def _preprocess(image_path: str) -> Image.Image:
        """Grayscale + autocontrast to improve recognition on photographed/scanned pages."""
        img = Image.open(image_path).convert("L")
        img = ImageOps.autocontrast(img, cutoff=1)
        return img

    @staticmethod
    def _group_into_lines(
        detections: List[Tuple[list, str, float]]
    ) -> List[Tuple[float, str, float]]:
        """
        Merges word/phrase-level EasyOCR detections into reading-ordered lines.
        Each detection is (bbox, text, confidence); bbox is 4 (x, y) corners.
        Returns a list of (line_center_y, line_text, mean_confidence), sorted top-to-bottom.
        """
        items = []
        for bbox, text, conf in detections:
            ys = [p[1] for p in bbox]
            xs = [p[0] for p in bbox]
            center_y = sum(ys) / len(ys)
            height = max(ys) - min(ys)
            items.append({"x": min(xs), "y": center_y, "h": height, "text": text, "conf": conf})

        items.sort(key=lambda d: d["y"])

        lines: List[List[dict]] = []
        for item in items:
            placed = False
            for line in lines:
                ref = line[0]
                tolerance = max(ref["h"], item["h"]) * 0.6
                if abs(item["y"] - ref["y"]) <= tolerance:
                    line.append(item)
                    placed = True
                    break
            if not placed:
                lines.append([item])

        grouped = []
        for line in lines:
            line.sort(key=lambda d: d["x"])
            line_text = " ".join(d["text"] for d in line)
            mean_conf = sum(d["conf"] for d in line) / len(line)
            mean_y = sum(d["y"] for d in line) / len(line)
            grouped.append((mean_y, line_text, mean_conf))

        grouped.sort(key=lambda t: t[0])
        return grouped

    @staticmethod
    def _normalize_math_ocr(text: str) -> str:
        """
        Corrects a small set of recurring handwriting-OCR confusions in math
        notation that the detector/recognizer otherwise gets systematically
        wrong (not general spelling correction, which would mask genuine
        student errors the symbolic layer needs to see):
        - Cursive "s" is frequently misread as "$" (dollar sign never
          legitimately appears in a physics derivation).
        - A handwritten "×" multiplication sign is often misread as a bare
          "x"/"X" letter when it sits between two numeric operands.
        """
        normalized = text.replace("$", "s")
        normalized = re.sub(r'(?<=[\d\)])\s+[xX]\s+(?=[\d\(])', ' * ', normalized)
        return normalized

    def extract_text(self, image_path: str) -> str:
        """
        Runs OCR on the image and returns reconstructed multi-line raw text,
        suitable as `raw_submission_text` for NeuroSymbolicVerificationEngine.verify_submission.
        """
        reader = self._get_reader()
        preprocessed = self._preprocess(image_path)
        import numpy as np
        detections = reader.readtext(np.array(preprocessed), detail=1)
        lines = self._group_into_lines(detections)
        raw_text = "\n".join(text for _, text, _ in lines)
        return self._normalize_math_ocr(raw_text)

    def extract_text_with_confidence(self, image_path: str) -> Tuple[str, float]:
        """
        Same as extract_text, but also returns the mean per-line OCR confidence,
        which can be fed upstream as a genuine perception-uncertainty signal.
        """
        reader = self._get_reader()
        preprocessed = self._preprocess(image_path)
        import numpy as np
        detections = reader.readtext(np.array(preprocessed), detail=1)
        lines = self._group_into_lines(detections)
        if not lines:
            return "", 0.0
        text = self._normalize_math_ocr("\n".join(t for _, t, _ in lines))
        mean_conf = sum(c for _, _, c in lines) / len(lines)
        return text, round(mean_conf, 3)
