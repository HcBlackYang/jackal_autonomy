#!/usr/bin/env python3
import easyocr
import numpy as np

reader = easyocr.Reader(['en'], gpu=False)

def ocr_detector(img):
    h, w = img.shape[:2]
    center = np.array([w / 2, h / 2])

    results = reader.readtext(img)

    best_score = -float('inf')
    best_digit = ""

    for bbox, text, conf in results:
        text = text.strip()
        if not text.isdigit():
            continue

        # Only take single digit
        digit = text[-1]  # Take the last character

        # Distance to center
        pts = np.array(bbox)
        text_center = pts.mean(axis=0)
        dist = np.linalg.norm(text_center - center)

        # Box area (approximate as rectangle)
        width = np.linalg.norm(pts[0] - pts[1])
        height = np.linalg.norm(pts[1] - pts[2])
        area = width * height

        # Calculate score
        score = area - dist * 0.5

        if score > best_score:
            best_score = score
            best_digit = digit

    return best_digit