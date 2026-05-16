"""
util.py
=======
Low-level helpers:
  • Image preprocessing pipeline for license-plate crops
  • EasyOCR wrapper  (returns raw text + confidence)
  • Vehicle-to-plate assignment  (get_car)
  • CSV writer for YOLO frame results

Plate normalisation / validation now lives in plate_stabilizer.py.
util.py only returns the best raw OCR string; the stabilizer decides
whether it is a valid VN plate.
"""

import cv2
import numpy as np
import easyocr

# ── EasyOCR reader (initialised once at import time) ─────────────────────
reader = easyocr.Reader(['en'], gpu=False)


# ══════════════════════════════════════════════════════════════════════════
# Image preprocessing
# ══════════════════════════════════════════════════════════════════════════

def preprocess_license_plate(crop: np.ndarray) -> list[np.ndarray]:
    """
    Return a list of preprocessed variants of the plate crop.
    Each variant is tried for OCR; the one with the best valid result wins.

    Pipeline:
      grayscale → resize to h=64 → denoise → CLAHE
        → [CLAHE, Otsu, inv-Otsu, Adaptive, Sharpened]
    """
    # 1. Grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()

    # 2. Upscale (OCR works better with larger images)
    h, w = gray.shape[:2]
    if h < 64:
        scale = 64 / h
        gray = cv2.resize(gray, (int(w * scale), 64),
                          interpolation=cv2.INTER_CUBIC)

    # 3. Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # 4. CLAHE
    clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(denoised)

    # 5. Otsu
    _, otsu = cv2.threshold(
        clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 6. Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        clahe_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8,
    )

    # 7. Sharpened
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp  = cv2.filter2D(clahe_img, -1, kernel)

    return [
        clahe_img,                      # variant 0 – baseline
        otsu,                           # variant 1 – binary (light bg)
        cv2.bitwise_not(otsu),          # variant 2 – binary (dark bg)
        adaptive,                       # variant 3 – adaptive
        sharp,                          # variant 4 – sharpened
    ]


# ══════════════════════════════════════════════════════════════════════════
# OCR
# ══════════════════════════════════════════════════════════════════════════

def read_license_plate(crop: np.ndarray) -> tuple[str | None, float | None]:
    """
    Run EasyOCR on all preprocessed variants of `crop`.

    Returns the (raw_combined_text, avg_confidence) of the best reading,
    or (None, None) if nothing was detected.

    NOTE: This function does NOT validate VN plate format.
          Validation + normalisation is handled by PlateStabilizer.feed().
    """
    variants = preprocess_license_plate(crop)

    best_text:  str | None = None
    best_score: float      = 0.0

    for img in variants:
        try:
            detections = reader.readtext(img)
        except Exception:
            continue

        if not detections:
            continue

        # Concatenate all fragments on this plate image
        combined = ''
        total_score = 0.0
        for _, text, score in detections:
            combined    += text.upper().replace(' ', '')
            total_score += score
        avg_score = total_score / len(detections)

        # Pick the variant that gives the longest non-empty result
        # with the highest average confidence
        if combined and avg_score > best_score:
            best_text  = combined
            best_score = avg_score

    return (best_text, best_score) if best_text else (None, None)


# ══════════════════════════════════════════════════════════════════════════
# Vehicle–plate assignment
# ══════════════════════════════════════════════════════════════════════════

def get_car(
    license_plate: tuple,
    vehicle_track_ids,
) -> tuple:
    """
    Find which tracked vehicle contains the given license-plate region.

    Args:
        license_plate: (x1, y1, x2, y2, score, class_id)
        vehicle_track_ids: array of [x1, y1, x2, y2, track_id] rows

    Returns:
        (xcar1, ycar1, xcar2, ycar2, car_id)  or  (-1, -1, -1, -1, -1)
    """
    x1, y1, x2, y2 = license_plate[:4]

    for row in vehicle_track_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = row
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return row  # type: ignore[return-value]

    return -1, -1, -1, -1, -1


# ══════════════════════════════════════════════════════════════════════════
# CSV export  (raw YOLO / OCR results per frame)
# ══════════════════════════════════════════════════════════════════════════

def write_csv(results: dict, output_path: str) -> None:
    """
    Write per-frame YOLO + OCR results to CSV.

    Columns:
        frame_nmr, car_id, car_bbox, license_plate_bbox,
        license_plate_bbox_score, license_number, license_number_score,
        timestamp
    """
    header = ('frame_nmr,car_id,car_bbox,license_plate_bbox,'
               'license_plate_bbox_score,license_number,'
               'license_number_score,timestamp\n')

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        f.write(header)

        for frame_nmr, frame_data in results.items():
            for car_id, car_data in frame_data.items():
                if not ('car' in car_data
                        and 'license_plate' in car_data
                        and 'text' in car_data['license_plate']):
                    continue

                cb  = car_data['car']['bbox']
                lp  = car_data['license_plate']
                lb  = lp['bbox']

                f.write(
                    '{},{},{},{},{},{},{},{}\n'.format(
                        frame_nmr,
                        car_id,
                        '[{:.1f} {:.1f} {:.1f} {:.1f}]'.format(*cb),
                        '[{:.1f} {:.1f} {:.1f} {:.1f}]'.format(*lb),
                        f"{lp['bbox_score']:.4f}",
                        lp['text'],
                        f"{lp['text_score']:.4f}",
                        car_data.get('timestamp', ''),
                    )
                )


# ══════════════════════════════════════════════════════════════════════════
# Misc
# ══════════════════════════════════════════════════════════════════════════

def crop_lower_vehicle(frame: np.ndarray,
                       car_bbox: tuple,
                       ratio: float = 0.22) -> np.ndarray:
    """
    Crop the lower `ratio` fraction of a vehicle bounding box.
    Used as a fallback when the LP detector model is absent.
    """
    x1, y1, x2, y2 = (int(v) for v in car_bbox)
    crop_y1 = max(0, y2 - int((y2 - y1) * ratio))
    return frame[crop_y1:y2, x1:x2]
