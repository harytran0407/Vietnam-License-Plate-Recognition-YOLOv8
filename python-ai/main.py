"""
main.py — AI Parking System
============================
Pipeline per frame:
  Webcam → YOLOv8 vehicle detect → SORT track
        → LP detect / fallback crop
        → EasyOCR  →  PlateStabilizer (majority-vote buffer)
        → ParkingManager (check-in / check-out / cooldown)
        → HUD overlay + sidebar

Controls:  Q = quit   S = screenshot
"""

import cv2
import numpy as np
import datetime
import os
import sys
import time

from ultralytics import YOLO

try:
    from sort.sort import Sort
except ImportError:
    print('[ERROR] SORT not found — clone https://github.com/abewley/sort '
          'and place the "sort/" folder next to this script.')
    sys.exit(1)

from util import get_car, read_license_plate, write_csv
from plate_stabilizer import PlateStabilizer, DisplayState
from parking_manager import ParkingManager
from video_stream import start_stream_server, frame_buffer, stop_stream_server

# ══════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════
WEBCAM_INDEX       = 0
YOLO_VEHICLE_MODEL = 'yolov8n.pt'
LP_DETECTOR_MODEL  = './models/license_plate_detector.pt'
OUTPUT_CSV         = './test.csv'
VEHICLE_CLASSES    = [2, 3, 5, 7]          # COCO: car, motorbike, bus, truck
USE_LP_FALLBACK    = True                   # crop lower vehicle if no model
WINDOW_NAME        = 'AI Parking System  |  Q = Thoat'
HUD_W              = 360                    # sidebar width in pixels

# BGR colour palette
C_GREEN    = (0,   230, 118)
C_BLUE     = (68,  138, 255)
C_ORANGE   = (0,   165, 255)
C_RED      = (60,  60,  220)
C_YELLOW   = (0,   220, 220)
C_GREY     = (130, 130, 130)
C_DARK     = (22,  22,  30)
C_PANEL    = (28,  28,  38)
C_TITLE_BG = (42,  42,  62)
C_WHITE    = (220, 220, 220)

# State → colour mapping for plate overlay
STATE_CLR = {
    'IDLE':      C_GREY,
    'VERIFYING': C_ORANGE,
    'CONFIRMED': C_GREEN,
}
EVENT_CLR = {
    'checkin':  C_GREEN,
    'checkout': C_BLUE,
    'cooldown': (100, 130, 60),
}


# ══════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ══════════════════════════════════════════════════════════════════════════
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_corner_box(img, tl, br, color, thickness=2, seg=28):
    x1, y1 = tl; x2, y2 = br
    for p, q in [
        ((x1, y1), (x1, y1+seg)), ((x1, y1), (x1+seg, y1)),
        ((x1, y2), (x1, y2-seg)), ((x1, y2), (x1+seg, y2)),
        ((x2, y1), (x2-seg, y1)), ((x2, y1), (x2, y1+seg)),
        ((x2, y2), (x2-seg, y2)), ((x2, y2), (x2, y2-seg)),
    ]:
        cv2.line(img, p, q, color, thickness)


def bg_text(img, text, org, scale=0.55, color=C_WHITE,
            thick=1, bg=C_DARK, pad=3):
    (tw, th), bl = cv2.getTextSize(text, _FONT, scale, thick)
    x, y = org
    cv2.rectangle(img,
                  (x - pad,      y - th - bl - pad),
                  (x + tw + pad, y + pad),
                  bg, -1)
    cv2.putText(img, text, (x, y - bl), _FONT, scale, color, thick, cv2.LINE_AA)


def badge(img, text, org, bg_color, scale=0.58, thick=2, pad=5):
    (tw, th), bl = cv2.getTextSize(text, _FONT, scale, thick)
    x, y = org
    cv2.rectangle(img,
                  (x - pad,      y - th - bl - pad),
                  (x + tw + pad, y + pad),
                  bg_color, -1)
    cv2.putText(img, text, (x, y - bl), _FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)


def progress_bar(img, org, width, progress, color, height=6):
    """Horizontal progress bar, progress ∈ [0, 1]."""
    x, y = org
    cv2.rectangle(img, (x, y), (x + width, y + height), C_DARK, -1)
    filled = int(width * min(max(progress, 0.0), 1.0))
    if filled > 0:
        cv2.rectangle(img, (x, y), (x + filled, y + height), color, -1)


# ══════════════════════════════════════════════════════════════════════════
# Per-plate overlay  (drawn on the video frame)
# ══════════════════════════════════════════════════════════════════════════

def draw_plate_overlay(frame, x1, y1, x2, y2, ds: DisplayState,
                       xcar1, ycar1, active_since: datetime.datetime | None):
    """
    Draw all plate-related overlays on the main video frame.

    Layers (bottom → top):
      • plate bounding box  (colour = state)
      • candidate plate text + confidence
      • VERIFYING progress bar  or  CONFIRMED / IN / OUT badge
      • elapsed parking time on vehicle box (if PARKING)
    """
    clr = STATE_CLR.get(ds.status, C_GREY)

    # Plate bounding box
    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), clr, 2)

    if ds.status == 'IDLE' or not ds.plate:
        return

    # Plate text + confidence
    label = f'{ds.plate}  {ds.confidence:.2f}'
    bg_text(frame, label, (int(x1), int(y1) - 26),
            scale=0.65, color=C_YELLOW, thick=2)

    if ds.status == 'VERIFYING':
        # Vote counter
        vote_txt = f'{ds.votes}/{ds.needed}'
        bg_text(frame, vote_txt, (int(x1), int(y1) - 52),
                scale=0.5, color=C_ORANGE, thick=1)
        # Progress bar
        bar_w = int(x2 - x1)
        progress_bar(frame, (int(x1), int(y1) - 60),
                     bar_w, ds.progress, C_ORANGE)

    elif ds.status == 'CONFIRMED':
        badge(frame, 'CONFIRMED', (int(x1), int(y1) - 52), C_GREEN)

    # Elapsed time on vehicle corner box
    if active_since is not None:
        el = (datetime.datetime.now() - active_since).total_seconds() / 60
        el_txt = f'{int(el)}p {int((el % 1) * 60):02d}s'
        bg_text(frame, el_txt, (int(xcar1), int(ycar1) - 6),
                scale=0.48, color=C_GREEN, thick=1)


# ══════════════════════════════════════════════════════════════════════════
# Sidebar panel
# ══════════════════════════════════════════════════════════════════════════

def build_sidebar(h: int, mgr: ParkingManager, events: list) -> np.ndarray:
    panel = np.full((h, HUD_W, 3), C_PANEL, dtype=np.uint8)

    def txt(s, y, scale=0.50, color=C_WHITE, thick=1, x=10):
        cv2.putText(panel, s, (x, y), _FONT, scale, color, thick, cv2.LINE_AA)

    # ── Title bar ────────────────────────────────────────────────────
    cv2.rectangle(panel, (0, 0), (HUD_W, 42), C_TITLE_BG, -1)
    txt('BAI GIU XE AI', 27, scale=0.62, color=(120, 220, 255), thick=2)
    now_str = datetime.datetime.now().strftime('%H:%M:%S')
    (tw, _), _ = cv2.getTextSize(now_str, _FONT, 0.48, 1)
    cv2.putText(panel, now_str, (HUD_W - tw - 10, 27),
                _FONT, 0.48, C_GREY, 1, cv2.LINE_AA)

    # ── Stats ────────────────────────────────────────────────────────
    st = mgr.get_stats()
    y = 68
    for label, val, col in [
        ('Dang gui xe', st['currently_parked'], C_GREEN),
        ('Hom nay',     st['total_today'],      C_BLUE),
        ('Tong cong',   st['total_all_time'],   C_WHITE),
    ]:
        txt(f'{label}:', y, scale=0.48, color=C_GREY)
        val_str = str(val)
        (tw, _), _ = cv2.getTextSize(val_str, _FONT, 0.60, 2)
        cv2.putText(panel, val_str, (HUD_W - tw - 10, y),
                    _FONT, 0.60, col, 2, cv2.LINE_AA)
        y += 26

    # ── Divider ──────────────────────────────────────────────────────
    cv2.line(panel, (10, y + 4), (HUD_W - 10, y + 4), (55, 55, 70), 1)
    y += 16
    txt('LICH SU DETECT', y, scale=0.44, color=(110, 110, 140))
    y += 22

    # ── Event log ────────────────────────────────────────────────────
    for ev in reversed(events[-16:]):
        if y > h - 38:
            break
        evtype = ev['event']
        col, tag = {
            'checkin':  (C_GREEN,  '[VAO]'),
            'checkout': (C_BLUE,   '[RA ]'),
            'cooldown': (C_GREY,   '[---]'),
        }.get(evtype, (C_GREY, '[???]'))

        txt(f"{tag} {ev['plate']}", y, scale=0.48, color=col, thick=1)
        y += 17
        detail = f"     {ev['time']}"
        if ev.get('duration'):
            detail += f"  {ev['duration']}"
        txt(detail, y, scale=0.40, color=(100, 100, 110))
        y += 19

    # ── Active list ───────────────────────────────────────────────────
    active = mgr.active_vehicles
    if active:
        cv2.line(panel, (10, h - 28 - len(active) * 20 - 6),
                 (HUD_W - 10, h - 28 - len(active) * 20 - 6),
                 (55, 55, 70), 1)
        ay = h - 28 - len(active[-6:]) * 20
        for rec in active[-6:]:
            el = (datetime.datetime.now() - rec.checkin_time).total_seconds() / 60
            txt(f'{rec.license_plate}  {int(el)}p',
                ay, scale=0.44, color=(80, 210, 100))
            ay += 20

    # ── Bottom hint ───────────────────────────────────────────────────
    cv2.rectangle(panel, (0, h - 26), (HUD_W, h), (35, 35, 48), -1)
    txt('Q=Thoat  S=Chup man hinh', h - 9,
        scale=0.38, color=(90, 90, 100))

    return panel


# ══════════════════════════════════════════════════════════════════════════
# Load models
# ══════════════════════════════════════════════════════════════════════════
print('[INFO] Loading YOLOv8 vehicle detector …')
coco_model = YOLO(YOLO_VEHICLE_MODEL)

lp_model_available = os.path.isfile(LP_DETECTOR_MODEL)
if lp_model_available:
    print('[INFO] Loading license plate detector …')
    lp_detector = YOLO(LP_DETECTOR_MODEL)
else:
    print(f'[WARN] {LP_DETECTOR_MODEL!r} not found — fallback crop only.')

# ══════════════════════════════════════════════════════════════════════════
# Webcam
# ══════════════════════════════════════════════════════════════════════════
print(f'[INFO] Opening webcam {WEBCAM_INDEX} …')
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print(f'[ERROR] Cannot open webcam {WEBCAM_INDEX}.')
    sys.exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print('[INFO] Ready. Q=quit  S=screenshot')

# ══════════════════════════════════════════════════════════════════════════
# Runtime objects
# ══════════════════════════════════════════════════════════════════════════
mot_tracker = Sort()
stabilizer  = PlateStabilizer()
parking_mgr = ParkingManager()

# ── Start MJPEG stream server (background thread) ─────────────────────────
start_stream_server()

recent_events: list[dict] = []
yolo_results:  dict       = {}
frame_nmr = -1

# ══════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════
while True:
    ret, frame = cap.read()
    if not ret:
        print('[WARN] Lost webcam feed.')
        break

    frame_nmr += 1
    ts_str   = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    mono_now = time.monotonic()
    yolo_results[frame_nmr] = {}

    # ── 1. Vehicle detection ─────────────────────────────────────────
    veh_dets = []
    for det in coco_model(frame, verbose=False)[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = det
        if int(class_id) in VEHICLE_CLASSES:
            veh_dets.append([x1, y1, x2, y2, score])
            draw_corner_box(frame,
                            (int(x1), int(y1)), (int(x2), int(y2)),
                            C_GREEN, thickness=2)

    # ── 2. SORT tracking ─────────────────────────────────────────────
    track_ids = (mot_tracker.update(np.asarray(veh_dets))
                 if veh_dets
                 else mot_tracker.update(np.empty((0, 5))))

    # ── 3. Plate regions ─────────────────────────────────────────────
    lp_regions: list[tuple] = []  # (x1,y1,x2,y2, score, from_model)

    if lp_model_available:
        for lp in lp_detector(frame, verbose=False)[0].boxes.data.tolist():
            lp_regions.append((*lp[:5], True))

    if USE_LP_FALLBACK:
        for xc1, yc1, xc2, yc2, _ in track_ids:
            # Skip if the model already found a plate in this vehicle
            if any(px1 > xc1 and py1 > yc1 and px2 < xc2 and py2 < yc2
                   for px1, py1, px2, py2, _, fm in lp_regions if fm):
                continue
            h_v = yc2 - yc1
            lp_regions.append((xc1, max(0, yc2 - int(h_v * 0.22)),
                                xc2, yc2, 0.5, False))

    # ── 4. OCR → stabilizer → parking manager ────────────────────────
    active_map = {r.license_plate: r for r in parking_mgr.active_vehicles}

    for x1, y1, x2, y2, lp_score, from_model in lp_regions:
        # Assign plate region to a tracked vehicle
        fake_lp = (x1, y1, x2, y2, lp_score, 0)
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(fake_lp, track_ids)
        if car_id == -1:
            continue

        lp_crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if lp_crop.size == 0:
            continue

        # ── OCR ──────────────────────────────────────────────────
        raw_text, ocr_score = read_license_plate(lp_crop)

        # Feed into stabilizer even if OCR returned None (tick handled below)
        confirmed_plate = None
        if raw_text is not None:
            result = stabilizer.feed(car_id, raw_text,
                                     ocr_score or 0.0, mono_now)
            if result is not None:
                # ── Majority vote passed ──────────────────────
                confirmed_plate = result.plate
                event = parking_mgr.process_confirmed(
                    confirmed_plate, frame,
                    votes=result.votes,
                    confidence=result.confidence,
                )

                # Append to sidebar event log
                entry = {
                    'plate': confirmed_plate,
                    'event': event,
                    'time':  datetime.datetime.now().strftime('%H:%M:%S'),
                }
                if event == 'checkout':
                    done = [r for r in parking_mgr.history
                            if r.license_plate == confirmed_plate
                            and r.status == 'CHECKED_OUT']
                    if done:
                        m = done[-1].total_minutes
                        entry['duration'] = f'{int(m)}p{int((m % 1) * 60):02d}s'
                if event != 'cooldown' or (
                    not recent_events or
                    recent_events[-1].get('plate') != confirmed_plate
                ):
                    recent_events.append(entry)

                # After confirm → reset buffer so car can be processed
                # again after the cooldown expires
                stabilizer.reset(car_id)

                # Store in YOLO CSV
                yolo_results[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox':       [x1, y1, x2, y2],
                        'text':       confirmed_plate,
                        'bbox_score': lp_score,
                        'text_score': result.confidence,
                    },
                    'timestamp': ts_str,
                }

        # ── Display state for this car_id ─────────────────────
        ds = stabilizer.get_state(car_id)

        # If just confirmed, show CONFIRMED state for this frame
        if confirmed_plate:
            from plate_stabilizer import DisplayState as DS
            ds = DS(
                status='CONFIRMED',
                plate=confirmed_plate,
                confidence=ocr_score or 0.0,
                votes=0, needed=0, progress=1.0,
            )

        # Elapsed time if this vehicle is actively parked
        plate_key = ds.plate or confirmed_plate or ''
        since = active_map.get(plate_key, None)
        since_dt = since.checkin_time if since else None

        draw_plate_overlay(frame, x1, y1, x2, y2, ds,
                           xcar1, ycar1, since_dt)

        # Draw event badge on plate box if just confirmed
        if confirmed_plate and recent_events:
            last_ev = recent_events[-1]
            if last_ev['plate'] == confirmed_plate:
                ev_txt = {'checkin': 'VAO', 'checkout': 'RA',
                          'cooldown': '...'}.get(last_ev['event'], '')
                ev_clr = EVENT_CLR.get(last_ev['event'], C_GREY)
                if ev_txt:
                    badge(frame, ev_txt,
                          (int(x1), int(y1) - 80), ev_clr)

    # ── 5. Stabilizer housekeeping (evict stale buffers) ─────────────
    stabilizer.tick(mono_now)

    # ── 6. Frame HUD (top-left) ───────────────────────────────────────
    bg_text(frame, f'Frame {frame_nmr}  {ts_str}',
            (8, 22), scale=0.46, color=(160, 160, 160))
    st = parking_mgr.get_stats()
    bg_text(frame,
            f"Dang gui: {st['currently_parked']}  "
            f"Hom nay: {st['total_today']}",
            (8, 44), scale=0.48, color=C_GREEN)

    # ── 7. Composite = video frame + sidebar ──────────────────────────
    sidebar   = build_sidebar(frame.shape[0], parking_mgr, recent_events)
    composite = np.hstack([frame, sidebar])
    cv2.imshow(WINDOW_NAME, composite)

    # ── Push to MJPEG stream (browser dashboard) ──────────────────────
    frame_buffer.update(composite)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print('[INFO] Quitting …')
        break
    elif key == ord('s'):
        fname = f'screenshot_{frame_nmr}.jpg'
        cv2.imwrite(fname, composite)
        print(f'[INFO] Screenshot → {fname}')

# ══════════════════════════════════════════════════════════════════════════
# Cleanup
# ══════════════════════════════════════════════════════════════════════════
write_csv(yolo_results, OUTPUT_CSV)
print(f'[INFO] YOLO CSV           → {OUTPUT_CSV}')
print('[INFO] Parking history    → parking_history.csv')
print('[INFO] Snapshots          → snapshots/')
stop_stream_server()
cap.release()
cv2.destroyAllWindows()
