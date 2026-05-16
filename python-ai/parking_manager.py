"""
parking_manager.py
==================
Parking state machine – operates ONLY on confirmed (stabilised) plates.

Flow:
    PlateStabilizer confirms plate
        → ParkingManager.process_confirmed()
            → CHECK-IN  (plate not seen in last 30 s & not PARKING)
            → CHECK-OUT (plate already PARKING)
            → COOLDOWN  (seen within last 30 s)
"""

from __future__ import annotations

import csv
import os
import time
import uuid
import datetime
import threading
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import requests

log = logging.getLogger('ParkingManager')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)

# ══════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════
COOLDOWN_SECONDS = 30
ASPNET_API_BASE  = 'http://localhost:5000/api'
ASPNET_TIMEOUT   = 3
CAMERA_ID        = 'CAM-01'
CSV_PATH         = 'parking_history.csv'
SNAPSHOT_DIR     = 'snapshots'
SAVE_SNAPSHOTS   = True

CSV_FIELDS = [
    'id', 'license_plate', 'camera_id',
    'checkin_time', 'checkout_time',
    'total_minutes', 'total_hours',
    'status', 'snapshot_path',
    'votes', 'confidence',
]


# ══════════════════════════════════════════════════════════════════════════
# Data model
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class ParkingRecord:
    id:            str
    license_plate: str
    camera_id:     str
    checkin_time:  datetime.datetime
    checkout_time: Optional[datetime.datetime] = None
    total_minutes: float = 0.0
    total_hours:   float = 0.0
    status:        str   = 'PARKING'
    snapshot_path: str   = ''
    votes:         int   = 0
    confidence:    float = 0.0

    def to_dict(self) -> dict:
        return {
            'id':            self.id,
            'license_plate': self.license_plate,
            'camera_id':     self.camera_id,
            'checkin_time':  self.checkin_time.isoformat(sep=' '),
            'checkout_time': (self.checkout_time.isoformat(sep=' ')
                              if self.checkout_time else ''),
            'total_minutes': round(self.total_minutes, 2),
            'total_hours':   round(self.total_hours, 4),
            'status':        self.status,
            'snapshot_path': self.snapshot_path,
            'votes':         self.votes,
            'confidence':    round(self.confidence, 3),
        }


# ══════════════════════════════════════════════════════════════════════════
# Manager
# ══════════════════════════════════════════════════════════════════════════
class ParkingManager:

    def __init__(self) -> None:
        self._lock      = threading.Lock()
        self._active:   dict[str, ParkingRecord] = {}   # plate → record
        self._last_seen: dict[str, float]        = {}   # plate → monotonic time
        self._history:  list[ParkingRecord]      = []

        Path(SNAPSHOT_DIR).mkdir(exist_ok=True)
        self._ensure_csv()
        self._load_csv()

    # ── Main entry point ──────────────────────────────────────────────

    def process_confirmed(
        self,
        plate:      str,
        frame=None,
        votes:      int   = 0,
        confidence: float = 0.0,
    ) -> str:
        """
        Called when PlateStabilizer emits a StabilizedPlate.

        Returns: 'checkin' | 'checkout' | 'cooldown'
        """
        now_mono = time.monotonic()
        with self._lock:
            elapsed = now_mono - self._last_seen.get(plate, 0)
            if elapsed < COOLDOWN_SECONDS:
                remaining = int(COOLDOWN_SECONDS - elapsed)
                log.debug(f'[COOLDOWN] {plate} – {remaining}s remaining')
                return 'cooldown'

            self._last_seen[plate] = now_mono

            if plate not in self._active:
                return self._checkin(plate, frame, votes, confidence)
            else:
                return self._checkout(plate, frame)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def active_vehicles(self) -> list[ParkingRecord]:
        with self._lock:
            return list(self._active.values())

    @property
    def history(self) -> list[ParkingRecord]:
        with self._lock:
            return list(self._history)

    def get_stats(self) -> dict:
        with self._lock:
            today = datetime.date.today()
            return {
                'currently_parked': len(self._active),
                'total_today':      sum(
                    1 for r in self._history
                    if r.checkin_time.date() == today
                ),
                'total_all_time':   len(self._history),
            }

    # ── Check-in ──────────────────────────────────────────────────────

    def _checkin(self, plate: str, frame, votes: int, confidence: float) -> str:
        rec = ParkingRecord(
            id            = str(uuid.uuid4())[:8].upper(),
            license_plate = plate,
            camera_id     = CAMERA_ID,
            checkin_time  = datetime.datetime.now(),
            status        = 'PARKING',
            votes         = votes,
            confidence    = confidence,
        )
        if SAVE_SNAPSHOTS and frame is not None:
            rec.snapshot_path = self._snapshot(plate, frame, 'in')

        self._active[plate] = rec
        self._history.append(rec)
        self._append_csv(rec)
        log.info(f'CHECK-IN  {plate}  votes={votes}  conf={confidence:.2f}  id={rec.id}')
        self._post_async('checkin', rec)
        return 'checkin'

    # ── Check-out ─────────────────────────────────────────────────────

    def _checkout(self, plate: str, frame) -> str:
        rec = self._active.pop(plate)
        rec.checkout_time = datetime.datetime.now()
        delta             = rec.checkout_time - rec.checkin_time
        rec.total_minutes = delta.total_seconds() / 60
        rec.total_hours   = rec.total_minutes / 60
        rec.status        = 'CHECKED_OUT'

        if SAVE_SNAPSHOTS and frame is not None:
            out_path = self._snapshot(plate, frame, 'out')
            if not rec.snapshot_path:
                rec.snapshot_path = out_path

        self._rewrite_csv()
        log.info(f'CHECK-OUT {plate}  {rec.total_minutes:.1f}min  id={rec.id}')
        self._post_async('checkout', rec)
        return 'checkout'

    # ── CSV ───────────────────────────────────────────────────────────

    def _ensure_csv(self) -> None:
        if not os.path.isfile(CSV_PATH):
            with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    def _load_csv(self) -> None:
        try:
            with open(CSV_PATH, newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    ci = datetime.datetime.fromisoformat(row['checkin_time'])
                    co = (datetime.datetime.fromisoformat(row['checkout_time'])
                          if row.get('checkout_time') else None)
                    rec = ParkingRecord(
                        id            = row['id'],
                        license_plate = row['license_plate'],
                        camera_id     = row.get('camera_id', CAMERA_ID),
                        checkin_time  = ci,
                        checkout_time = co,
                        total_minutes = float(row.get('total_minutes') or 0),
                        total_hours   = float(row.get('total_hours')   or 0),
                        status        = row['status'],
                        snapshot_path = row.get('snapshot_path', ''),
                        votes         = int(row.get('votes')      or 0),
                        confidence    = float(row.get('confidence') or 0),
                    )
                    self._history.append(rec)
                    if rec.status == 'PARKING':
                        self._active[rec.license_plate] = rec
            log.info(f'Loaded {len(self._history)} records '
                     f'({len(self._active)} still PARKING)')
        except Exception as exc:
            log.warning(f'CSV load skipped: {exc}')

    def _append_csv(self, rec: ParkingRecord) -> None:
        with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(rec.to_dict())

    def _rewrite_csv(self) -> None:
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(r.to_dict() for r in self._history)

    # ── Snapshot ──────────────────────────────────────────────────────

    def _snapshot(self, plate: str, frame, suffix: str) -> str:
        safe = re.sub(r'[^A-Z0-9]', '_', plate)
        ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(SNAPSHOT_DIR, f'{safe}_{suffix}_{ts}.jpg')
        cv2.imwrite(path, frame)
        return path

    # ── ASP.NET ───────────────────────────────────────────────────────

    def _post_async(self, event: str, rec: ParkingRecord) -> None:
        threading.Thread(target=self._post, args=(event, rec), daemon=True).start()

    def _post(self, event: str, rec: ParkingRecord) -> None:
        event = event.lower().strip()
        url = f'{ASPNET_API_BASE}/parking/{event}'
        
        # Build payload chuẩn theo đúng mong đợi của C# Dto
        if event == "checkin":
            payload = {
                "id": rec.id,
                "licensePlate": rec.license_plate,
                "cameraId": rec.camera_id,
                "checkinTime": rec.checkin_time.isoformat(),
                "snapshotPath": rec.snapshot_path
            }
        elif event == "checkout":
            payload = {
                "id": rec.id,
                "licensePlate": rec.license_plate,
                "cameraId": rec.camera_id,
                "checkoutTime": rec.checkout_time.isoformat() if rec.checkout_time else None,
                "snapshotPath": rec.snapshot_path,
                "totalMinutes": rec.total_minutes,
                "totalHours": rec.total_hours
            }
        else:
            log.warning(f"Sự kiện không hợp lệ: {event}")
            return

        try:
            # Gửi payload đã chuẩn hóa thay vì rec.to_dict()
            r = requests.post(url, json=payload, timeout=ASPNET_TIMEOUT)
            log.info(f'API {event} → {r.status_code} | {r.text}')
        except requests.exceptions.ConnectionError:
            log.warning(f'API {event}: ASP.NET not reachable ({url})')
        except Exception as exc:
            log.warning(f'API {event}: {exc}')


# ── late import to avoid circular at module level ──────────────────────────
import re   # noqa: E402 (used in _snapshot)
