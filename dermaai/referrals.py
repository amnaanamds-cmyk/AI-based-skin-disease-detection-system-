"""Teledermatology referrals: consented case submission and a clinician review queue.

Images are stored only when the patient explicitly sends a case to a
clinician. The patient receives a one-time secret that lets them check the
reply and delete the case at any time; only its SHA-256 hash is stored.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import secrets
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

URGENCY_RANK = {"high": 0, "moderate": 1, "retake": 2, "low": 3}
STATUSES = ["new", "in_review", "closed"]
FHIR_BASE = os.getenv("DERMAAI_FHIR_BASE", "https://dermaai.local/fhir")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id TEXT PRIMARY KEY,
    token_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'new',
    urgency TEXT NOT NULL,
    urgency_rank INTEGER NOT NULL,
    malignancy REAL NOT NULL,
    patient TEXT NOT NULL,
    analysis TEXT NOT NULL,
    image BLOB NOT NULL,
    response TEXT,
    clinical_impression TEXT,
    reviewer TEXT,
    reviewed_at TEXT
);
CREATE INDEX IF NOT EXISTS cases_queue ON cases(status, urgency_rank, malignancy DESC, created_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _jpeg(img: Image.Image, max_side: int = 1024) -> bytes:
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def summarize_analysis(result: dict) -> dict:
    """Keep what a clinician needs; drop large images and outlines."""
    return {
        "triage": result["triage"],
        "probabilities": result["prediction"]["probabilities"],
        "uncertainty": result["uncertainty"],
        "quality": {k: result["quality"][k] for k in ("score", "acceptable", "issues")},
        "abcde": {k: v for k, v in (result.get("abcde") or {}).items() if k != "outline"} or None,
        "ood": result.get("ood"),
        "model": result["model"],
        "analysis_id": result["id"],
    }


class CaseStore:
    def __init__(self, path: str | Path | None = None):
        self.path = str(path or os.getenv("DERMAAI_DB", "dermaai_cases.db"))
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def create(self, image: Image.Image, analysis: dict, patient: dict) -> tuple[str, str]:
        case_id, token = uuid.uuid4().hex[:12], secrets.token_urlsafe(24)
        t = analysis["triage"]
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO cases (id, token_hash, created_at, urgency, urgency_rank, malignancy, patient, analysis, image)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (case_id, _hash(token), _now(), t["level"], URGENCY_RANK.get(t["level"], 9),
                 float(t["malignancy_probability"]), json.dumps(patient), json.dumps(analysis), _jpeg(image)))
        return case_id, token

    def _row(self, case_id: str):
        with self._lock:
            return self._db.execute("SELECT * FROM cases WHERE id = ?", (case_id,)).fetchone()

    def verify(self, case_id: str, token: str) -> bool:
        row = self._row(case_id)
        return bool(row) and secrets.compare_digest(row["token_hash"], _hash(token))

    def get(self, case_id: str, with_image: bool = False) -> dict | None:
        row = self._row(case_id)
        if not row:
            return None
        out = {k: row[k] for k in row.keys() if k not in ("token_hash", "image", "patient", "analysis")}
        out["patient"] = json.loads(row["patient"])
        out["analysis"] = json.loads(row["analysis"])
        if with_image:
            out["image"] = "data:image/jpeg;base64," + base64.b64encode(row["image"]).decode()
        return out

    def patient_view(self, case_id: str) -> dict:
        c = self.get(case_id)
        return {k: c[k] for k in ("id", "created_at", "status", "urgency", "response", "reviewed_at")}

    def queue(self, status: str | None = None, limit: int = 200) -> list[dict]:
        sql = ("SELECT id, created_at, status, urgency, malignancy, patient, analysis, reviewed_at FROM cases"
               + (" WHERE status = ?" if status else "")
               + " ORDER BY CASE status WHEN 'closed' THEN 1 ELSE 0 END, urgency_rank, malignancy DESC, created_at LIMIT ?")
        with self._lock:
            rows = self._db.execute(sql, (status, limit) if status else (limit,)).fetchall()
        out = []
        for r in rows:
            a, p = json.loads(r["analysis"]), json.loads(r["patient"])
            out.append({"id": r["id"], "created_at": r["created_at"], "status": r["status"], "urgency": r["urgency"],
                        "malignancy": r["malignancy"], "top": a["probabilities"][0]["name"],
                        "age": p.get("age"), "sex": p.get("sex"), "localization": p.get("localization"),
                        "reviewed_at": r["reviewed_at"]})
        return out

    def review(self, case_id: str, status: str, response: str | None, impression: str | None, reviewer: str | None) -> bool:
        with self._lock, self._db:
            cur = self._db.execute(
                "UPDATE cases SET status = ?, response = ?, clinical_impression = ?, reviewer = ?, reviewed_at = ? WHERE id = ?",
                (status, response, impression, reviewer, _now(), case_id))
        return cur.rowcount == 1

    def delete(self, case_id: str) -> bool:
        with self._lock, self._db:
            return self._db.execute("DELETE FROM cases WHERE id = ?", (case_id,)).rowcount == 1

    def stats(self) -> dict:
        with self._lock:
            rows = self._db.execute("SELECT status, urgency, COUNT(*) n FROM cases GROUP BY status, urgency").fetchall()
        out = {"total": 0, "open": 0, "by_status": {}, "open_by_urgency": {}}
        for r in rows:
            out["total"] += r["n"]
            out["by_status"][r["status"]] = out["by_status"].get(r["status"], 0) + r["n"]
            if r["status"] != "closed":
                out["open"] += r["n"]
                out["open_by_urgency"][r["urgency"]] = out["open_by_urgency"].get(r["urgency"], 0) + r["n"]
        return out


def to_fhir(case: dict, image_data_url: str | None = None) -> dict:
    """HL7 FHIR R4 Bundle (DiagnosticReport + Observations + Media) for EHR import."""
    a, p = case["analysis"], case["patient"]
    pid, rid = f"patient-{case['id']}", f"report-{case['id']}"
    status = "final" if case["status"] == "closed" else "preliminary"
    obs = []
    for i, pr in enumerate(a["probabilities"]):
        obs.append({
            "resourceType": "Observation", "id": f"obs-{case['id']}-{i}", "status": status,
            "code": {"text": f"AI estimated probability: {pr['name']}"},
            "subject": {"reference": f"Patient/{pid}"},
            "valueQuantity": {"value": round(pr["probability"] * 100, 2), "unit": "%",
                              "system": "http://unitsofmeasure.org", "code": "%"},
        })
    patient = {"resourceType": "Patient", "id": pid}
    if p.get("sex") in {"male", "female"}:
        patient["gender"] = p["sex"]
    conclusion = (f"AI triage: {a['triage']['level']} priority. Top estimate: {a['probabilities'][0]['name']} "
                  f"({a['probabilities'][0]['probability']:.0%}); malignancy probability "
                  f"{a['triage']['malignancy_probability']:.0%}. Model {a['model']['arch']} v{a['model']['version']}"
                  f"{' (DEMO, untrained)' if a['model'].get('demo_mode') else ''}.")
    if case.get("clinical_impression"):
        conclusion += f" Clinician impression: {case['clinical_impression']}"
    report = {
        "resourceType": "DiagnosticReport", "id": rid, "status": status,
        "category": [{"text": "Dermatology"}],
        "code": {"text": "AI-assisted skin lesion assessment"},
        "subject": {"reference": f"Patient/{pid}"},
        "effectiveDateTime": case["created_at"],
        "issued": case.get("reviewed_at") or case["created_at"],
        "result": [{"reference": f"Observation/{o['id']}"} for o in obs],
        "conclusion": conclusion,
    }
    if p.get("localization"):
        report["code"]["text"] += f", body site: {p['localization']}"
    entries = [patient, report, *obs]
    if image_data_url:
        media = {"resourceType": "Media", "id": f"media-{case['id']}", "status": "completed",
                 "subject": {"reference": f"Patient/{pid}"},
                 "content": {"contentType": "image/jpeg", "data": image_data_url.split(",", 1)[1]}}
        report["media"] = [{"link": {"reference": f"Media/{media['id']}"}}]
        entries.append(media)
    return {"resourceType": "Bundle", "type": "collection", "timestamp": _now(),
            # Server-style fullUrls so the relative references ("Patient/...") resolve inside the bundle.
            "entry": [{"fullUrl": f"{FHIR_BASE}/{e['resourceType']}/{e['id']}", "resource": e} for e in entries]}
