"""Download ISIC 2019 (25,331 dermoscopy images, 8 diagnoses) into DermaAI's CSV format.

The 9.8 GB archive is never written to disk: the zip is read remotely with
HTTP range requests and each image is resized on the fly (shorter side
``--size`` px), which needs only a few hundred MB. Re-running resumes.

    python scripts/prepare_isic2019.py      # writes data/raw/images/ + data/raw/labels.csv
    python training/train.py                # then train on it

ISIC 2019 combines HAM10000 (Vienna), BCN20000 (Barcelona) and MSK (New York).
Data licence: CC BY-NC 4.0, cite Tschandl 2018, Codella 2018 and Combalia 2019.
"""

from __future__ import annotations

import argparse
import csv
import io
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

BASE = "https://isic-challenge-data.s3.amazonaws.com/2019"
ZIP_URL = f"{BASE}/ISIC_2019_Training_Input.zip"
GT_URL = f"{BASE}/ISIC_2019_Training_GroundTruth.csv"
META_URL = f"{BASE}/ISIC_2019_Training_Metadata.csv"

LABELS = {"MEL": "mel", "NV": "nv", "BCC": "bcc", "AK": "akiec", "BKL": "bkl", "DF": "df", "VASC": "vasc", "SCC": "scc"}
# ISIC 2019 general anatomic sites -> DermaAI localisations (HAM10000 vocabulary).
SITES = {"anterior torso": "trunk", "posterior torso": "back", "lateral torso": "trunk", "head/neck": "face",
         "upper extremity": "upper extremity", "lower extremity": "lower extremity", "palms/soles": "acral",
         "oral/genital": "genital"}


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url: str, session: requests.Session):
        self.url, self.s, self.pos = url, session, 0
        r = session.head(url, timeout=60)
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + offset
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        for attempt in range(5):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
                r.raise_for_status()
                data = r.content
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


_local = threading.local()


def _zip() -> zipfile.ZipFile:
    if not hasattr(_local, "zf"):
        s = requests.Session()
        _local.zf = zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(ZIP_URL, s), buffer_size=2 << 20))
    return _local.zf


def _process(name: str, out_dir: Path, size: int) -> str:
    image_id = Path(name).stem.replace("_downsampled", "")
    dst = out_dir / f"{image_id}.jpg"
    if dst.exists():
        return "skip"
    img = Image.open(io.BytesIO(_zip().read(name))).convert("RGB")
    s = size / min(img.size)
    if s < 1:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    tmp = dst.with_suffix(".tmp")
    img.save(tmp, "JPEG", quality=92)
    tmp.rename(dst)
    return "ok"


def _csv(url: str) -> list[dict]:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return list(csv.DictReader(io.StringIO(r.text)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/raw")
    p.add_argument("--size", type=int, default=256, help="shorter side in pixels")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="only the first N images (for testing)")
    a = p.parse_args()
    out = Path(a.out)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    gt, meta = _csv(GT_URL), {m["image"]: m for m in _csv(META_URL)}
    rows = []
    for g in gt:
        dx = next((LABELS[k] for k in LABELS if float(g.get(k) or 0) == 1.0), None)
        if dx is None:  # UNK has no training images
            continue
        image_id = g["image"].replace("_downsampled", "")
        m = meta.get(g["image"]) or meta.get(image_id) or {}
        lesion = (m.get("lesion_id") or "").strip() or image_id
        source = "HAM10000" if lesion.startswith("HAM") else "BCN20000" if lesion.startswith("BCN") else "MSK"
        rows.append({"image": image_id, "label": dx, "lesion_id": lesion, "age": m.get("age_approx", ""),
                     "sex": m.get("sex", ""), "localization": SITES.get(m.get("anatom_site_general", ""), ""),
                     "source": source})
    if a.limit:
        rows = rows[:a.limit]
    wanted = {r["image"] for r in rows}

    names = [n for n in _zip().namelist() if n.lower().endswith(".jpg")
             and Path(n).stem.replace("_downsampled", "") in wanted]
    print(f"{len(rows)} labelled images, {len(names)} found in archive; writing to {img_dir}", flush=True)
    t0, done, errors = time.time(), 0, 0
    with ThreadPoolExecutor(a.workers) as pool:
        futures = {pool.submit(_process, n, img_dir, a.size): n for n in names}
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:  # keep going; a re-run retries failures
                errors += 1
                print(f"  failed {futures[f]}: {e}", flush=True)
            done += 1
            if done % 1000 == 0 or done == len(names):
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(names)} ({rate:.0f} img/s, ETA {(len(names) - done) / rate / 60:.1f} min)", flush=True)

    present = {f.stem for f in img_dir.glob("*.jpg")}
    rows = [r for r in rows if r["image"] in present]
    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    print(f"wrote {out / 'labels.csv'}: {len(rows)} images, {errors} errors, classes {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
