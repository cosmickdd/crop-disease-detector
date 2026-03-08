"""Quick smoke test — hits all API endpoints including /detect-disease."""
import json
import sys
from pathlib import Path

import requests

BASE = "http://localhost:8000"

def check(label, url, method="GET", **kwargs):
    try:
        r = requests.request(method, url, **kwargs)
        print(f"[{r.status_code}] {label}")
        return r
    except Exception as e:
        print(f"[ERR] {label}: {e}")
        return None

# 1. Root
check("GET /", f"{BASE}/")

# 2. Health
r = check("GET /api/v1/health", f"{BASE}/api/v1/health")
if r:
    d = r.json()
    print(f"       model_loaded={d.get('model_loaded')}  device={d.get('device')}  arch={d.get('arch')}")

# 3. Classes
r = check("GET /api/v1/classes", f"{BASE}/api/v1/classes")
if r:
    d = r.json()
    print(f"       num_classes={d.get('num_classes')}")

# 4. Disease info
r = check("GET /api/v1/disease/Tomato_Late_blight", f"{BASE}/api/v1/disease/Tomato_Late_blight")
if r and r.status_code == 200:
    d = r.json()
    print(f"       crop={d.get('crop')}  disease={d.get('disease')}")

# 5. Single-image inference
test_dir = Path(__file__).resolve().parents[1] / "data" / "dataset" / "test"
img_path = next(test_dir.rglob("*.JPG"), None) or next(test_dir.rglob("*.jpg"), None)
if img_path:
    with open(img_path, "rb") as f:
        r = check(
            f"POST /api/v1/detect-disease ({img_path.parent.name})",
            f"{BASE}/api/v1/detect-disease",
            method="POST",
            files={"image": (img_path.name, f, "image/jpeg")},
        )
    if r and r.status_code == 200:
        d = r.json()
        print(f"       crop={d['crop']}  disease={d['disease']}")
        print(f"       confidence={d['confidence']:.2%}  severity={d['severity']}")
        print(f"       inference_time_ms={d['inference_time_ms']:.1f}ms")
        top3_fmt = [(t['class_name'].split('___')[-1], "{:.2%}".format(t['confidence'])) for t in d['top3_predictions']]
        print(f"       top3={top3_fmt}")
else:
    print("[SKIP] No test images found")

# 6. Batch inference
print("\n── Batch inference ──")
batch_imgs = sorted(test_dir.rglob("*.JPG"))[:3]
if len(batch_imgs) < 2:
    batch_imgs = sorted(test_dir.rglob("*.jpg"))[:3]
if len(batch_imgs) >= 2:
    files = [
        ("images", (p.name, open(p, "rb"), "image/jpeg"))
        for p in batch_imgs
    ]
    r = check(
        f"POST /api/v1/batch-detect ({len(batch_imgs)} images)",
        f"{BASE}/api/v1/batch-detect",
        method="POST",
        files=files,
    )
    if r and r.status_code == 200:
        d = r.json()
        print(f"       total={d['total_images']}  failed={d['failed_images']}  "
              f"time={d['total_time_ms']:.0f}ms")
        for i, res in enumerate(d.get('results', [])):
            print(f"       [{i+1}] {res['crop']} — {res['disease']} "
                  f"({res['confidence']:.0%})  [{res['backend']}]")
else:
    print("[SKIP] Need ≥ 2 test images for batch test")

print("\nDone.")
