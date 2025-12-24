
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime, timezone
import os

BEST_WEIGHTS = os.getenv("BEST_WEIGHTS", "runs/detect/train2/weights/best.pt")
CONF   = float(os.getenv("CONF", "0.30"))
IOU    = float(os.getenv("IOU", "0.65"))
IMG_SZ = int(os.getenv("IMG_SZ", "832"))
MAX_DET = int(os.getenv("MAX_DET", "500"))
AUGMENT = os.getenv("AUGMENT", "true").lower() == "true"

model = YOLO(BEST_WEIGHTS)

def infer(pil_img):
    results = model.predict(
        source=pil_img,
        conf=CONF, iou=IOU, imgsz=IMG_SZ,
        max_det=MAX_DET, augment=AUGMENT,
        verbose=False
    )

    r = results[0]
    names = r.names

    detections = []
    by_sku = defaultdict(int)

    for b in r.boxes:
        cid = int(b.cls.item())
        confv = float(b.conf.item())
        x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
        w, h = x2 - x1, y2 - y1

        sku = names.get(cid, f"class_{cid}")

        detections.append({
            "sku": sku,
            "class_id": cid,
            "confidence": round(confv, 4),
            "bbox_xyxy": [round(x1,2), round(y1,2), round(x2,2), round(y2,2)],
            "bbox_xywh": [round(x1,2), round(y1,2), round(w,2), round(h,2)]
        })
        by_sku[sku] += 1

    payload = {
        "datetime": datetime.now(timezone.utc).isoformat(),
        "model": os.path.basename(BEST_WEIGHTS),
        "imgsz": IMG_SZ,
        "conf_threshold": CONF,
        "detections": detections,
        "counts": {
            "total_detections": len(detections),
            "by_sku": by_sku
        }
    }

    boxed = r.plot()  # numpy BGR

    return payload, boxed
