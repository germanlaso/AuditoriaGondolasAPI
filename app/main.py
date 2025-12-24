
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io, os, uuid, base64
import numpy as np
from .yolo_runner import infer

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs_api")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Auditoria Gondolas API")

@app.get("/")
def root():
    return {"status": "ok", "message": "API Auditoría Góndolas funcionando"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    store_code: str = Form(None),
    room_code: str = Form(None),
    auditor: str = Form(None),
    return_image: bool = Form(False)
):
    content = await file.read()
    pil_img = Image.open(io.BytesIO(content)).convert("RGB")

    payload, boxed = infer(pil_img)

    guid = str(uuid.uuid4())
    raw_path = os.path.join(OUTPUT_DIR, f"{guid}_raw.jpg")
    ann_path = os.path.join(OUTPUT_DIR, f"{guid}_ann.jpg")

    with open(raw_path, "wb") as f:
        f.write(content)

    ann_rgb = boxed[:, :, ::-1]
    Image.fromarray(ann_rgb).save(ann_path, format="JPEG", quality=90)

    payload.update({
        "image_path": raw_path,
        "annotated_path": ann_path,
        "meta": {
            "store_code": store_code,
            "room_code": room_code,
            "auditor": auditor
        }
    })

    if return_image:
        with open(ann_path, "rb") as f:
            payload["annotated_base64"] = base64.b64encode(f.read()).decode("utf-8")

    return JSONResponse(payload)
