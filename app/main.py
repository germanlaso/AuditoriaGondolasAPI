
# main.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io, os, uuid, base64
import numpy as np
from typing import Optional, Dict, Any

from .yolo_runner import infer  # reutiliza tu función actual

# === Config ===
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs_api")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Auditoria Gondolas API")

# === Modelos para JSON/base64 ===
class PredictBase64Body(BaseModel):
    image_base64: str
    store_code: Optional[str] = None
    room_code: Optional[str] = None
    auditor: Optional[str] = None
    return_image: bool = True
    # Si después quieres usarlos, puedes pasarlos a infer():
    conf_threshold: Optional[float] = None
    imgsz: Optional[int] = None

# === Utilidad común: procesa bytes y arma respuesta estándar ===
def process_image_bytes(
    content: bytes,
    store_code: Optional[str],
    room_code: Optional[str],
    auditor: Optional[str],
    return_image: bool
) -> Dict[str, Any]:
    """
    Decodifica bytes → PIL, corre infer(pil), guarda raw y anotada,
    y devuelve el payload con image_path / annotated_path / meta / annotated_base64 (opcional).
    """
    # 1) Bytes → PIL RGB
    pil_img = Image.open(io.BytesIO(content)).convert("RGB")

    # 2) Inferencia (tu función: retorna payload y 'boxed')
    #    payload: dict con 'counts', 'detections', etc.
    #    boxed: imagen anotada como ndarray BGR o PIL.Image
    payload, boxed = infer(pil_img)

    # 3) Guardado de archivos
    guid = str(uuid.uuid4())
    raw_path = os.path.join(OUTPUT_DIR, f"{guid}_raw.jpg")
    ann_path = os.path.join(OUTPUT_DIR, f"{guid}_ann.jpg")

    with open(raw_path, "wb") as f:
        f.write(content)

    # Si boxed es ndarray (típico BGR), convertir a RGB
    if isinstance(boxed, np.ndarray):
        if boxed.ndim == 3 and boxed.shape[2] == 3:
            ann_rgb = boxed[:, :, ::-1]  # BGR → RGB
            Image.fromarray(ann_rgb).save(ann_path, format="JPEG", quality=90)
        else:
            Image.fromarray(boxed).save(ann_path, format="JPEG", quality=90)
    elif isinstance(boxed, Image.Image):
        boxed.convert("RGB").save(ann_path, format="JPEG", quality=90)
    else:
        # Si por alguna razón no recibimos imagen anotada, guardamos la original
        Image.open(io.BytesIO(content)).convert("RGB").save(ann_path, format="JPEG", quality=90)

    # 4) Completar payload estándar
    payload.update({
        "image_path": raw_path,
        "annotated_path": ann_path,
        "meta": {
            "store_code": store_code,
            "room_code": room_code,
            "auditor": auditor
        }
    })

    # 5) Evidencia embebida (base64) si se solicitó
    if return_image:
        with open(ann_path, "rb") as f:
            payload["annotated_base64"] = base64.b64encode(f.read()).decode("utf-8")

    return payload

# === Health ===
@app.get("/")
def root():
    return {"status": "ok", "message": "API Auditoría Góndolas funcionando"}

# === Endpoint existente (multipart/form-data) ===
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    store_code: str = Form(None),
    room_code: str = Form(None),
    auditor: str = Form(None),
    return_image: bool = Form(False)
):
    try:
        content = await file.read()
        payload = process_image_bytes(content, store_code, room_code, auditor, return_image)
        return JSONResponse(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# === Nuevo endpoint (JSON + base64) para Power Automate HTTP simple ===
@app.post("/predict_base64")
def predict_base64(body: PredictBase64Body):
    try:
        if not body.image_base64:
            raise ValueError("image_base64 vacío")

        # Decodificar base64 → bytes
        content = base64.b64decode(body.image_base64)

        # Reutilizar la misma lógica
        payload = process_image_bytes(content, body.store_code, body.room_code, body.auditor, body.return_image)

        # (Opcional) si luego integras conf/imgsz en infer(), aquí puedes pasarlos
        # por ahora se mantienen como en tu infer actual.

        return JSONResponse(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
``

