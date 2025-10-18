from fastapi import FastAPI, UploadFile, File
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2, numpy as np, json, time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print('haar_loaded', not face_cascade.empty())

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000', 'http://127.0.0.1:3000', '*'],
    allow_methods=['*'],
    allow_headers=['*']
)

class Landmark(BaseModel):
    x: float
    y: float

class LandmarkPayload(BaseModel):
    landmarks: list[Landmark]
    width: int
    height: int

@app.get("/")
def health():
    return {"msg": "RatioMaxx ML service running"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # placeholder: later we’ll use DeepFace / OpenCV here
    contents = await file.read()
    size_kb = round(len(contents) / 1024, 2)
    return {"status": "ok", "file_size_kb": size_kb}

@app.post('/analyze/landmarks')
def analyze_landmarks(p: LandmarkPayload):
    required = [234, 454, 468, 473, 61, 291, 98, 327]
    n = len(p.landmarks)
    if n == 0 or max(required) >= n:
        raise HTTPException(status_code=400, detail=f"insufficient landmarks: got {n}, need ≥ {max(required)+1}")

    def dist(i, j):
        a, b = p.landmarks[i], p.landmarks[j]
        return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5

    face_width = dist(234, 454)
    eye_dist = dist(468, 473)
    mouth_w = dist(61, 291)
    nose_w = dist(98, 327)

    ratios = {
        'eye_face': eye_dist / face_width if face_width else 0.0,
        'mouth_face': mouth_w / face_width if face_width else 0.0,
        'nose_face': nose_w / face_width if face_width else 0.0
    }

    return { 'ratios': ratios }

@app.websocket('/ws/analyze')
async def ws_analyze(ws: WebSocket):
  await ws.accept()
  try:
    while True:
      data = await ws.receive_bytes()
      arr = np.frombuffer(data, np.uint8)
      img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

      t0 = time.time()
      h, w = img.shape[:2]

      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces_np = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48)
      )

      faces = [
        { 'x': int(x), 'y': int(y), 'w': int(wi), 'h': int(hi) }
        for (x, y, wi, hi) in faces_np
      ]

      payload = {
          'ts': time.time(),
          'shape': { 'w': w, 'h': h },
          'faces': faces,
          'facesCount': len(faces),
          'proc_ms': round((time.time() - t0) * 1000, 2)
      }

      await ws.send_text(json.dumps(payload))
  except WebSocketDisconnect:
    pass