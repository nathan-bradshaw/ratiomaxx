from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json, time, cv2, numpy as np
from core.detector import FaceDetector
from core.skin import SkinAnalyzer
from core.sticky import StickyBoxes
from core.utils import proc_time_ms

router = APIRouter()
fd = FaceDetector()
skin = SkinAnalyzer()
sticky = StickyBoxes()

def _upd_skin(ws, key, val, a=0.2, scale=8.0):
  prev = ws.skin_ema.get(key)
  avg = val if prev is None else (a * val + (1 - a) * prev)
  # EMA of per-frame delta -> jitter measure
  prev_j = ws.skin_jit.get(key)
  dj = abs((avg if prev is None else avg) - (prev if prev is not None else val))
  jit = (a * dj + (1 - a) * (prev_j if prev_j is not None else dj))
  ws.skin_ema[key] = avg
  ws.skin_jit[key] = jit
  # map jitter to 0–100 confidence (less jitter -> higher conf)
  conf = 100.0 - min(100.0, (jit / max(1e-6, scale)) * 100.0)
  return avg, max(0.0, min(100.0, conf))

@router.websocket('/ws/stream')
async def ws_stream(ws: WebSocket):
  await ws.accept()
  # per-socket EMA state for skin metrics
  if not hasattr(ws, 'skin_ema'):
    ws.skin_ema = {}
    ws.skin_jit = {}
  state = {'even_hist': None, 'prev_mask': None}
  trackers = []
  frame_idx = 0
  try:
    while True:
      msg = await ws.receive()
      if msg.get('bytes') is None:
        await ws.send_text(json.dumps({'error': 'expected binary frame data'}))
        continue

      t0 = time.time()
      arr = np.frombuffer(msg['bytes'], np.uint8)
      img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
      h, w = img.shape[:2]

      det_img, gray = fd.preprocess(img)
      run_detector = (frame_idx % 3 == 0)
      faces_np, trackers = fd.detect_or_track(det_img, gray, img, run_detector, trackers)

      curr = [{ 'x': int(x), 'y': int(y), 'w': int(wi), 'h': int(hi) } for (x,y,wi,hi) in faces_np]
      faces = sticky.update(curr, img.shape[:2])
      faces = [fd.clamp_box(b, w, h) for b in faces]

      skin_payload = None
      if faces:
        fx, fy, fw, fh = faces[0]['x'], faces[0]['y'], faces[0]['w'], faces[0]['h']
        roi = img[max(0,fy):fy+fh, max(0,fx):fx+fw]
        skin_payload, state = skin.compute(roi, state)
    
      if skin_payload is not None:
        for k, v in skin_payload.items():
            now_val = float(v['now'])
            # tune alpha/scale per metric if you want
            alpha = 0.2 if k != 'evenness' else 0.1
            scale = 8.0
            avg, conf = _upd_skin(ws, k, now_val, a=alpha, scale=scale)
            v['avg'] = round(avg, 1)
            v['conf'] = round(conf, 1)

      payload = {
        'ts': time.time(),
        'shape': { 'w': w, 'h': h },
        'faces': faces,
        'facesCount': len(faces),
        'proc_ms': proc_time_ms(t0)
      }
      if skin_payload is not None:
        payload['skin'] = skin_payload
      await ws.send_text(json.dumps(payload))
      frame_idx += 1
  except WebSocketDisconnect:
    pass