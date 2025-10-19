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
  ws.eye_poly = None
  ws.teeth_poly = None
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
      # handle JSON control messages (eye/teeth polygons) sent before frames
      if msg.get('text') is not None:
        try:
          j = json.loads(msg['text'])
          if isinstance(j, dict):
            if 'eyePoly' in j:
              ws.eye_poly = j['eyePoly']
            if 'teethPoly' in j:
              ws.teeth_poly = j['teethPoly']
        except Exception:
          pass
        continue

      if msg.get('bytes') is None:
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

        # map normalized frame polys into ROI pixel coords if available
        state['eye_poly'] = None
        state['teeth_poly'] = None
        if ws.eye_poly and fw > 0 and fh > 0:
          def map_poly(poly):
            out = []
            for p in poly:
              try:
                px = int(p.get('x', 0.0) * w) - fx
                py = int(p.get('y', 0.0) * h) - fy
                # clamp to ROI bounds
                px = max(0, min(px, fw - 1))
                py = max(0, min(py, fh - 1))
                out.append({'x': px, 'y': py})
              except Exception:
                continue
            return out
          state['eye_poly'] = {
            'left':  map_poly(ws.eye_poly.get('left', [])) if isinstance(ws.eye_poly, dict) else None,
            'right': map_poly(ws.eye_poly.get('right', [])) if isinstance(ws.eye_poly, dict) else None
          }
        if ws.teeth_poly and fw > 0 and fh > 0:
          def map_poly_list(poly):
            out = []
            for p in poly:
              try:
                px = int(p.get('x', 0.0) * w) - fx
                py = int(p.get('y', 0.0) * h) - fy
                px = max(0, min(px, fw - 1))
                py = max(0, min(py, fh - 1))
                out.append({'x': px, 'y': py})
              except Exception:
                continue
            return out
          state['teeth_poly'] = map_poly_list(ws.teeth_poly) if isinstance(ws.teeth_poly, list) else None

        skin_payload, state = skin.compute(roi, state)
    
      if isinstance(skin_payload, dict) and isinstance(skin_payload.get('metrics'), dict):
        metrics = skin_payload['metrics']
        for k, v in metrics.items():
          if isinstance(v, dict) and 'now' in v:
            try:
              now_val = float(v['now'])
            except Exception:
              continue
            # tune alpha/scale per metric if you want
            alpha = 0.2 if k != 'evenness' else 0.1
            scale = 8.0
            avg, conf = _upd_skin(ws, k, now_val, a=alpha, scale=scale)
            v['avg'] = round(avg, 1)
            v['conf'] = round(conf, 1)
        skin_payload['metrics'] = metrics
      else:
        skin_payload = None

      proc = proc_time_ms(t0)
      fps = round(1000.0 / max(1.0, proc), 1)

      # normalize faces into { bbox: {x,y,w,h}, conf? }
      faces_norm = []
      for b in faces:
        item = { 'bbox': { 'x': b['x'], 'y': b['y'], 'w': b['w'], 'h': b['h'] } }
        if 'conf' in b:
          try:
            item['conf'] = round(float(b['conf']), 1)
          except Exception:
            item['conf'] = b['conf']
        faces_norm.append(item)

      payload = {}
      if skin_payload is not None:
        payload['skin'] = skin_payload
      payload.update({
        'summary': {
          'ts': time.time(),
          'proc_ms': proc,
          'fps': fps
        },
        'video': { 'w': w, 'h': h },
        'faces': {
          'count': len(faces_norm),
          'items': faces_norm
        }
      })

      await ws.send_text(json.dumps(payload))
      frame_idx += 1
  except WebSocketDisconnect:
    pass