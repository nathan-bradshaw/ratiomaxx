from fastapi import FastAPI, UploadFile, File
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2, numpy as np, json, time
from collections import deque

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print('haar_loaded', not face_cascade.empty())

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000', 'http://127.0.0.1:3000', '*'],
    allow_methods=['*'],
    allow_headers=['*']
)

@app.get("/")
def health():
    return {"msg": "RatioMaxx ML service running"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # placeholder: later we’ll use DeepFace / OpenCV here
    contents = await file.read()
    size_kb = round(len(contents) / 1024, 2)
    return {"status": "ok", "file_size_kb": size_kb}

@app.websocket('/ws/stream')
async def ws_stream(ws: WebSocket):
  await ws.accept()
  # per socket EMA for stream derived signals
  if not hasattr(ws, 'sig_ema'):
    ws.sig_ema = {}
    ws.sig_jit = {}
  if not hasattr(ws, 'even_hist'):
    ws.even_hist = deque(maxlen=12)
  if not hasattr(ws, 'prev_mask'):
    ws.prev_mask = None
  def conf2(jit, scale=10.0):
    j = 0.0 if jit is None else float(jit)
    c = 100.0 - min(100.0, (j / max(1e-6, scale)) * 100.0)
    return max(0.0, min(100.0, c))
  # keep track of past boxes for smoothing
  prev_boxes = []  # list of dicts {x,y,w,h}
  stale = 0        # count how many frames with no detections
  frame_idx = 0    # simple frame counter for detection frequency

  trackers = []  # list of (tracker, id)

  def new_tracker():
    # try CSRT then KCF then MOSSE, compatible across OpenCV versions
    try:
      return cv2.legacy.TrackerCSRT_create()
    except Exception:
      try:
        return cv2.legacy.TrackerKCF_create()
      except Exception:
        try:
          return cv2.legacy.TrackerMOSSE_create()
        except Exception:
          return None

  # iou and smoothing helpers
  def iou(a, b):
    ax1, ay1, ax2, ay2 = a['x'], a['y'], a['x'] + a['w'], a['y'] + a['h']
    bx1, by1, bx2, by2 = b['x'], b['y'], b['x'] + b['w'], b['y'] + b['h']
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
      return 0.0
    area_a = a['w'] * a['h']
    area_b = b['w'] * b['h']
    return inter / float(area_a + area_b - inter)

  def smooth_box(prev, curr, alpha=0.6):
    # smooth transition to new box
    return {
      'x': int(prev['x'] * alpha + curr['x'] * (1 - alpha)),
      'y': int(prev['y'] * alpha + curr['y'] * (1 - alpha)),
      'w': int(prev['w'] * alpha + curr['w'] * (1 - alpha)),
      'h': int(prev['h'] * alpha + curr['h'] * (1 - alpha))
    }

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
      # normalize luminance each frame by estimating average brightness
      # apply gamma curve to balance washed-out bright scenes and crushed dark ones
      # ensures the detector sees consistent mid-tones across lighting conditions
      ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
      Y, Cr, Cb = cv2.split(ycrcb)
      meanY = float(np.mean(Y))
      target = 140.0
      gamma = float(np.clip(
        np.log(max(1e-6, target / 255.0)) / np.log(max(1e-6, meanY / 255.0)),
        0.6, 1.6
      ))
      img_gamma = np.clip((img.astype(np.float32) / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)

      # convert to YCrCb and apply CLAHE on Y (luma) only to boost local contrast
      # enhances edges like eyes, nose, jawline without amplifying noise
      # makes facial features pop for the cascade detector
      ycrcb2 = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2YCrCb)
      Y2, Cr2, Cb2 = cv2.split(ycrcb2)
      clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
      Y2c = clahe.apply(Y2)
      det_img = cv2.cvtColor(cv2.merge([Y2c, Cr2, Cb2]), cv2.COLOR_YCrCb2BGR)

      gray = cv2.cvtColor(det_img, cv2.COLOR_BGR2GRAY)
      frame = img  # trackers want original BGR images

      # helpers for skin metrics
      def ema(prev, x, alpha=0.2):
        return x if prev is None else (alpha * x + (1 - alpha) * prev)

      def to_lab(img_bgr):
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

      def to_hsv(img_bgr):
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

      def lap_var(gray):
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
      # run detection every 3rd frame; track in between
      run_detector = (frame_idx % 3 == 0)
      faces_np = []
      if run_detector:
        faces_np = face_cascade.detectMultiScale(
          gray,
          scaleFactor=1.05,
          minNeighbors=5,
          flags=cv2.CASCADE_SCALE_IMAGE,
          minSize=(max(48, w // 12), max(48, h // 12))
        )
        if len(faces_np) == 0:
          # fallback: lightly denoise and relax thresholds
          blur = cv2.bilateralFilter(gray, 5, 50, 50)
          faces_np = face_cascade.detectMultiScale(
            blur,
            scaleFactor=1.03,
            minNeighbors=3,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=(max(40, w // 16), max(40, h // 16))
          )
        # rebuild trackers from fresh detections
        trackers = []
        for (x, y, wi, hi) in faces_np:
          tr = new_tracker()
          if tr is None:
            continue
          ok = tr.init(frame, (int(x), int(y), int(wi), int(hi)))
          if ok:
            trackers.append(tr)
      else:
        # update trackers only
        updated = []
        for tr in trackers:
          ok, box = tr.update(frame)
          if ok:
            bx, by, bw, bh = box
            faces_np.append((int(bx), int(by), int(bw), int(bh)))
            updated.append(tr)
        trackers = updated

      # current detections as dicts
      curr = [{ 'x': int(x), 'y': int(y), 'w': int(wi), 'h': int(hi) } for (x, y, wi, hi) in faces_np]

      # reuse tracker boxes if detection skipped
      if not run_detector:
        faces = prev_boxes
      else:
        faces = []
        used_prev = [False] * len(prev_boxes)
        # match current boxes to previous ones and smooth
        for c in curr:
          best_j, best_iou = -1, 0.0
          for j, pbox in enumerate(prev_boxes):
            if used_prev[j]:
              continue
            jacc = iou(pbox, c)
            if jacc > best_iou:
              best_iou, best_j = jacc, j
          if best_j >= 0 and best_iou >= 0.2:
            faces.append(smooth_box(prev_boxes[best_j], c, alpha=0.65))
            used_prev[best_j] = True
          else:
            faces.append(c)

        # keep last boxes a bit if no detections
        if len(curr) == 0:
          stale += 1
          if stale <= 5:
            faces = prev_boxes
        else:
          stale = 0

        prev_boxes = faces

      frame_idx += 1

      def clamp_box(b):
        x = max(0, min(b['x'], w - 1))
        y = max(0, min(b['y'], h - 1))
        bw = max(1, min(b['w'], w - x))
        bh = max(1, min(b['h'], h - y))
        return {'x': int(x), 'y': int(y), 'w': int(bw), 'h': int(bh)}

      faces = [clamp_box(b) for b in faces]

      # skin metrics on primary face ROI if any
      skin_payload = None
      if len(faces):
        fx, fy, fw, fh = faces[0]['x'], faces[0]['y'], faces[0]['w'], faces[0]['h']
        roi = frame[max(0,fy):fy+fh, max(0,fx):fx+fw]
        if roi.size:
          # build a rough skin mask by excluding proportional eye/mouth regions
          mask = np.ones(roi.shape[:2], np.uint8) * 255
          # exclude eyes using two rectangles in the upper half
          eh = int(0.18 * fh); ew = int(0.28 * fw)
          ey = int(0.32 * fh)
          exL = int(0.18 * fw); exR = fw - exL - ew
          cv2.rectangle(mask, (exL, ey), (exL + ew, ey + eh), 0, -1)
          cv2.rectangle(mask, (exR, ey), (exR + ew, ey + eh), 0, -1)
          # exclude mouth region
          mw = int(0.5 * fw); mh = int(0.18 * fh)
          mx = int((fw - mw) / 2); my = int(0.68 * fh)
          cv2.rectangle(mask, (mx, my), (mx + mw, min(fh, my + mh)), 0, -1)

          # exclude hairline/forehead top band
          top_h = int(0.22 * fh)
          cv2.rectangle(mask, (0, 0), (fw, max(1, top_h)), 0, -1)

          # exclude brow band (slightly above eyes across most width)
          brow_y1 = int(0.24 * fh)
          brow_y2 = int(0.36 * fh)
          cv2.rectangle(mask, (int(0.08 * fw), brow_y1), (int(0.92 * fw), brow_y2), 0, -1)

          # exclude side margins where hair curls in (left/right 10%)
          side_w = int(0.10 * fw)
          cv2.rectangle(mask, (0, 0), (side_w, fh), 0, -1)
          cv2.rectangle(mask, (fw - side_w, 0), (fw, fh), 0, -1)

          # refine mask with a simple YCrCb skin threshold
          roi_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
          Yc, Cr, Cb = cv2.split(roi_ycrcb)
          color_mask = cv2.inRange(roi_ycrcb, (0, 135, 85), (255, 180, 135))  # classic skin range
          mask = cv2.bitwise_and(mask, color_mask)
          if ws.prev_mask is not None and isinstance(ws.prev_mask, np.ndarray) and ws.prev_mask.shape == mask.shape:
            pm = ws.prev_mask
            if pm.dtype != mask.dtype:
              pm = pm.astype(mask.dtype)
            mask = cv2.bitwise_or(mask, cv2.erode(pm, np.ones((3,3), np.uint8), 1))
          else:
            ws.prev_mask = None
          mask = cv2.medianBlur(mask, 5)
          ws.prev_mask = mask.copy()

          roi_lab = to_lab(roi)
          L, A, B = cv2.split(roi_lab)
          roi_hsv = to_hsv(roi)
          H, S, V = cv2.split(roi_hsv)

          m = mask > 0
          if np.any(m):
            Lv = L[m].astype(np.float32)
            Av = A[m].astype(np.float32)
            Vv = V[m].astype(np.float32)
            Sv = S[m].astype(np.float32)

            # clarity uses inverted Laplacian variance so smoother skin scores higher
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            clar_raw = lap_var(gray)
            clarity = max(0.0, min(100.0, 100.0 - np.interp(clar_raw, [50, 300], [0, 100])))

            # redness uses LAB a*, normalized to 0-100 and inverted so higher means less redness
            redness_raw = float(np.clip(np.interp(Av.mean() if Av.size else 128, [135, 160], [0, 100]), 0.0, 100.0))
            redness = float(100.0 - redness_raw)  # higher = better (less redness)

            # evenness: robust L* spread with light blur and rolling median
            if Lv.size:
              Lv_blur = cv2.GaussianBlur(L, (0,0), 1.0)[m].astype(np.float32)
              p_low = np.percentile(Lv_blur, 15)
              p_high = np.percentile(Lv_blur, 85)
              spread = float(p_high - p_low)
            else:
              spread = 0.0
            ws.even_hist.append(spread)
            spread_med = float(np.median(ws.even_hist))
            spread_c = float(np.clip(spread_med, 12.0, 60.0))
            evenness = float(100.0 - np.interp(spread_c, [12.0, 60.0], [0.0, 100.0]))

            # shine looks for bright low saturation areas, uses relaxed thresholds, reports 0-100 percent, and is inverted so higher is better
            shine_raw = float(np.clip((np.mean((Vv > 210) & (Sv < 60)) * 100.0) if Vv.size else 0.0, 0.0, 100.0))
            shine = float(100.0 - shine_raw)  # higher = better (less shine)

            # dark circles compare an under eye band to the cheek and invert so higher is better
            cy = int(0.42 * fh)
            band_h = max(1, int(0.05 * fh))
            x1, x2 = int(0.2 * fw), int(0.8 * fw)
            band = L[cy:cy+band_h, x1:x2]
            under = float(band.mean()) if band.size else 0.0
            cheek = float(L[int(0.6*fh):int(0.75*fh), x1:x2].mean() if L.size else 0.0)
            dark_raw = float(np.clip(np.interp(cheek - under, [0, 20], [0, 100]), 0.0, 100.0))
            dark_circles = float(100.0 - dark_raw)  # higher = better (lighter under-eyes)

            # blemish density uses small high frequency spots per area, normalized to 0-100 and inverted so higher is better
            blur = cv2.GaussianBlur(gray, (0,0), 1.2)
            hi = cv2.absdiff(gray, blur)
            spots = cv2.threshold(hi, 18, 255, cv2.THRESH_BINARY)[1]
            spots = cv2.bitwise_and(spots, mask)
            cnt = cv2.countNonZero(spots)
            blem_frac = float(np.clip(cnt / max(2000, (fw*fh*0.15)), 0.0, 1.0))
            blemishes = float((1.0 - blem_frac) * 100.0)

            # build confidences from EMA of absolute deltas per metric
            def upd(key, val, a=0.2):
              p = ws.sig_ema.get(key)
              avg = val if p is None else (a * val + (1 - a) * p)
              djp = ws.sig_jit.get(key)
              dj = abs((avg if p is None else avg) - (p if p is not None else val))
              dje = a * dj + (1 - a) * (djp if djp is not None else dj)
              ws.sig_ema[key] = avg
              ws.sig_jit[key] = dje
              return avg, conf2(dje, 8.0)

            c_a, c_a_conf = upd('clarity', clarity)
            c_r, c_r_conf = upd('redness', redness)
            c_e, c_e_conf = upd('evenness', evenness, a=0.1)
            c_s, c_s_conf = upd('shine', shine)
            c_d, c_d_conf = upd('dark', dark_circles)
            c_b, c_b_conf = upd('blem', blemishes)

            skin_payload = {
              'clarity': { 'now': round(clarity, 1), 'avg': round(c_a, 1) if c_a is not None else None, 'conf': round(c_a_conf, 1) },
              'redness': { 'now': round(redness, 1), 'avg': round(c_r, 1) if c_r is not None else None, 'conf': round(c_r_conf, 1) },
              'evenness': { 'now': round(evenness, 1), 'avg': round(c_e, 1) if c_e is not None else None, 'conf': round(c_e_conf, 1) },
              'shine': { 'now': round(shine, 1), 'avg': round(c_s, 1) if c_s is not None else None, 'conf': round(c_s_conf, 1) },
              'dark_circles': { 'now': round(dark_circles, 1), 'avg': round(c_d, 1) if c_d is not None else None, 'conf': round(c_d_conf, 1) },
              'blemishes': { 'now': round(blemishes, 1), 'avg': round(c_b, 1) if c_b is not None else None, 'conf': round(c_b_conf, 1) }
            }

            # lightweight extra signals
            # teeth whiteness uses bright low red pixels in a lower central mouth box
            tx1, tx2 = int(0.35*fw), int(0.65*fw)
            ty1, ty2 = int(0.62*fh), int(0.80*fh)
            mouth_roi = roi[ty1:ty2, tx1:tx2]
            teeth_score = None
            if mouth_roi.size:
              lab_m = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2LAB)
              Lm, Am, Bm = cv2.split(lab_m)
              # mask: high L*, not very red
              mask_t = (Lm > 180) & (Am < 140)
              if np.any(mask_t):
                whiten = float(np.clip(np.mean(Lm[mask_t]) / 255.0 * 100.0, 0.0, 100.0))
                teeth_score = whiten

            # eye sclera health uses low redness in eye rectangles
            exw, exh = int(0.28*fw), int(0.18*fh)
            ey = int(0.32*fh)
            exL, exR = int(0.18*fw), fw - int(0.18*fw) - exw
            sclera_score = None
            for (sx, sy) in [(exL, ey), (exR, ey)]:
              eroi = roi[sy:sy+exh, sx:sx+exw]
              if eroi.size:
                lab_e = cv2.cvtColor(eroi, cv2.COLOR_BGR2LAB)
                Le, Ae, Be = cv2.split(lab_e)
                # sclera uses bright and low saturation as a proxy
                mask_s = (Le > 160)
                if np.any(mask_s):
                  red = float(np.clip(np.mean(Ae[mask_s]), 120.0, 160.0))
                  score = float(np.interp(red, [160.0, 120.0], [0.0, 100.0]))  # less red means a higher score
                  sclera_score = score if sclera_score is None else (sclera_score + score) / 2.0

            signals = {}
            if teeth_score is not None:
              a, cf = upd('teeth_white', teeth_score)
              signals['teeth_whiteness'] = { 'now': round(teeth_score,1), 'avg': round(a,1) if a is not None else None, 'conf': round(cf,1) }
            if sclera_score is not None:
              a, cf = upd('sclera', sclera_score)
              signals['sclera_health'] = { 'now': round(sclera_score,1), 'avg': round(a,1) if a is not None else None, 'conf': round(cf,1) }

      payload = {
        'ts': time.time(),
        'shape': { 'w': w, 'h': h },
        'faces': faces,
        'facesCount': len(faces),
        'proc_ms': round((time.time() - t0) * 1000, 2)
      }
      if skin_payload is not None:
        payload['skin'] = skin_payload
      if 'signals' in locals() and signals:
        payload['signals'] = signals
      await ws.send_text(json.dumps(payload))
  except WebSocketDisconnect:
    pass

@app.websocket('/ws/analyze')
async def ws_analyze(ws: WebSocket):
  await ws.accept()
  # smoothing helpers for ratios
  def ema(prev, x, alpha=0.2):
    return x if prev is None else (alpha * x + (1 - alpha) * prev)

  # simple confidence from EMA of absolute deltas
  def conf(jit, scale=0.2):
    # smaller jitter means higher confidence 0-100
    j = 0.0 if jit is None else float(jit)
    c = 100.0 - min(100.0, (j / max(1e-6, scale)) * 100.0)
    return max(0.0, min(100.0, c))

  # per-socket EMA state
  if not hasattr(ws, 'r_ema'):
    ws.r_ema = {}
    ws.r_jit = {}
  try:
    while True:
      msg = await ws.receive()
      if msg.get('text') is None:
        await ws.send_text(json.dumps({'error': 'expected JSON landmarks'}))
        continue

      try:
        p = json.loads(msg['text'])
        lms = p.get('landmarks') or []
        using_z = any('z' in lm for lm in lms)
        video = p.get('video') or {}
      except Exception:
        await ws.send_text(json.dumps({'error': 'invalid json'}))
        continue

      if len(lms) < 474:
        await ws.send_text(json.dumps({'error': 'insufficient landmarks', 'count': len(lms)}))
        continue

      def d(i, j):
        ax, ay, az = lms[i].get('x', 0.0), lms[i].get('y', 0.0), lms[i].get('z', 0.0)
        bx, by, bz = lms[j].get('x', 0.0), lms[j].get('y', 0.0), lms[j].get('z', 0.0)
        dz = az - bz if (('z' in lms[i]) or ('z' in lms[j])) else 0.0
        return float(((ax - bx)**2 + (ay - by)**2 + dz**2) ** 0.5)

      face_w = d(234, 454) or 1e-9


      PHI = 1.6180339887
      import math
      def closeness(v, target):
        return math.exp(-abs(math.log(v / target)) * 3.5)

      face_len = d(10, 152)
      left_eye_w = d(33, 133)
      right_eye_w = d(362, 263)
      avg_eye_w = (left_eye_w + right_eye_w) / 2 or 1e-9
      inter_ocular = d(133, 362)
      mouth_w = d(61, 291)
      nose_w = d(49, 279) or 1e-9

      r_face = face_len / face_w
      r_eyes = inter_ocular / avg_eye_w
      r_mouth_nose = mouth_w / nose_w

      # lip fullness: lower vs upper lip height relative to mouth center
      mx = (lms[61]['x'] + lms[291]['x']) / 2.0
      my = (lms[61]['y'] + lms[291]['y']) / 2.0
      mz = ((lms[61].get('z', 0.0) + lms[291].get('z', 0.0)) / 2.0) if (('z' in lms[61]) or ('z' in lms[291])) else None
      def d_point(i, x, y, z=None):
        px, py, pz = lms[i].get('x', 0.0), lms[i].get('y', 0.0), lms[i].get('z', 0.0)
        if z is None:
          return float(((px - x)**2 + (py - y)**2) ** 0.5)
        return float(((px - x)**2 + (py - y)**2 + (pz - z)**2) ** 0.5)
      upper_vermilion = d_point(13, mx, my, mz)
      lower_vermilion = d_point(14, mx, my, mz)
      lip_fullness_ratio = lower_vermilion / (upper_vermilion or 1e-9)

      # chin to philtrum ratio: subnasale to upper lip vs lower lip to chin
      upper_len_phi = d(2, 13)
      lower_len_phi = d(14, 152)
      chin_to_philtrum = lower_len_phi / (upper_len_phi or 1e-9)

      s_face = closeness(r_face, PHI)
      s_eyes = closeness(r_eyes, PHI)
      s_mouth_nose = closeness(r_mouth_nose, PHI)
      s_lip_fullness = closeness(lip_fullness_ratio, PHI)
      s_chin_philtrum = closeness(chin_to_philtrum, 2.0)
      golden_score = round((s_face + s_eyes + s_mouth_nose + s_lip_fullness) / 4 * 100, 1)

      def angle_between(p1, p2, p3):
        a = np.array([p1.get('x', 0.0), p1.get('y', 0.0), p1.get('z', 0.0)])
        b = np.array([p2.get('x', 0.0), p2.get('y', 0.0), p2.get('z', 0.0)])
        c = np.array([p3.get('x', 0.0), p3.get('y', 0.0), p3.get('z', 0.0)])
        ba = a - b
        bc = c - b
        denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
        cosine = float(np.dot(ba, bc) / denom)
        cosine = float(np.clip(cosine, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

      chin = lms[152]; left_jaw = lms[234]; right_jaw = lms[454]
      angle_chin = angle_between(left_jaw, chin, right_jaw)
      angle_left_jaw = angle_between(chin, left_jaw, lms[10])
      angle_right_jaw = angle_between(chin, right_jaw, lms[338])
      jaw_angle_diff = abs(angle_left_jaw - angle_right_jaw)
      symmetry_score = max(0.0, 100 - (jaw_angle_diff / 180 * 100))

      jawline = {
        'angles': {
          'chin_angle': round(angle_chin, 2),
          'left_jaw_angle': round(angle_left_jaw, 2),
          'right_jaw_angle': round(angle_right_jaw, 2)
        },
        'symmetry_score': round(symmetry_score, 1)
      }

      # ideal facial ratios we can compute reliably with FaceMesh
      def midpt(i, j):
        return {
          'x': (lms[i].get('x', 0.0) + lms[j].get('x', 0.0)) / 2.0,
          'y': (lms[i].get('y', 0.0) + lms[j].get('y', 0.0)) / 2.0,
          'z': (lms[i].get('z', 0.0) + lms[j].get('z', 0.0)) / 2.0
        }

      def d_pts(pa, pb):
        ax, ay, az = pa.get('x', 0.0), pa.get('y', 0.0), pa.get('z', 0.0)
        bx, by, bz = pb.get('x', 0.0), pb.get('y', 0.0), pb.get('z', 0.0)
        dz = az - bz if (('z' in pa) or ('z' in pb)) else 0.0
        return float(((ax - bx)**2 + (ay - by)**2 + dz**2) ** 0.5)

      # bizygomatic width (cheekbone to cheekbone)
      bizy = d(234, 454)

      # FWHR: bizygomatic / eyebrow-middle to upper lip height
      brow_mid = midpt(105, 334)  # approx arch midpoints
      upper_lip = lms[13]
      brow_to_upper = d_pts(brow_mid, upper_lip) or 1e-9
      fwhr = bizy / brow_to_upper

      # interpupillary distance (approx): between eye centers
      left_eye_center = midpt(33, 133)
      right_eye_center = midpt(263, 362)
      ipd = d_pts(left_eye_center, right_eye_center)

      # midface ratio: ipd / (nasion to upper lip). Use landmark 168 for nasion; fallback to brow_mid
      nasion = lms[168] if 168 < len(lms) else brow_mid
      n_to_upper = d_pts(nasion, upper_lip) or 1e-9
      midface_ratio = ipd / n_to_upper

      # ESR: ipd / bizygomatic
      esr = ipd / (bizy or 1e-9)

      # EME angle: angle at mouth center with arms to eye centers
      mouth_center = midpt(13, 14)
      def angle_at(vertex, a, b):
        va = np.array([a.get('x',0.0)-vertex.get('x',0.0), a.get('y',0.0)-vertex.get('y',0.0), a.get('z',0.0)-vertex.get('z',0.0)])
        vb = np.array([b.get('x',0.0)-vertex.get('x',0.0), b.get('y',0.0)-vertex.get('y',0.0), b.get('z',0.0)-vertex.get('z',0.0)])
        denom = (np.linalg.norm(va)*np.linalg.norm(vb)) + 1e-9
        cosv = float(np.dot(va, vb) / denom)
        cosv = float(np.clip(cosv, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosv)))
      eme_angle = angle_at(mouth_center, left_eye_center, right_eye_center)

      # canthal tilt is roll compensated. zero degrees is horizontal and positive means the lateral side is higher
      # angles in image space use y down so convert both the segment angle and roll to a standard math space
      def seg_angle_deg(med, lat):
        dx = lms[lat].get('x',0.0) - lms[med].get('x',0.0)
        dy_img = lms[lat].get('y',0.0) - lms[med].get('y',0.0)
        ang_img = np.arctan2(dy_img, dx)         # image coords (y down)
        ang_math = -ang_img                      # flip to math coords (y up)
        return float(np.degrees(ang_math))

      # head roll estimated from the outer eye corners
      dx_eyes = lms[263].get('x',0.0) - lms[33].get('x',0.0)
      dy_eyes_img = lms[263].get('y',0.0) - lms[33].get('y',0.0)
      roll_img = np.arctan2(dy_eyes_img, dx_eyes)
      roll_math_deg = float(np.degrees(-roll_img))

      left_raw = seg_angle_deg(133, 33)     # medial→lateral (left eye)
      right_raw = seg_angle_deg(362, 263)   # medial→lateral (right eye)

      # remove roll and normalize to the range minus ninety to ninety
      def norm_deg(a):
        while a > 90: a -= 180
        while a < -90: a += 180
        return a

      left_ct = norm_deg(left_raw - roll_math_deg)
      right_ct = norm_deg(right_raw - roll_math_deg)
      canthal_tilt = (left_ct + right_ct) / 2.0

      # palpebral fissure length to height ratio per eye uses medial to lateral length over upper to lower height
      def fissure_ratio(med, lat, upper, lower):
        length = d(med, lat)
        height = d(upper, lower) or 1e-9
        return length / height
      # indices based on FaceMesh reference (approx mids for lids)
      pfr_left = fissure_ratio(133, 33, 159, 145)
      pfr_right = fissure_ratio(362, 263, 386, 374)
      pfr_avg = (pfr_left + pfr_right) / 2.0

      # collect into a section with targets when they apply
      ideals = {
        'cheekbone_width_over_brow_to_upperlip_height': round(fwhr, 3),
        'interpupillary_over_nasion_to_upperlip_height': round(midface_ratio, 3),
        'interpupillary_over_cheekbone_width': round(esr, 3),
        'eye_mouth_eye_angle_deg': round(eme_angle, 1),
        'canthal_tilt_angle_deg': round(canthal_tilt, 1),
        'eye_length_over_height': round(pfr_avg, 2),
        'mouth_width_over_nose_width': round(r_mouth_nose, 3),
        'lower_lip_height_over_upper_lip_height': round(lip_fullness_ratio, 3),
        'chin_height_over_philtrum_height': round(chin_to_philtrum, 3)
      }

      # build smoothed metrics and confidence without breaking existing fields
      raw_ideals = {
        'cheekbone_width_over_brow_to_upperlip_height': float(fwhr),
        'interpupillary_over_nasion_to_upperlip_height': float(midface_ratio),
        'interpupillary_over_cheekbone_width': float(esr),
        'eye_mouth_eye_angle_deg': float(eme_angle),
        'canthal_tilt_angle_deg': float(canthal_tilt),
        'eye_length_over_height': float(pfr_avg),
        'mouth_width_over_nose_width': float(r_mouth_nose),
        'lower_lip_height_over_upper_lip_height': float(lip_fullness_ratio),
        'chin_height_over_philtrum_height': float(chin_to_philtrum)
      }

      ideals_smooth = {}
      for k, v in raw_ideals.items():
        prev = ws.r_ema.get(k)
        avg = ema(prev, v)
        # jitter EMA
        dj_prev = ws.r_jit.get(k)
        dj = abs((avg if prev is None else avg) - (prev if prev is not None else v))
        dj_ema = ema(dj_prev, dj, 0.3)
        ws.r_ema[k] = avg
        ws.r_jit[k] = dj_ema
        # choose a scale per-metric (rough, keeps conf readable)
        scale = 0.15 if k in (
          'cheekbone_width_over_brow_to_upperlip_height',
          'interpupillary_over_nasion_to_upperlip_height',
          'interpupillary_over_cheekbone_width',
          'eye_length_over_height',
          'lower_lip_height_over_upper_lip_height'
        ) else 5.0
        ideals_smooth[k] = {
          'now': round(v, 3),
          'avg': round(avg, 3) if avg is not None else None,
          'conf': round(conf(dj_ema, scale), 1)
        }

      # extended symmetry score 0-100 mirrors left landmarks across the midline and compares to the right
      x_mid = (left_eye_center['x'] + right_eye_center['x']) / 2.0
      pairs = [
        (33, 263),   # outer eye corners
        (133, 362),  # inner eye corners
        (61, 291),   # mouth corners
        (234, 454),  # jaw corners
        (70, 300),   # brow heads
        (105, 334),  # brow tails
        (49, 279)    # nose alae
      ]
      errs = []
      for Li, Ri in pairs:
        Lx, Ly = lms[Li]['x'], lms[Li]['y']
        Rx, Ry = lms[Ri]['x'], lms[Ri]['y']
        Lx_mirror = 2.0 * x_mid - Lx
        dx = (Lx_mirror - Rx)
        dy = (Ly - Ry)
        errs.append((dx*dx + dy*dy) ** 0.5)
      err = float(np.mean(errs)) if errs else 0.0
      err_norm = err / (face_w or 1e-6)
      symmetry_ext = max(0.0, 100.0 - min(100.0, err_norm * 400.0))
      # smooth symmetry
      prev_sym = ws.r_ema.get('sym_ext')
      sym_avg = ema(prev_sym, symmetry_ext)
      dj_prev = ws.r_jit.get('sym_ext')
      dj = abs((sym_avg if prev_sym is None else sym_avg) - (prev_sym if prev_sym is not None else symmetry_ext))
      dj_ema = ema(dj_prev, dj, 0.3)
      ws.r_ema['sym_ext'] = sym_avg
      ws.r_jit['sym_ext'] = dj_ema
      symmetry_block = { 'now': round(symmetry_ext,1), 'avg': round(sym_avg,1) if sym_avg is not None else None, 'conf': round(conf(dj_ema, 5.0),1) }

      # face shape classifier uses simple rules
      # features used by the classifier
      jaw_deg = angle_chin
      r_fw = fwhr
      jaw_diff = abs(angle_left_jaw - angle_right_jaw)
      # simple heuristics
      label = 'oval'
      if r_fw >= 1.95 and jaw_deg <= 128:
        label = 'square'
      elif r_fw < 1.75 and jaw_deg >= 132:
        label = 'round'
      elif r_fw >= 1.85 and jaw_deg >= 132:
        label = 'heart'
      face_shape = {
        'label': label,
        'features': {
          'cheekbone_width_over_brow_to_upperlip_height': round(r_fw,3),
          'chin_angle': round(jaw_deg,1),
          'jaw_sym_delta': round(jaw_diff,1)
        }
      }

      payload = {
        'golden': {
          'score_pct': golden_score,
          'ratios': {
            'face_length_to_width': round(r_face, 3),
            'ipd_to_eye_width': round(r_eyes, 3),
            'mouth_to_nose_width': round(r_mouth_nose, 3),
            'lower_to_upper_lip_height': round(lip_fullness_ratio, 3),
          },
        },
        'jaw': jawline,
        'ideals': ideals,
        'ideals_smooth': ideals_smooth,
        'symmetry_ext': symmetry_block,
        'face_shape': face_shape,
        'video': video if video else None,
        'usingZ': using_z
      }

      await ws.send_text(json.dumps(payload))
  except WebSocketDisconnect:
    pass