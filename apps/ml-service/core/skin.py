import cv2, numpy as np
from collections import deque

def _ema(prev, x, a=0.2):
  return x if prev is None else (a * x + (1 - a) * prev)

def _conf(jit, scale=8.0):
  j = 0.0 if jit is None else float(jit)
  c = 100.0 - min(100.0, (j / max(1e-6, scale)) * 100.0)
  return max(0.0, min(100.0, c))

class SkinAnalyzer:
  def __init__(self):
    pass

  def compute(self, roi, state):
    if state.get('even_hist') is None:
      state['even_hist'] = deque(maxlen=12)
    if state.get('prev_mask', None) is None:
      state['prev_mask'] = None

    fh, fw = roi.shape[:2]
    mask = np.ones(roi.shape[:2], np.uint8) * 255

    eh = int(0.18 * fh); ew = int(0.28 * fw)
    ey = int(0.32 * fh)
    exL = int(0.18 * fw); exR = fw - exL - ew
    cv2.rectangle(mask, (exL, ey), (exL + ew, ey + eh), 0, -1)
    cv2.rectangle(mask, (exR, ey), (exR + ew, ey + eh), 0, -1)

    mw = int(0.5 * fw); mh = int(0.18 * fh)
    mx = int((fw - mw) / 2); my = int(0.68 * fh)
    cv2.rectangle(mask, (mx, my), (mx + mw, min(fh, my + mh)), 0, -1)

    top_h = int(0.22 * fh)
    cv2.rectangle(mask, (0, 0), (fw, max(1, top_h)), 0, -1)

    brow_y1 = int(0.24 * fh)
    brow_y2 = int(0.36 * fh)
    cv2.rectangle(mask, (int(0.08 * fw), brow_y1), (int(0.92 * fw), brow_y2), 0, -1)

    side_w = int(0.10 * fw)
    cv2.rectangle(mask, (0, 0), (side_w, fh), 0, -1)
    cv2.rectangle(mask, (fw - side_w, 0), (fw, fh), 0, -1)

    roi_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    color_mask = cv2.inRange(roi_ycrcb, (0, 135, 85), (255, 180, 135))
    mask = cv2.bitwise_and(mask, color_mask)
    if state['prev_mask'] is not None and isinstance(state['prev_mask'], np.ndarray) and state['prev_mask'].shape == mask.shape:
      pm = state['prev_mask']
      if pm.dtype != mask.dtype:
        pm = pm.astype(mask.dtype)
      mask = cv2.bitwise_or(mask, cv2.erode(pm, np.ones((3,3), np.uint8), 1))
    else:
      state['prev_mask'] = None
    mask = cv2.medianBlur(mask, 5)
    state['prev_mask'] = mask.copy()

    roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(roi_lab)
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(roi_hsv)
    m = mask > 0

    skin_payload = None
    if np.any(m):
      Lv = L[m].astype(np.float32)
      Av = A[m].astype(np.float32)
      Vv = V[m].astype(np.float32)
      Sv = S[m].astype(np.float32)

      gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
      clar_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
      clarity = max(0.0, min(100.0, 100.0 - np.interp(clar_raw, [50, 300], [0, 100])))

      redness_raw = float(np.clip(np.interp(Av.mean() if Av.size else 128, [135, 160], [0, 100]), 0.0, 100.0))
      redness = float(100.0 - redness_raw)

      if Lv.size:
        Lv_blur = cv2.GaussianBlur(L, (0,0), 1.0)[m].astype(np.float32)
        p_low = np.percentile(Lv_blur, 15)
        p_high = np.percentile(Lv_blur, 85)
        spread = float(p_high - p_low)
      else:
        spread = 0.0
      state['even_hist'].append(spread)
      spread_med = float(np.median(state['even_hist']))
      spread_c = float(np.clip(spread_med, 12.0, 60.0))
      evenness = float(100.0 - np.interp(spread_c, [12.0, 60.0], [0.0, 100.0]))

      shine_raw = float(np.clip((np.mean((Vv > 210) & (Sv < 60)) * 100.0) if Vv.size else 0.0, 0.0, 100.0))
      shine = float(100.0 - shine_raw)

      cy = int(0.42 * fh)
      band_h = max(1, int(0.05 * fh))
      x1, x2 = int(0.2 * fw), int(0.8 * fw)
      band = L[cy:cy+band_h, x1:x2]
      under = float(band.mean()) if band.size else 0.0
      cheek = float(L[int(0.6*fh):int(0.75*fh), x1:x2].mean() if L.size else 0.0)
      dark_raw = float(np.clip(np.interp(cheek - under, [0, 20], [0, 100]), 0.0, 100.0))
      dark_circles = float(100.0 - dark_raw)

      blur = cv2.GaussianBlur(gray, (0,0), 1.2)
      hi = cv2.absdiff(gray, blur)
      spots = cv2.threshold(hi, 18, 255, cv2.THRESH_BINARY)[1]
      spots = cv2.bitwise_and(spots, mask)
      cnt = cv2.countNonZero(spots)
      blem_frac = float(np.clip(cnt / max(2000, (fw*fh*0.15)), 0.0, 1.0))
      blemishes = float((1.0 - blem_frac) * 100.0)

      # simple per-socket EMA/conf state lives in class on caller, so compute raw here
      skin_payload = {
        'clarity': { 'now': round(clarity, 1), 'avg': None, 'conf': None },
        'redness': { 'now': round(redness, 1), 'avg': None, 'conf': None },
        'evenness': { 'now': round(evenness, 1), 'avg': None, 'conf': None },
        'shine': { 'now': round(shine, 1), 'avg': None, 'conf': None },
        'dark_circles': { 'now': round(dark_circles, 1), 'avg': None, 'conf': None },
        'blemishes': { 'now': round(blemishes, 1), 'avg': None, 'conf': None }
      }

    return skin_payload, state