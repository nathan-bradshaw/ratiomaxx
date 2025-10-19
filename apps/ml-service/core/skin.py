import cv2, numpy as np
from collections import deque

def _poly_mask(poly_pts, roi_w, roi_h):
  if not poly_pts or len(poly_pts) < 3:
    return np.zeros((roi_h, roi_w), np.uint8)
  pts = np.array([[int(p['x']), int(p['y'])] for p in poly_pts], np.int32)
  m = np.zeros((roi_h, roi_w), np.uint8)
  cv2.fillPoly(m, [pts], 255)
  return m

def _mean_in_mask(channel, mask):
  if mask is None:
    return None
  m = mask > 0
  if not np.any(m):
    return None
  return float(channel[m].astype(np.float32).mean())

def _ema(prev, x, a=0.2):
  return x if prev is None else (a * x + (1 - a) * prev)

def _conf(jit, scale=8.0):
  j = 0.0 if jit is None else float(jit)
  c = 100.0 - min(100.0, (j / max(1e-6, scale)) * 100.0)
  return max(0.0, min(100.0, c))

# -- helpers --
def build_skin_mask(roi, state):
  fh, fw = roi.shape[:2]
  mask = np.ones((fh, fw), np.uint8) * 255
  # block eyes
  eh = int(0.18 * fh); ew = int(0.28 * fw)
  ey = int(0.32 * fh)
  exL = int(0.18 * fw); exR = fw - exL - ew
  cv2.rectangle(mask, (exL, ey), (exL + ew, ey + eh), 0, -1)
  cv2.rectangle(mask, (exR, ey), (exR + ew, ey + eh), 0, -1)
  # block mouth
  mw = int(0.5 * fw); mh = int(0.18 * fh)
  mx = int((fw - mw) / 2); my = int(0.68 * fh)
  cv2.rectangle(mask, (mx, my), (mx + mw, min(fh, my + mh)), 0, -1)
  # block upper forehead band
  top_h = int(0.22 * fh)
  cv2.rectangle(mask, (0, 0), (fw, max(1, top_h)), 0, -1)
  # block brow band
  brow_y1 = int(0.24 * fh); brow_y2 = int(0.36 * fh)
  cv2.rectangle(mask, (int(0.08 * fw), brow_y1), (int(0.92 * fw), brow_y2), 0, -1)
  # block sides
  side_w = int(0.10 * fw)
  cv2.rectangle(mask, (0, 0), (side_w, fh), 0, -1)
  cv2.rectangle(mask, (fw - side_w, 0), (fw, fh), 0, -1)
  # skin color gate
  roi_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
  color_mask = cv2.inRange(roi_ycrcb, (0, 135, 85), (255, 180, 135))
  mask = cv2.bitwise_and(mask, color_mask)
  # temporal stickiness
  pm = state.get('prev_mask')
  if isinstance(pm, np.ndarray) and pm.shape == mask.shape:
    if pm.dtype != mask.dtype:
      pm = pm.astype(mask.dtype)
    mask = cv2.bitwise_or(mask, cv2.erode(pm, np.ones((3,3), np.uint8), 1))
  mask = cv2.medianBlur(mask, 5)
  state['prev_mask'] = mask.copy()
  return mask

def split_channels(roi):
  roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
  L, A, B = cv2.split(roi_lab)
  roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
  H, S, V = cv2.split(roi_hsv)
  return L, A, B, H, S, V

# eye whiteness using contrast inside combined eye mask
def eye_whiteness_from_polys(L, eye_poly, fw, fh, state):
  if not eye_poly or not isinstance(eye_poly, dict):
    return state.get('eye_last')
  mL = _poly_mask(eye_poly.get('left', []),  fw, fh)
  mR = _poly_mask(eye_poly.get('right', []), fw, fh)
  mEyes = None
  if mL is not None and mR is not None:
    mEyes = cv2.bitwise_or(mL, mR)
  elif mL is not None:
    mEyes = mL
  elif mR is not None:
    mEyes = mR
  if mEyes is None or cv2.countNonZero(mEyes) == 0:
    return state.get('eye_last')
  eye_L = cv2.bitwise_and(L, L, mask=mEyes)
  vals = eye_L[eye_L > 0].astype(np.float32)
  if not vals.size:
    return state.get('eye_last')
  p10 = float(np.percentile(vals, 10))
  p90 = float(np.percentile(vals, 90))
  contrast = max(0.0, p90 - p10)
  val = float(np.clip(np.interp(contrast, [12.0, 70.0], [0.0, 100.0]), 0.0, 100.0))
  prev = state.get('eye_last')
  smoothed = _ema(prev, val, a=0.25)
  if prev is not None:
    max_step = 6.0
    smoothed = prev + max(-max_step, min(max_step, smoothed - prev))
  state['eye_last'] = smoothed
  return smoothed

# teeth whiteness from inner-lip polygon
def teeth_whiteness_from_poly(L, A, teeth_poly, fw, fh):
  if not teeth_poly or not isinstance(teeth_poly, list):
    return None
  mT = _poly_mask(teeth_poly, fw, fh)
  mm = mT > 0
  if not np.any(mm):
    return None
  Lv = L[mm].astype(np.float32)
  Av = A[mm].astype(np.float32)
  keep = Av <= 140
  if keep.size and keep.any():
    Lv = Lv[keep]
  if not Lv.size:
    return None
  return float(np.clip(np.interp(Lv.mean(), [120, 215], [0.0, 100.0]), 0.0, 100.0))

# core metric calculators
def calc_clarity(gray):
  clar_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
  return max(0.0, min(100.0, 100.0 - np.interp(clar_raw, [50, 300], [0, 100])))

def calc_redness(A):
  if A.size:
    redness_raw = float(np.clip(np.interp(A.mean(), [135, 160], [0, 100]), 0.0, 100.0))
  else:
    redness_raw = 50.0
  return float(100.0 - redness_raw)

def calc_evenness(L, m, state):
  if L.size and np.any(m):
    Lv_blur = cv2.GaussianBlur(L, (0,0), 1.0)[m].astype(np.float32)
    p_low = np.percentile(Lv_blur, 15)
    p_high = np.percentile(Lv_blur, 85)
    spread = float(p_high - p_low)
  else:
    spread = 0.0
  state['even_hist'].append(spread)
  spread_med = float(np.median(state['even_hist']))
  spread_c = float(np.clip(spread_med, 12.0, 60.0))
  return float(100.0 - np.interp(spread_c, [12.0, 60.0], [0.0, 100.0]))

def calc_shine(V, S, m):
  if V.size and S.size and np.any(m):
    shine_raw = float(np.clip((np.mean((V[m] > 210) & (S[m] < 60)) * 100.0), 0.0, 100.0))
  else:
    shine_raw = 0.0
  return float(100.0 - shine_raw)

def calc_dark_circles(L, fw, fh):
  cy = int(0.42 * fh)
  band_h = max(1, int(0.05 * fh))
  x1, x2 = int(0.2 * fw), int(0.8 * fw)
  band = L[cy:cy+band_h, x1:x2]
  under = float(band.mean()) if band.size else 0.0
  cheek = float(L[int(0.6*fh):int(0.75*fh), x1:x2].mean() if L.size else 0.0)
  dark_raw = float(np.clip(np.interp(cheek - under, [0, 20], [0, 100]), 0.0, 100.0))
  return float(100.0 - dark_raw)

def calc_blemishes(gray, mask, fw, fh):
  blur = cv2.GaussianBlur(gray, (0,0), 1.2)
  hi = cv2.absdiff(gray, blur)
  spots = cv2.threshold(hi, 18, 255, cv2.THRESH_BINARY)[1]
  spots = cv2.bitwise_and(spots, mask)
  cnt = cv2.countNonZero(spots)
  blem_frac = float(np.clip(cnt / max(2000, (fw*fh*0.15)), 0.0, 1.0))
  return float((1.0 - blem_frac) * 100.0)

def summarize_skin(skin_payload):
  core_for_score = ['clarity','redness','evenness','shine','blemishes']
  core_vals = [skin_payload[k]['now'] for k in core_for_score if isinstance(skin_payload.get(k), dict) and 'now' in skin_payload[k]]
  quality_score = round(float(np.mean(core_vals)), 1) if core_vals else None
  return { 'summary': { 'quality_score_pct': quality_score }, 'metrics': skin_payload }

class SkinAnalyzer:
  def __init__(self):
    pass

  def compute(self, roi, state):
    if state.get('even_hist') is None:
      state['even_hist'] = deque(maxlen=12)
    if state.get('prev_mask', None) is None:
      state['prev_mask'] = None
    if state.get('eye_last', None) is None:
      state['eye_last'] = None
      state['eye_fail'] = 0

    fh, fw = roi.shape[:2]
    mask = build_skin_mask(roi, state)

    L, A, B, H, S, V = split_channels(roi)

    eye_whiteness = eye_whiteness_from_polys(L, state.get('eye_poly'), fw, fh, state)

    teeth_whiteness = teeth_whiteness_from_poly(L, A, state.get('teeth_poly'), fw, fh)

    if np.any(m := (mask > 0)):
      gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
      clarity = calc_clarity(gray)
      redness = calc_redness(A[m].astype(np.float32) if A.size else A)
      evenness = calc_evenness(L, m, state)
      shine = calc_shine(V, S, m)
      dark_circles = calc_dark_circles(L, fw, fh)
      blemishes = calc_blemishes(gray, mask, fw, fh)
      skin_payload = {
        'clarity': { 'now': round(clarity, 1), 'avg': None, 'conf': None },
        'redness': { 'now': round(redness, 1), 'avg': None, 'conf': None },
        'evenness': { 'now': round(evenness, 1), 'avg': None, 'conf': None },
        'shine': { 'now': round(shine, 1), 'avg': None, 'conf': None },
        'dark_circles': { 'now': round(dark_circles, 1), 'avg': None, 'conf': None },
        'blemishes': { 'now': round(blemishes, 1), 'avg': None, 'conf': None }
      }
      if eye_whiteness is not None:
        skin_payload['eye_whiteness'] = { 'now': round(eye_whiteness, 1), 'avg': None, 'conf': None }
      if teeth_whiteness is not None:
        skin_payload['teeth_whiteness'] = { 'now': round(teeth_whiteness, 1), 'avg': None, 'conf': None }
    else:
      skin_payload = {}
      if eye_whiteness is not None:
        skin_payload['eye_whiteness'] = { 'now': round(eye_whiteness, 1), 'avg': None, 'conf': None }
      if teeth_whiteness is not None:
        skin_payload['teeth_whiteness'] = { 'now': round(teeth_whiteness, 1), 'avg': None, 'conf': None }

    result = summarize_skin(skin_payload)
    return result, state