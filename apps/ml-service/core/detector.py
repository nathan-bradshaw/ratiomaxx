import cv2, numpy as np

class FaceDetector:
  def __init__(self):
    self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

  def preprocess(self, img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    meanY = float(np.mean(Y))
    target = 140.0
    num = np.log(max(1e-6, target / 255.0))
    den = np.log(max(1e-6, meanY / 255.0))
    gamma = float(np.clip(num / den if den != 0 else 1.0, 0.6, 1.6))
    img_gamma = np.clip((img.astype(np.float32) / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)

    ycrcb2 = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2YCrCb)
    Y2, Cr2, Cb2 = cv2.split(ycrcb2)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    Y2c = clahe.apply(Y2)
    det_img = cv2.cvtColor(cv2.merge([Y2c, Cr2, Cb2]), cv2.COLOR_YCrCb2BGR)

    gray = cv2.cvtColor(det_img, cv2.COLOR_BGR2GRAY)
    return det_img, gray

  def detect_or_track(self, det_img, gray, frame, run_detector, trackers):
    faces_np = []
    if run_detector:
      h, w = gray.shape[:2]
      faces_np = self.cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(max(48, w // 12), max(48, h // 12))
      )
      if len(faces_np) == 0:
        blur = cv2.bilateralFilter(gray, 5, 50, 50)
        faces_np = self.cascade.detectMultiScale(
          blur, scaleFactor=1.03, minNeighbors=3, flags=cv2.CASCADE_SCALE_IMAGE,
          minSize=(max(40, w // 16), max(40, h // 16))
        )
      trackers = []
      for (x, y, wi, hi) in faces_np:
        tr = self._new_tracker()
        if tr is None: continue
        ok = tr.init(frame, (int(x), int(y), int(wi), int(hi)))
        if ok: trackers.append(tr)
    else:
      updated = []
      for tr in trackers:
        ok, box = tr.update(frame)
        if ok:
          bx, by, bw, bh = box
          faces_np.append((int(bx), int(by), int(bw), int(bh)))
          updated.append(tr)
      trackers = updated
    return faces_np, trackers

  def _new_tracker(self):
    try: return cv2.legacy.TrackerCSRT_create()
    except Exception:
      try: return cv2.legacy.TrackerKCF_create()
      except Exception:
        try: return cv2.legacy.TrackerMOSSE_create()
        except Exception: return None

  def clamp_box(self, b, w, h):
    x = max(0, min(b['x'], w - 1))
    y = max(0, min(b['y'], h - 1))
    bw = max(1, min(b['w'], w - x))
    bh = max(1, min(b['h'], h - y))
    out = {'x': int(x), 'y': int(y), 'w': int(bw), 'h': int(bh)}
    if 'conf' in b: out['conf'] = float(b['conf'])
    return out