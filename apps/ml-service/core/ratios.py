import numpy as np
import math

def analyze_landmarks(p, state):
  lms = p.get('landmarks') or []
  using_z = any('z' in lm for lm in lms)
  video = p.get('video') or {}
  if len(lms) < 474:
    return {'error': 'insufficient landmarks', 'count': len(lms)}

  def d(i, j):
    ax, ay, az = lms[i].get('x',0.0), lms[i].get('y',0.0), lms[i].get('z',0.0)
    bx, by, bz = lms[j].get('x',0.0), lms[j].get('y',0.0), lms[j].get('z',0.0)
    dz = az - bz if (('z' in lms[i]) or ('z' in lms[j])) else 0.0
    return float(((ax-bx)**2 + (ay-by)**2 + dz**2) ** 0.5)

  def midpt(i, j):
    return {
      'x': (lms[i].get('x',0.0) + lms[j].get('x',0.0)) / 2.0,
      'y': (lms[i].get('y',0.0) + lms[j].get('y',0.0)) / 2.0,
      'z': (lms[i].get('z',0.0) + lms[j].get('z',0.0)) / 2.0
    }

  def d_pts(pa, pb):
    ax, ay, az = pa.get('x',0.0), pa.get('y',0.0), pa.get('z',0.0)
    bx, by, bz = pb.get('x',0.0), pb.get('y',0.0), pb.get('z',0.0)
    dz = az - bz if (('z' in pa) or ('z' in pb)) else 0.0
    return float(((ax-bx)**2 + (ay-by)**2 + dz**2) ** 0.5)

  PHI = 1.6180339887
  def close_to(v, target):
    return math.exp(-abs(math.log(v / target)) * 3.5)

  # basic dimensions
  face_w = d(234, 454) or 1e-9
  face_len = d(10, 152)
  left_eye_w = d(33, 133)
  right_eye_w = d(362, 263)
  avg_eye_w = (left_eye_w + right_eye_w) / 2 or 1e-9
  inter_ocular = d(133, 362)
  mouth_w = d(61, 291)
  nose_w = d(49, 279) or 1e-9

  # base ratios
  r_face = face_len / face_w
  r_eyes = inter_ocular / avg_eye_w
  r_mouth_nose = mouth_w / nose_w

  # lip fullness
  mx = (lms[61]['x'] + lms[291]['x']) / 2.0
  my = (lms[61]['y'] + lms[291]['y']) / 2.0
  mz = ((lms[61].get('z',0.0) + lms[291].get('z',0.0)) / 2.0) if (('z' in lms[61]) or ('z' in lms[291])) else None
  def d_point(i, x, y, z=None):
    px, py, pz = lms[i].get('x',0.0), lms[i].get('y',0.0), lms[i].get('z',0.0)
    if z is None: return float(((px-x)**2 + (py-y)**2) ** 0.5)
    return float(((px-x)**2 + (py-y)**2 + (pz-z)**2) ** 0.5)
  upper_vermilion = d_point(13, mx, my, mz)
  lower_vermilion = d_point(14, mx, my, mz)
  lip_fullness_ratio = lower_vermilion / (upper_vermilion or 1e-9)

  # chin to philtrum
  upper_len_phi = d(2, 13)
  lower_len_phi = d(14, 152)
  chin_to_philtrum = lower_len_phi / (upper_len_phi or 1e-9)

  s_face = close_to(r_face, PHI)
  s_eyes = close_to(r_eyes, PHI)
  s_mouth_nose = close_to(r_mouth_nose, PHI)
  s_lip_fullness = close_to(lip_fullness_ratio, PHI)
  s_chin_philtrum = close_to(chin_to_philtrum, 2.0)
  golden_score = round((s_face + s_eyes + s_mouth_nose + s_lip_fullness) / 4 * 100, 1)

  # helper: angle between 3 points
  def angle_between(p1, p2, p3):
    a = np.array([p1.get('x',0.0), p1.get('y',0.0), p1.get('z',0.0)])
    b = np.array([p2.get('x',0.0), p2.get('y',0.0), p2.get('z',0.0)])
    c = np.array([p3.get('x',0.0), p3.get('y',0.0), p3.get('z',0.0)])
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
    cosine = float(np.dot(ba, bc) / denom)
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))

  # jaw and symmetry
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

  # derived midpoints
  brow_mid = midpt(105, 334)
  upper_lip = lms[13]
  bizy = d(234, 454)
  brow_to_upper = d_pts(brow_mid, upper_lip) or 1e-9
  fwhr = bizy / brow_to_upper

  left_eye_center = midpt(33, 133)
  right_eye_center = midpt(263, 362)
  ipd = d_pts(left_eye_center, right_eye_center)

  nasion = lms[168] if 168 < len(lms) else brow_mid
  n_to_upper = d_pts(nasion, upper_lip) or 1e-9
  midface_ratio = ipd / n_to_upper
  esr = ipd / (bizy or 1e-9)

  mouth_center = midpt(13, 14)
  def angle_at(vertex, a, b):
    va = np.array([a.get('x',0.0)-vertex.get('x',0.0), a.get('y',0.0)-vertex.get('y',0.0), a.get('z',0.0)-vertex.get('z',0.0)])
    vb = np.array([b.get('x',0.0)-vertex.get('x',0.0), b.get('y',0.0)-vertex.get('y',0.0), b.get('z',0.0)-vertex.get('z',0.0)])
    denom = (np.linalg.norm(va)*np.linalg.norm(vb)) + 1e-9
    cosv = float(np.dot(va, vb) / denom)
    cosv = float(np.clip(cosv, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))
  eme_angle = angle_at(mouth_center, left_eye_center, right_eye_center)

  # canthal tilt (roll-compensated)
  def seg_angle_deg(med, lat):
    dx = lms[lat].get('x',0.0) - lms[med].get('x',0.0)
    dy_img = lms[lat].get('y',0.0) - lms[med].get('y',0.0)
    ang_img = np.arctan2(dy_img, dx)
    ang_math = -ang_img
    return float(np.degrees(ang_math))

  dx_eyes = lms[263].get('x',0.0) - lms[33].get('x',0.0)
  dy_eyes_img = lms[263].get('y',0.0) - lms[33].get('y',0.0)
  roll_img = np.arctan2(dy_eyes_img, dx_eyes)
  roll_math_deg = float(np.degrees(-roll_img))

  def norm_deg(a):
    while a > 90: a -= 180
    while a < -90: a += 180
    return a

  left_ct = norm_deg(seg_angle_deg(133, 33) - roll_math_deg)
  right_ct = norm_deg(seg_angle_deg(362, 263) - roll_math_deg)
  canthal_tilt = (left_ct + right_ct) / 2.0

  # palpebral fissure ratio
  def fissure_ratio(med, lat, upper, lower):
    length = d(med, lat)
    height = d(upper, lower) or 1e-9
    return length / height
  pfr_left = fissure_ratio(133, 33, 159, 145)
  pfr_right = fissure_ratio(362, 263, 386, 374)
  pfr_avg = (pfr_left + pfr_right) / 2.0

  # symmetry ext
  x_mid = (left_eye_center['x'] + right_eye_center['x']) / 2.0
  pairs = [
    (33, 263), (133, 362), (61, 291), (234, 454),
    (70, 300), (105, 334), (49, 279)
  ]
  errs = []
  for Li, Ri in pairs:
    Lx, Ly = lms[Li]['x'], lms[Li]['y']
    Rx, Ry = lms[Ri]['x'], lms[Ri]['y']
    Lx_mirror = 2.0 * x_mid - Lx
    dx, dy = (Lx_mirror - Rx), (Ly - Ry)
    errs.append((dx*dx + dy*dy) ** 0.5)
  err = float(np.mean(errs)) if errs else 0.0
  err_norm = err / (face_w or 1e-6)
  symmetry_ext = max(0.0, 100.0 - min(100.0, err_norm * 400.0))

  # face shape classifier
  jaw_deg = angle_chin
  r_fw = fwhr
  jaw_diff = abs(angle_left_jaw - angle_right_jaw)
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
    'symmetry_ext': round(symmetry_ext, 1),
    'face_shape': face_shape,
    'video': video if video else None
  }

  return payload