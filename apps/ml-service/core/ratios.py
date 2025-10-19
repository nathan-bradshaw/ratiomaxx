import numpy as np
import math

# -- helpers --
PHI = 1.6180339887

def _get(lms, i):
  return lms[i] if 0 <= i < len(lms) else {'x':0.0,'y':0.0,'z':0.0}

def d_idx(lms, i, j):
  a, b = _get(lms, i), _get(lms, j)
  ax, ay, az = a.get('x',0.0), a.get('y',0.0), a.get('z',0.0)
  bx, by, bz = b.get('x',0.0), b.get('y',0.0), b.get('z',0.0)
  dz = az - bz if (('z' in a) or ('z' in b)) else 0.0
  return float(((ax-bx)**2 + (ay-by)**2 + dz*dz) ** 0.5)

def midpt_idx(lms, i, j):
  a, b = _get(lms, i), _get(lms, j)
  return {
    'x': (a.get('x',0.0) + b.get('x',0.0)) / 2.0,
    'y': (a.get('y',0.0) + b.get('y',0.0)) / 2.0,
    'z': (a.get('z',0.0) + b.get('z',0.0)) / 2.0,
  }

def d_pts(pa, pb):
  ax, ay, az = pa.get('x',0.0), pa.get('y',0.0), pa.get('z',0.0)
  bx, by, bz = pb.get('x',0.0), pb.get('y',0.0), pb.get('z',0.0)
  dz = az - bz if (('z' in pa) or ('z' in pb)) else 0.0
  return float(((ax-bx)**2 + (ay-by)**2 + dz*dz) ** 0.5)

def close_to(v, target):
  import math
  if v <= 0 or target <= 0:
    return 0.0
  return math.exp(-abs(math.log(v / target)) * 3.5)

def angle_between_pts(p1, p2, p3):
  a = np.array([p1.get('x',0.0), p1.get('y',0.0), p1.get('z',0.0)])
  b = np.array([p2.get('x',0.0), p2.get('y',0.0), p2.get('z',0.0)])
  c = np.array([p3.get('x',0.0), p3.get('y',0.0), p3.get('z',0.0)])
  ba = a - b
  bc = c - b
  denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-9
  cosine = float(np.dot(ba, bc) / denom)
  cosine = float(np.clip(cosine, -1.0, 1.0))
  return float(np.degrees(np.arccos(cosine)))

# -- ratio calculators --

def calc_face_length_to_width(lms):
  face_w = d_idx(lms, 234, 454) or 1e-9
  face_len = d_idx(lms, 10, 152)
  return round(face_len / face_w, 3)

def calc_ipd_to_eye_width(lms):
  inter_ocular = d_idx(lms, 133, 362)
  left_eye_w = d_idx(lms, 33, 133)
  right_eye_w = d_idx(lms, 362, 263)
  avg_eye_w = (left_eye_w + right_eye_w) / 2 or 1e-9
  return round(inter_ocular / avg_eye_w, 3)

def calc_mouth_to_nose_width(lms):
  mouth_w = d_idx(lms, 61, 291)
  nose_w = d_idx(lms, 49, 279) or 1e-9
  return round(mouth_w / nose_w, 3)

def calc_lip_fullness_ratio(lms):
  mx = ( _get(lms,61)['x'] + _get(lms,291)['x'] ) / 2.0
  my = ( _get(lms,61)['y'] + _get(lms,291)['y'] ) / 2.0
  mz = None
  if ('z' in _get(lms,61)) or ('z' in _get(lms,291)):
    mz = ( _get(lms,61).get('z',0.0) + _get(lms,291).get('z',0.0) ) / 2.0
  def d_point(i, x, y, z=None):
    p = _get(lms, i)
    px, py, pz = p.get('x',0.0), p.get('y',0.0), p.get('z',0.0)
    if z is None:
      return float(((px-x)**2 + (py-y)**2) ** 0.5)
    return float(((px-x)**2 + (py-y)**2 + (pz-z)**2) ** 0.5)
  upper_vermilion = d_point(13, mx, my, mz)
  lower_vermilion = d_point(14, mx, my, mz)
  return round(lower_vermilion / (upper_vermilion or 1e-9), 3)

def calc_chin_philtrum_ratio(lms):
  upper_len_phi = d_idx(lms, 2, 13)
  lower_len_phi = d_idx(lms, 14, 152)
  return round(lower_len_phi / (upper_len_phi or 1e-9), 3)

def calc_fwhr_variant(lms):
  brow_mid = midpt_idx(lms, 105, 334)
  upper_lip = _get(lms, 13)
  bizy = d_idx(lms, 234, 454)
  brow_to_upper = d_pts(brow_mid, upper_lip) or 1e-9
  return round(bizy / brow_to_upper, 3)

def calc_midface_ratio(lms):
  left_eye_center = midpt_idx(lms, 33, 133)
  right_eye_center = midpt_idx(lms, 263, 362)
  ipd = d_pts(left_eye_center, right_eye_center)
  nasion = _get(lms, 168)
  upper_lip = _get(lms, 13)
  n_to_upper = d_pts(nasion, upper_lip) or 1e-9
  return round(ipd / n_to_upper, 3)

def calc_esr(lms):
  left_eye_center = midpt_idx(lms, 33, 133)
  right_eye_center = midpt_idx(lms, 263, 362)
  ipd = d_pts(left_eye_center, right_eye_center)
  bizy = d_idx(lms, 234, 454) or 1e-9
  return round(ipd / bizy, 3)

def calc_eme_angle(lms):
  mouth_center = midpt_idx(lms, 13, 14)
  left_eye_center = midpt_idx(lms, 33, 133)
  right_eye_center = midpt_idx(lms, 263, 362)
  return round(angle_between_pts(mouth_center, left_eye_center, right_eye_center), 1)

def _seg_angle_deg(lms, med, lat):
  dx = _get(lms, lat).get('x',0.0) - _get(lms, med).get('x',0.0)
  dy_img = _get(lms, lat).get('y',0.0) - _get(lms, med).get('y',0.0)
  ang_img = np.arctan2(dy_img, dx)
  ang_math = -ang_img
  return float(np.degrees(ang_math))

def _norm_deg(a):
  while a > 90: a -= 180
  while a < -90: a += 180
  return a

def calc_canthal_tilt(lms):
  dx_eyes = _get(lms,263).get('x',0.0) - _get(lms,33).get('x',0.0)
  dy_eyes_img = _get(lms,263).get('y',0.0) - _get(lms,33).get('y',0.0)
  roll_img = np.arctan2(dy_eyes_img, dx_eyes)
  roll_math_deg = float(np.degrees(-roll_img))
  left_ct = _norm_deg(_seg_angle_deg(lms,133,33) - roll_math_deg)
  right_ct = _norm_deg(_seg_angle_deg(lms,362,263) - roll_math_deg)
  return round((left_ct + right_ct) / 2.0, 1)

def calc_pfr_avg(lms):
  def fissure_ratio(med, lat, upper, lower):
    length = d_idx(lms, med, lat)
    height = d_idx(lms, upper, lower) or 1e-9
    return length / height
  pfr_left = fissure_ratio(133, 33, 159, 145)
  pfr_right = fissure_ratio(362, 263, 386, 374)
  return round((pfr_left + pfr_right) / 2.0, 2)

def analyze_landmarks(p, state):
  lms = p.get('landmarks') or []
  video = p.get('video') or {}
  if len(lms) < 474:
    return {'error': 'insufficient landmarks', 'count': len(lms)}

  r_face = calc_face_length_to_width(lms)
  r_eyes = calc_ipd_to_eye_width(lms)
  r_mouth_nose = calc_mouth_to_nose_width(lms)

  lip_fullness_ratio = calc_lip_fullness_ratio(lms)

  chin_to_philtrum = calc_chin_philtrum_ratio(lms)

  s_face = close_to(r_face, PHI)
  s_eyes = close_to(r_eyes, PHI)
  s_mouth_nose = close_to(r_mouth_nose, PHI)
  s_lip_fullness = close_to(lip_fullness_ratio, PHI)
  s_chin_philtrum = close_to(chin_to_philtrum, 2.0)
  golden_score = round((s_face + s_eyes + s_mouth_nose + s_lip_fullness) / 4 * 100, 1)

  chin = lms[152]; left_jaw = lms[234]; right_jaw = lms[454]
  angle_chin = angle_between_pts(left_jaw, chin, right_jaw)
  angle_left_jaw = angle_between_pts(chin, left_jaw, lms[10])
  angle_right_jaw = angle_between_pts(chin, right_jaw, lms[338])
  jaw_angle_diff = abs(angle_left_jaw - angle_right_jaw)
  symmetry_score = max(0.0, 100 - (jaw_angle_diff / 180 * 100))

  jawline = {
    'angles': {
      'chin_angle': round(angle_chin, 2),
      'left_jaw_angle': round(angle_left_jaw, 2),
      'right_jaw_angle': round(angle_right_jaw, 2)
    },
    'jaw_symmetry_score': round(symmetry_score, 1)
  }

  brow_mid = midpt_idx(lms, 105, 334)
  upper_lip = lms[13]
  fwhr = calc_fwhr_variant(lms)

  left_eye_center = midpt_idx(lms, 33, 133)
  right_eye_center = midpt_idx(lms, 263, 362)
  ipd = d_pts(left_eye_center, right_eye_center)

  midface_ratio = calc_midface_ratio(lms)
  esr = calc_esr(lms)

  eme_angle = calc_eme_angle(lms)

  canthal_tilt = calc_canthal_tilt(lms)

  pfr_avg = calc_pfr_avg(lms)

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
  err_norm = err / (d_idx(lms, 234, 454) or 1e-6)
  overall_symmetry_score = max(0.0, 100.0 - min(100.0, err_norm * 400.0))

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

  result = {
    'summary': {
      'golden_score_pct': golden_score
    },
    'ratios': {
      'face_length_to_width': round(r_face, 3),
      'ipd_to_eye_width': round(r_eyes, 3),
      'mouth_to_nose_width': round(r_mouth_nose, 3),
      'lower_to_upper_lip_height': round(lip_fullness_ratio, 3),
      'fwhr_variant': round(fwhr, 3),
      'midface_ratio': round(midface_ratio, 3),
      'eye_separation_ratio': round(esr, 3),
      'palpebral_fissure_ratio_avg': round(pfr_avg, 2),
      'chin_over_philtrum': round(chin_to_philtrum, 3)
    },
    'angles': {
      'chin_deg': round(angle_chin, 2),
      'jaw_left_deg': round(angle_left_jaw, 2),
      'jaw_right_deg': round(angle_right_jaw, 2),
      'eme_deg': round(eme_angle, 1),
      'canthal_tilt_deg': round(canthal_tilt, 1)
    },
    'symmetry': {
      'score_pct': round(overall_symmetry_score, 1),
      'jaw_delta_deg': round(jaw_diff, 1),
      'jaw': jawline

    },
    'face_shape': {
      'label': face_shape['label'],
      'features': face_shape['features']
    },
    'jaw': jawline,
    'video': video if video else None
  }

  return result