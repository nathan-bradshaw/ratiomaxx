def _iou(a, b):
  ax1, ay1, ax2, ay2 = a['x'], a['y'], a['x'] + a['w'], a['y'] + a['h']
  bx1, by1, bx2, by2 = b['x'], b['y'], b['x'] + b['w'], b['y'] + b['h']
  inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
  inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
  iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
  inter = iw * ih
  if inter == 0: return 0.0
  area_a = a['w'] * a['h']
  area_b = b['w'] * b['h']
  return inter / float(area_a + area_b - inter)

def _smooth(prev, curr, alpha=0.65):
  return {
    'x': int(prev['x'] * alpha + curr['x'] * (1 - alpha)),
    'y': int(prev['y'] * alpha + curr['y'] * (1 - alpha)),
    'w': int(prev['w'] * alpha + curr['w'] * (1 - alpha)),
    'h': int(prev['h'] * alpha + curr['h'] * (1 - alpha))
  }

class StickyBoxes:
  def __init__(self):
    self.prev_boxes = []
    self.conf_decay = 0.85
    self.conf_boost = 0.30
    self.conf_min = 0.25

  def update(self, curr, shape_hw):
    faces = []
    used_prev = [False] * len(self.prev_boxes)
    for c in curr:
      best_j, best_iou = -1, 0.0
      for j, pbox in enumerate(self.prev_boxes):
        if used_prev[j]: continue
        jacc = _iou(pbox, c)
        if jacc > best_iou:
          best_iou, best_j = jacc, j
      if best_j >= 0 and best_iou >= 0.2:
        sm = _smooth(self.prev_boxes[best_j], c, alpha=0.65)
        prev_conf = float(self.prev_boxes[best_j].get('conf', 0.8))
        sm['conf'] = min(1.0, prev_conf * (1.0 - self.conf_boost) + 1.0 * self.conf_boost)
        faces.append(sm)
        used_prev[best_j] = True
      else:
        c2 = { **c, 'conf': 0.7 }
        faces.append(c2)

    for j, pbox in enumerate(self.prev_boxes):
      if used_prev[j]: continue
      dec = { **pbox }
      dec['conf'] = float(dec.get('conf', 0.7)) * self.conf_decay
      if dec['conf'] >= self.conf_min:
        faces.append(dec)

    faces = [b for b in faces if float(b.get('conf', 0.0)) >= self.conf_min]
    self.prev_boxes = faces
    return faces