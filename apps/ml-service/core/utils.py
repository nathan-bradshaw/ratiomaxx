import time
def proc_time_ms(t0):
  return round((time.time() - t0) * 1000, 2)